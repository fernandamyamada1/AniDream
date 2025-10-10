import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


class ColorDiversityLoss(nn.Module):
    """
    Loss function that encourages a diverse color palette in generated images.
    Extracts dominant colors and maximizes the distance between them in color space.
    """
    def __init__(self, num_colors=16, min_saturation=0.1, min_brightness=0.1, max_brightness=0.9, device='cuda'):
        super(ColorDiversityLoss, self).__init__()
        self.num_colors = num_colors
        self.min_saturation = min_saturation
        self.min_brightness = min_brightness 
        self.max_brightness = max_brightness
        self.device = device
    
    def rgb_to_hsv(self, rgb):
        """
        Convert RGB tensor to HSV color space for better color analysis.
        
        Args:
            rgb: Tensor of shape [..., 3] in range [0, 1]
            
        Returns:
            HSV tensor of shape [..., 3] where H in [0, 1], S in [0, 1], V in [0, 1]
        """
        # Get R, G, B values
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        
        # Calculate max and min of RGB
        max_rgb, _ = torch.max(rgb, dim=-1)
        min_rgb, _ = torch.min(rgb, dim=-1)
        diff = max_rgb - min_rgb
        
        # Calculate H
        h = torch.zeros_like(max_rgb)
        
        # R is max
        mask_r = (max_rgb == r) & (diff != 0)
        h[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6
        
        # G is max
        mask_g = (max_rgb == g) & (diff != 0)
        h[mask_g] = ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 2
        
        # B is max
        mask_b = (max_rgb == b) & (diff != 0)
        h[mask_b] = ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 4
        
        h = h / 6.0
        
        # Calculate S
        s = torch.zeros_like(max_rgb)
        mask_non_zero = max_rgb != 0
        s[mask_non_zero] = diff[mask_non_zero] / max_rgb[mask_non_zero]
        
        # Calculate V
        v = max_rgb
        
        # Stack to get HSV
        hsv = torch.stack([h, s, v], dim=-1)
        return hsv
    
    def extract_dominant_colors(self, x, num_colors):
        """
        Extract dominant colors from image using k-means clustering.
        Filter out low-saturation and extreme brightness colors.
        
        Args:
            x: Input image tensor of shape (B, C, H, W) in range [0, 1]
            num_colors: Number of colors to extract
            
        Returns:
            dominant_colors: Tensor of shape (B, num_colors, 3) in RGB
        """
        B, C, H, W = x.shape
        img_np = x.detach().permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)
        dominant_colors = []

        for batch in range(B):
            pixels = img_np[batch].reshape(-1, 3)
            
            # Convert to HSV for better filtering
            pixels_tensor = torch.tensor(pixels, device=self.device)
            hsv_pixels = self.rgb_to_hsv(pixels_tensor).cpu().numpy()
            
            # Filter by saturation and brightness
            valid_mask = (hsv_pixels[:, 1] >= self.min_saturation) & \
                         (hsv_pixels[:, 2] >= self.min_brightness) & \
                         (hsv_pixels[:, 2] <= self.max_brightness)
            
            valid_pixels = pixels[valid_mask]
            
            if len(valid_pixels) < num_colors:
                # Not enough valid pixels, use original pixels
                valid_pixels = pixels
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=min(num_colors, len(valid_pixels)), 
                           n_init=3, random_state=42)
            kmeans.fit(valid_pixels)
            colors = kmeans.cluster_centers_
            
            # Handle case where we got fewer colors than requested
            if len(colors) < num_colors:
                # Pad with random variations of existing colors
                padding = np.random.normal(0, 0.1, size=(num_colors - len(colors), 3))
                # Take first colors and add variations, clamp to [0, 1]
                base_colors = colors[:min(len(colors), num_colors - len(colors))]
                padded_colors = np.clip(base_colors + padding, 0, 1)
                colors = np.vstack([colors, padded_colors])
            
            dominant_colors.append(torch.from_numpy(colors).float().to(x.device))

        return torch.stack(dominant_colors)  # (B, num_colors, 3)
    
    def color_distance_matrix(self, colors):
        """
        Compute pairwise color distance matrix in perceptual color space.
        
        Args:
            colors: Tensor of shape (B, N, 3) in RGB
            
        Returns:
            distance_matrix: Tensor of shape (B, N, N) of pairwise distances
        """
        B, N, _ = colors.shape
        
        # Reshape for pairwise distance calculation
        colors_expanded_1 = colors.unsqueeze(2)  # (B, N, 1, 3)
        colors_expanded_2 = colors.unsqueeze(1)  # (B, 1, N, 3)
        
        # Compute Euclidean distance in RGB space
        # More sophisticated perceptual metrics could be used here (LAB, DeltaE, etc.)
        diff = colors_expanded_1 - colors_expanded_2  # (B, N, N, 3)
        dist_matrix = torch.sqrt(torch.sum(diff**2, dim=-1))  # (B, N, N)
        
        return dist_matrix
    
    def diversity_score(self, colors):
        """
        Compute diversity score based on pairwise color distances.
        Higher score means more diverse color palette.
        
        Args:
            colors: Tensor of shape (B, N, 3) in RGB
            
        Returns:
            diversity_score: Tensor of shape (B,) indicating color diversity
        """
        dist_matrix = self.color_distance_matrix(colors)
        
        # For each color, find its nearest neighbor distance
        # Remove diagonal (self-distance) by setting to max value
        inf_diag = torch.diag_embed(
            torch.ones_like(dist_matrix[:, :, 0]) * float('inf'),
            dim1=1, dim2=2
        )
        dist_matrix = dist_matrix + inf_diag
        
        # Get minimum distance for each color (nearest neighbor)
        min_dist, _ = torch.min(dist_matrix, dim=2)  # (B, N)
        
        # Diversity score: mean of nearest neighbor distances
        diversity = torch.mean(min_dist, dim=1)  # (B,)
        
        return diversity
    
    def forward(self, x):
        """
        Compute color diversity loss. Higher loss = less diverse palette.
        
        Args:
            x: Input image tensor of shape (B, C, H, W) in range [0, 1]
            
        Returns:
            loss: Color diversity loss (lower values encourage more diverse palettes)
        """
        # Extract dominant colors
        colors = self.extract_dominant_colors(x, self.num_colors)
        
        # Calculate diversity score (higher is better)
        diversity = self.diversity_score(colors)
        
        # Return negative diversity (since we minimize loss)
        return -torch.mean(diversity)


class CellShadingLoss(nn.Module):
    """
    Combined loss function for anime-style cell-shading effect with color diversity.
    """
    def __init__(self, tone_weight=0.0, line_weight=0.5, diversity_weight=0.5, num_tones=16, use_edge_mask=True):
        """
        Combined loss to create a cell-shading effect with diverse color palette.
        
        Args:
            tone_weight (float): Weight for the anime tone quantization loss.
            line_weight (float): Weight for the line art (black outlines) loss.
            diversity_weight (float): Weight for the color diversity loss.
            num_tones (int): Number of color tones for quantization.
            use_edge_mask (bool): Whether to focus Sobel loss on high-gradient regions.
        """
        super(CellShadingLoss, self).__init__()
        
        # Initialize AnimeStyleLoss for quantization
        self.anime_loss = AnimeStyleLoss(num_tones=num_tones)
        
        # Initialize LineArtLoss for black outlines
        self.line_loss = LineArtLoss(threshold=0.3, line_width=3)
        
        # Initialize ColorDiversityLoss for diverse palette
        self.diversity_loss = ColorDiversityLoss(num_colors=num_tones)
        
        # Weights for combining losses
        self.tone_weight = tone_weight
        self.line_weight = line_weight
        self.diversity_weight = diversity_weight
    
    def forward(self, x, target=None):
        """
        Compute combined loss for cell-shading effect with color diversity.
        
        Args:
            x (torch.Tensor): Input image tensor 
                               Shape: [B, C, H, W] in range [0, 1]
            target (torch.Tensor, optional): Target/reference image 
                                             Shape: [B, C, H, W] (unused, kept for API compatibility)
        
        Returns:
            torch.Tensor: Combined loss for cell-shading with diverse colors
        """
        # Compute anime-style quantization loss
        tone_loss = self.anime_loss(x)

        # Compute line art loss for black outlines
        line_loss = self.line_loss(x)
        
        # Compute color diversity loss
        diversity_loss = self.diversity_loss(x)
        
        # Combine all losses
        total_loss = (self.tone_weight * tone_loss +  
                     self.line_weight * line_loss +
                     self.diversity_weight * diversity_loss)
        
        return total_loss
        
        
class AnimeStyleLoss(nn.Module):
    def __init__(self, num_tones=16, color_var=0.3, hue_diversity_weight=2.0):
        super(AnimeStyleLoss, self).__init__()
        self.num_tones = num_tones
        self.color_var = color_var
        self.hue_diversity_weight = hue_diversity_weight  # How strongly to prioritize hue diversity

    def extract_dominant_colors(self, x, x_colors=10):
        """
        Extract x dominant colors (excluding near-black/white) via k-means.
        Args:
            x: Input image (B, C, H, W) in [0, 1]
            x_colors: Number of base colors to extract
        Returns:
            dominant_colors: (x_colors, 3) RGB colors
        """
        B, C, H, W = x.shape
        img_np = x.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)
        dominant_colors = []

        for batch in range(B):
            pixels = img_np[batch].reshape(-1, 3)
            
            # Exclude near-black/white pixels
            #not_black = np.sum(pixels > 0.1, axis=1) > 0
            #not_white = np.sum(pixels < 0.9, axis=1) > 0
            #valid_pixels = pixels[not_black & not_white]
            gray_threshold=0.5
            diff = np.max(pixels, axis=1) - np.min(pixels, axis=1)
            not_gray = diff > gray_threshold
            valid_pixels = pixels[not_gray]

            if len(valid_pixels) == 0:
                # Fallback: random colors if no valid pixels
                colors = np.random.rand(x_colors, 5)
            else:
                # Use k-means to find dominant colors
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=x_colors, n_init=3)
                kmeans.fit(valid_pixels)
                colors = kmeans.cluster_centers_

            dominant_colors.append(torch.from_numpy(colors).float().to(x.device))

        return torch.stack(dominant_colors)  # (B, x_colors, 3)


    def generate_color_variations(self, colors):
        """
        Generate lighter/darker variations for each base color.
        Args:
            colors: (B, x_colors, 3) base colors
        Returns:
            palette: (B, x_colors * 3 + 2, 3) includes base, variations, black, white
        """
        B, x_colors, _ = colors.shape
        palette = []

        for batch in range(B):
            # Base colors
            base = colors[batch]  # (x_colors, 3)

            # Lighter variations (additive)
            lighter = torch.clamp(base + self.color_var, 0, 1)

            # Darker variations (subtractive)
            darker = torch.clamp(base - self.color_var, 0, 1)

            # Combine base + variations
            combined = torch.cat([base, lighter, darker], dim=0)  # (3 * x_colors, 3)

            # Add pure black and white
            black = torch.tensor([0, 0, 0], device=colors.device).unsqueeze(0)
            white = torch.tensor([1, 1, 1], device=colors.device).unsqueeze(0)
            combined = torch.cat([combined, black, white], dim=0)  # (3 * x_colors + 2, 3)

            palette.append(combined)

        return torch.stack(palette)  # (B, 3 * x_colors + 2, 3)

    def quantize_colors(self, x, palette):
        """
        Quantize image to the nearest color in the palette.
        Args:
            x: Input image (B, C, H, W)
            palette: (B, num_colors, 3) available colors
        Returns:
            quantized: Quantized image (B, C, H, W)
        """
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, -1, C)  # (B, H*W, 3)
        quantized = torch.zeros_like(x_flat)

        for batch in range(B):
            # Compute L2 distance to all palette colors
            dist = torch.cdist(x_flat[batch], palette[batch])  # (H*W, num_colors)
            # Find closest color for each pixel
            closest = torch.argmin(dist, dim=1)
            quantized[batch] = palette[batch][closest]

        return quantized.reshape(B, H, W, C).permute(0, 3, 1, 2)

    def forward(self, x):
        """
        Compute anime-style loss with color variations.
        Args:
            x: Input image (B, C, H, W) in [0, 1]
        Returns:
            loss: Quantization loss
        """
        # Calculate x_colors such that: 3 * x_colors + 2 (b/w) <= num_tones
        x_colors = (self.num_tones - 2) // 3

        # Extract dominant colors (excluding black/white)
        dominant_colors = self.extract_dominant_colors(x, x_colors)

        # Generate base + variations + black/white palette
        palette = self.generate_color_variations(dominant_colors)

        # Quantize image to the palette
        quantized = self.quantize_colors(x, palette)

        # L1 loss between original and quantized
        return F.l1_loss(x, quantized)



class LineArtLoss(nn.Module):
    """
    Additional loss to encourage black outlines between color regions using Sobel operator for edge detection.
    """
    def __init__(self, threshold=0.5, line_width=1):
        super(LineArtLoss, self).__init__()
        self.threshold = threshold
        self.line_width = line_width
        
        # Sobel kernels for edge detection
        self.sobel_x_kernel = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.sobel_y_kernel = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
    
    def forward(self, x):
        """
        Compute line art loss that encourages dark borders between color regions using Sobel edge detection.
        
        Args:
            x (torch.Tensor): Input image (B, C, H, W)
            
        Returns:
            torch.Tensor: Line art loss
        """
        # Convert to grayscale
        if x.size(1) == 3:
            x_gray = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]
        else:
            x_gray = x
        
        # Convert to BW (binary) using threshold
        #hreshold = 0.5  # Adjust threshold as needed
        #_gray = (x_gray > threshold).float()  # Binary thresholding
        
        # Pad input for applying kernels (with reflection padding to avoid boundary issues)
        x_pad = F.pad(x_gray, (1, 1, 1, 1), mode='reflect')
        
        # Apply Sobel X and Sobel Y filters to get gradient in both directions
        grad_x = F.conv2d(x_pad, self.sobel_x_kernel.to(x.device), stride=1, padding=0)
        grad_y = F.conv2d(x_pad, self.sobel_y_kernel.to(x.device), stride=1, padding=0)
        
        # Compute the magnitude of the gradient (edge strength)
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2) * 0.3
        
        # Get edge regions (high gradient magnitude values)
        edge_regions = grad_magnitude > self.threshold
        
        # We want these edge regions to be dark (close to 0)
        # So we penalize high values in edge regions
        edge_values = x_gray * edge_regions.float()
        return edge_values.mean()
