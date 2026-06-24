import torch


def generate_noisy_landscape(H=20, W=20, with_loop=False, z_scale=2.0):
    """Generates a landscape with noise to test persistence simplification.

    Args:
        H (int): Height of the generated landscape.
        W (int): Width of the generated landscape.
        with_loop (bool): If True, adds an extra topological feature (peak + slope)
                          that creates more complex manifold topology.
        z_scale (float): Scaling factor applied to the base landscape before adding noise.
                         (Default is 2.0. Use 1.0 for backward compatibility with older benchmarks).
    """
    y, x = torch.meshgrid(torch.linspace(-2, 2, H), torch.linspace(-2, 2, W), indexing="ij")

    peak1 = torch.exp(-((x - 1) ** 2 + (y) ** 2) / 0.5)
    peak2 = torch.exp(-((x + 1) ** 2 + (y) ** 2) / 0.5)
    valley1 = -0.5 * torch.exp(-((x) ** 2 + (y - 1) ** 2) / 0.5)
    valley2 = -1.0 * torch.exp(-((x) ** 2 + (y + 1) ** 2) / 0.5)

    z = peak1 + peak2 + valley1 + valley2
    if with_loop:
        peak3 = 1.0 * torch.exp(-(((x) ** 2 + (y + 1) ** 2) ** 3) / 0.5)
        slope = -y / 4
        z += peak3 + slope

    noise = (torch.rand(H, W) - 0.5) * 0.1
    z = z * z_scale + noise

    return z.to(torch.float32)
