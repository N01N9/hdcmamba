from setuptools import setup, find_packages

setup(
    name="hdcmamba",
    version="0.3.0",
    description="HdcMamba: Ultra-optimized SSM block with fused Triton kernels and O(D^2) VRAM scaling",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1",
        "triton>=2.2",
    ],
    extras_require={
        "dev": ["pytest", "einops"],
    },
    python_requires=">=3.10",
)
