from setuptools import setup, find_packages

setup(
    name="epiagent",
    version="0.0.1",
    description="Foundation model for single-cell epigenomic data.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Xiaoyang Chen",
    author_email="xychen20@mails.tsinghua.edu.cn",
    url="https://github.com/xy-chen16/EpiAgent",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "scanpy",
        "torch>=2.0.0",  # PyTorch: Installation details in README.md
        "transformers",
        "anndata",
        "flash-attn>=2.5.7"  # Flash-Attention: Installation details in README.md
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    extras_require={
        "gpu": ["torch>=2.0.0", "flash-attn>=2.5.7"],
    },
)
