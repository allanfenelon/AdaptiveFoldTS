from setuptools import setup, find_packages

setup(
    name="AdaptiveFoldTS",
    version="0.1.0",
    author="Allan",
    author_email="allanpereira.fenelon@example.com",
    description="Validação Cruzada Adaptativa para Séries Temporais com Priorização Inteligente de Folds",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/allanfenelon/AdaptiveFoldTS",  # alterar para seu repositório
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
    ],
)
