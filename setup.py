import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='sift',
    version='1.0.0',
    description='Sparse Increment Fine-Tuning method',
    package_dir={"": "src/SIFT"},
    packages=setuptools.find_packages(where="src/SIFT"),
)