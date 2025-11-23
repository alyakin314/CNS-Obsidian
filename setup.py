import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

# Read in requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as req_file:
    REQUIREMENTS = [
        line.strip()
        for line in req_file
        if line.strip() and not line.startswith("#")
    ]

setuptools.setup(
    name="cns_obsidian",
    version="0.1.0",
    author="Anton Alyakin (@alyakin314)",
    author_email="alyakin314@gmail.com",
    description="Training neurosurgical multimodal experts",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/nyuolab/CNS-Obsidian",
    install_requires=REQUIREMENTS, 
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)
