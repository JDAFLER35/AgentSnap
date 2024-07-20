# setup.py

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="advanced_ai_agent",
    version="1.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi==0.68.0",
        "uvicorn==0.15.0",
        "pydantic==1.8.2",
        "nltk==3.6.3",
        "numpy==1.21.2",
        "scikit-learn==0.24.2",
        "pyyaml==5.4.1",
        "spacy==3.1.3",
        "transformers==4.11.3",
        "torch==1.9.0",
        "allennlp==2.7.0",
        "SpeechRecognition==3.8.1",
        "pyttsx3==2.90",
        "pyaudio==0.2.11",
    ],
    extras_require={
        "dev": [
            "pytest==6.2.5",
            "black==21.9b0",
            "isort==5.9.3",
            "mypy==0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai_agent=src.main:main",
        ],
    },
    author="Mr. Dafler",
    author_email="musicisforthesoul420@gmail.com",
    description="An advanced AI agent with natural language processing and voice interaction capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/advanced_ai_agent",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.7',
)