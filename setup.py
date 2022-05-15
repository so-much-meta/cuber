"""Package setup script."""
import setuptools

setuptools.setup(
    name='cuber',
    version='0.1',
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'matplotlib',
    ],
    extras_require={
    },
    setup_requires=[],
    tests_require=[],
)