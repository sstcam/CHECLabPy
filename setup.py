from setuptools import setup, find_packages

PACKAGENAME = "CHECLabPy"
DESCRIPTION = "Python scripts for reduction and analysis of CHEC lab data"
AUTHOR = "Jason J Watson"
AUTHOR_EMAIL = "jason.watson@physics.ox.ac.uk"

version = {}
with open("CHECLabPy/version.py") as fp:
    exec(fp.read(), version)

setup(
    name=PACKAGENAME,
    packages=find_packages(),
    version=version['__version__'],
    description=DESCRIPTION,
    license='BSD3',
    install_requires=[
        'astropy',
        'scipy',
        'numpy',
        'matplotlib',
        'tqdm',
        'pandas>=0.21.0',
        'iminuit',
        'numba',
        'PyYAML',
        'packaging',
    ],
    setup_requires=['pytest-runner', ],
    tests_require=['pytest', ],
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    package_data={
        '': ['data/*'],
    },
    entry_points={'console_scripts': [
        'extract_dl1 = CHECLabPy.scripts.extract_dl1:main',
        'generate_dl1_config = CHECLabPy.scripts.generate_dl1_config:main',
        'extract_spe = CHECLabPy.scripts.extract_spe:main',
    ]}
)
