from setuptools import setup, find_packages

setup(
    name='alphamodel',
    version='0.0.1',
    author='Razvan Oprisor',
    author_email='razvan.oprisor@mail.utoronto.ca',
    packages=find_packages(),
    package_dir={'alphamodel': 'alphamodel'},
    package_data={'': ['*.csv', '*.yml']},
    include_package_data=True,
    license='Apache',
    zip_safe=False,
    description='Financial Alpha Modeling Package',
    install_requires=["pandas",
                      "numpy",
                      "matplotlib",
                      "cvxpy>=1.0.6",
                      "cvxportfolio"],
)
