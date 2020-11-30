"""Minimal setup file"""

from setuptools import setup, find_packages

setup(
    name='dereste_event_point_ranking_1_app',
    version='0.1.0',
    license='proprietary',
    description='Module Experiment',
    
    author='kazuya.minakuchi',
    author_email='mnkckzy@gmail.com',
    url='http://github.com/kzy611/dereste_event_point_ranking_1_app',
    
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"],
)
