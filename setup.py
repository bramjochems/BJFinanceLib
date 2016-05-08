from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='BJFinanceLib',
      version='0.1',
      description='Collection of finance utilities',
      url='',
      author='Bram Jochems',
      author_email='bram.jochems@gmail.com',
      license='MIT',
      packages=['BJFinanceLib'],
	  keywords='finance, quant, black-scholes, options',
	  install_requires=[
          'numpy',
	    'scipy',
          'pandas'
      ],
      zip_safe=False,
	  test_suite='nose.collector',
      tests_require=['nose'])