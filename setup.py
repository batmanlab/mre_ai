from setuptools import setup, find_packages

def readme():
  with open('README.rst') as f:
    return f.read()

setup(name='mre_ai',
    version='0.1',
    description='Full MRE software and tutorials',
    author='Brian Pollack',
    author_email='brianleepollack@gmail.com',
    license='Pitt',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False)
