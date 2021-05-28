from distutils.core import setup


def readme():
    """Import the README.md Markdown file and try to convert it to RST format."""
    try:
        import pypandoc
        return pypandoc.convert('README.md', 'rst')
    except(IOError, ImportError):
        with open('README.md') as readme_file:
            return readme_file.read()


setup(
    name='Stock Portfolio Management',
    version='0.1',
    description='Manages a portfolio of stocks using a deep reinforcement learning algorithm',
    long_description=readme(),
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    # Substitute <github_account> with the name of your GitHub account
    url='https://github.com/rajindermavi/stock_portfolio_management',
    author='Rajinder Mavi',   
    author_email='rsmavi.hb@gmail.com',  
    license='MIT',
    packages=[],
)