from setuptools import find_packages, setup
setup(
    name='pytheas',
    packages=find_packages(include=['pytheas','evaluation']),
    version='0.0.1',
    description='CSV table annotation tool',
    author='Christina Christodoulakis',
    license='MIT',
    install_requires=[ 'cchardet>=2.1.4',
			'dotmap',
                        'Unidecode>=1.1.1',
                        'pandas>=1.1.3',
			'sqlalchemy',
                        'inflection>=0.3.1',
                        'langdetect>=1.0.7',
                        'more-itertools>=7.2.0',
                        'numpy>=1.17.2',
                        'nltk>=3.4.5',
                        'requests>=2.22.0',
                        'stringutils>=0.3.0',
                        'python-string-utils>=0.6.0',
                        'psycopg2',
			'tqdm>=4.36.1',
                        'sortedcontainers>=2.1.0'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests'
)
# python setup.py sdist bdist_wheel
