from setuptools import setup

setup(
    name='MPRL',
    version='1.0.0',
    packages=['mprl', 'mprl.rl', 'mprl.rl.agent', 'mprl.rl.critic',
              'mprl.rl.policy', 'mprl.rl.sampler', 'mprl.rl.projection',
              'mprl.util'],
    url='',
    license='',
    author='',
    author_email='',
    description='',
    install_requires=[
        'numpy',
        'matplotlib',
        'torch',
        'addict',
        'wandb',
        'natsort',
        'tabulate',
    ],
)
