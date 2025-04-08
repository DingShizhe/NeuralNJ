from setuptools import setup, Extension, find_packages, Command
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
import sys
import setuptools
import os
import subprocess
import shutil
import glob

class CustomBuildExt(build_ext):
    def run(self):
        # 确保所需的第三方 C 库目录存在

        if True:
            
            if os.path.exists("pll-modules"):
                shutil.rmtree("pll-modules")

            subprocess.check_call(['git', 'clone', '--recursive', 'https://github.com/ddarriba/pll-modules'])

            os.chdir("pll-modules")

            subprocess.check_call(['./install-with-libpll.sh', 'install'])
            subprocess.check_call(['mkdir', "build"])
            os.chdir("build")
            subprocess.check_call(['cmake', "-DBUILD_PLLMODULES_SHARED=ON", ".."])
            subprocess.check_call(['make', "-j8"])

            os.chdir('../../')

        if True:

            if os.path.exists('./raxml-ng'):
                shutil.rmtree('./raxml-ng')

            subprocess.check_call(['git', 'clone', '--recursive', 'https://github.com/amkozlov/raxml-ng.git'])

            os.chdir("raxml-ng")

            subprocess.check_call(['mkdir', "build"])
            os.chdir("build")
            subprocess.check_call(['cmake', "..", "-DBUILD_AS_LIBRARY=ON"])
            subprocess.check_call(['make', "-j8"])
            
            os.chdir('../../')

        raxml_lib_dir = './build_raxmllib'
        pll_lib_dir = './build_plllib'

        os.makedirs(raxml_lib_dir, exist_ok=True)
        os.makedirs(pll_lib_dir, exist_ok=True)

        from pathlib import Path
        for lib_path in list(Path('./pll-modules/build').glob('**/*.so*')):
            shutil.copy(lib_path, pll_lib_dir)
        for lib_path in list(Path('./pll-modules/install/lib').glob('**/*.so*')):
            shutil.copy(lib_path, pll_lib_dir)
        
        for lib_path in list(Path('./raxml-ng/build').glob('**/*.so*')):
            shutil.copy(lib_path, os.path.join(raxml_lib_dir, "libraxml.so"))

        build_ext.run(self)



class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked."""

    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'raxmlpy.cpp_binding',
        ['cpp/raxmlpy.cpp'],
        include_dirs=[
            get_pybind_include(),
            "raxml-ng/src",
            "raxml-ng/build/localdeps/include/",
        ],
        libraries=[
            'raxml',
            'pllmodutil',
            'pllmodtree',
            'pllmodoptimize',
            'pllmodmsa',
            'pllmodbinary',
            'pllmodalgorithm',
            'pll',
            'pll_algorithm',
            'pll_optimize',
            'pll_tree',
            'pll_util',
            'm'
        ],
        library_dirs = [
            './build_raxmllib',
            './build_plllib',
        ],
        extra_compile_args=['-g', '-O3', '-Wall', '-Wsign-compare', '-Wno-unused-function'],
        language='c++'
    ),
]

setup(
    name='raxmlpy',
    version='0.1',
    author='Your Name',
    author_email='your.email@example.com',
    description='A test module for C++ extension',
    long_description='',
    ext_modules=ext_modules,
    packages=find_packages(),
    cmdclass={
        'build_ext': CustomBuildExt
    },
    zip_safe=False,
)
