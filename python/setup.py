# Copyright 2022 EdgeCortix Inc.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, exec-used
"""Setup TVM package."""
import os
import shutil
import sys
import sysconfig
import platform
import glob
import pathlib
import re

from setuptools import find_packages
from setuptools.dist import Distribution

# need to use distutils.core for correct placement of cython dll
if "--inplace" in sys.argv:
    from distutils.core import setup
    from distutils.extension import Extension
else:
    from setuptools import setup
    from setuptools.extension import Extension

CURRENT_DIR = os.path.dirname(__file__)
FFI_MODE = os.environ.get("TVM_FFI", "auto")
CONDA_BUILD = os.getenv("CONDA_BUILD") is not None


def get_lib_path():
    """Get library path, name and version"""
    # We can not import `libinfo.py` in setup.py directly since __init__.py
    # Will be invoked which introduces dependencies
    libinfo_py = os.path.join(CURRENT_DIR, "./tvm/_ffi/libinfo.py")
    libinfo = {"__file__": libinfo_py}
    exec(compile(open(libinfo_py, "rb").read(), libinfo_py, "exec"), libinfo, libinfo)
    version = libinfo["__version__"]
    if not CONDA_BUILD:
        lib_path = libinfo["find_lib_path"]()
        libs = [lib_path[0]]
        if libs[0].find("runtime") == -1:
            for name in lib_path[1:]:
                if name.find("runtime") != -1:
                    libs.append(name)
                    break
    else:
        libs = None
    return libs, version


def git_describe_version(original_version):
    """Get git describe version."""
    ver_py = os.path.join(CURRENT_DIR, "..", "version.py")
    libver = {"__file__": ver_py}
    exec(compile(open(ver_py, "rb").read(), ver_py, "exec"), libver, libver)
    _, gd_version = libver["git_describe_version"]()
    if gd_version != original_version and "--inplace" not in sys.argv:
        print("Use git describe based version %s" % gd_version)
    return gd_version


LIB_LIST, __version__ = get_lib_path()
__version__ = git_describe_version(__version__)


def config_cython():
    """Try to configure cython and return cython configuration"""
    if FFI_MODE not in ("cython"):
        if os.name == "nt" and not CONDA_BUILD:
            print("WARNING: Cython is not supported on Windows, will compile without cython module")
            return []
        sys_cflags = sysconfig.get_config_var("CFLAGS")
        if sys_cflags and "i386" in sys_cflags and "x86_64" in sys_cflags:
            print("WARNING: Cython library may not be compiled correctly with both i386 and x64")
            return []
    try:
        from Cython.Build import cythonize

        # from setuptools.extension import Extension
        if sys.version_info >= (3, 0):
            subdir = "_cy3"
        else:
            subdir = "_cy2"
        ret = []
        path = "tvm/_ffi/_cython"
        extra_compile_args = ["-std=c++14", "-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>"]
        if os.name == "nt":
            library_dirs = ["tvm", "../build/Release", "../build"]
            libraries = ["tvm"]
            extra_compile_args = None
            # library is available via conda env.
            if CONDA_BUILD:
                library_dirs = [os.environ["LIBRARY_LIB"]]
        else:
            library_dirs = None
            libraries = None

        for fn in os.listdir(path):
            if not fn.endswith(".pyx"):
                continue
            ret.append(
                Extension(
                    "tvm._ffi.%s.%s" % (subdir, fn[:-4]),
                    ["tvm/_ffi/_cython/%s" % fn],
                    include_dirs=[
                        "../include/",
                        "../3rdparty/dmlc-core/include",
                        "../3rdparty/dlpack/include",
                    ],
                    extra_compile_args=extra_compile_args,
                    library_dirs=library_dirs,
                    libraries=libraries,
                    language="c++",
                )
            )
        return cythonize(ret, compiler_directives={"language_level": 3})
    except ImportError as error:
        if FFI_MODE == "cython":
            raise error
        print("WARNING: Cython is not installed, will compile without cython module")
        return []


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


include_libs = False
wheel_include_libs = False
if not CONDA_BUILD:
    if "bdist_wheel" in sys.argv:
        wheel_include_libs = True
    else:
        include_libs = True

setup_kwargs = {}


def get_package_data_files():
    # Relay standard libraries
    return ["relay/std/prelude.rly", "relay/std/core.rly"]

class _MeraTvmPackageType:
    def __init__(self, name):
        self.name = name

MODES = {
    "host-only" : _MeraTvmPackageType('host-only'),
    "runtime" : _MeraTvmPackageType('runtime'),
    "full" : _MeraTvmPackageType('full')
}

dist_mode = os.environ["MERA_TVM_DIST_MODE"]
if dist_mode not in MODES:
    raise ValueError(f'Unknown distribution mode {dist_mode}. Can only be ({", ".join(MODES.keys())})')
tvm_mode : _MeraTvmPackageType = MODES[dist_mode]

if not os.environ.get("MERA_TVM_DIST_MODE", None):
    raise ValueError(f'Missing environment variable MERA_TVM_DIST_MODE ({", ".join(MODES.keys())})')
if not os.environ.get("MERA_HOME", None):
    raise ValueError(f'Missing environment variable MERA_HOME')
_IS_ARM = any([bool(re.match(r"--plat-name=.*_aarch64", a)) for a in sys.argv])
_arch = 'aarch64' if _IS_ARM else 'x86'

MERA_DNA_LIBS = glob.glob(f'{os.environ["MERA_HOME"]}/lib/libmeradna-{tvm_mode.name}.{_arch}*')

# For bdist_wheel only
if wheel_include_libs:
    with open("MANIFEST.in", "w") as fo:
        for path in LIB_LIST:
            shutil.copy(path, os.path.join(CURRENT_DIR, "tvm"))
            _, libname = os.path.split(path)
            fo.write("include tvm/%s\n" % libname)
        for path in MERA_DNA_LIBS:
            shutil.copy(path, os.path.join(CURRENT_DIR, "tvm"))
            fo.write(f"include tvm/{os.path.basename(path)}\n")
        for pkg_data in get_package_data_files():
            fo.write(f"include tvm/{pkg_data}\n")
    setup_kwargs = {"include_package_data": True}

if include_libs:
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    for i, path in enumerate(LIB_LIST):
        LIB_LIST[i] = os.path.relpath(path, curr_path)
    setup_kwargs = {"include_package_data": True, "data_files": [("tvm", LIB_LIST)]}


# Temporarily add this directory to the path so we can import the requirements generator
# tool.
sys.path.insert(0, os.path.dirname(__file__))
import gen_requirements

sys.path.pop(0)

requirements = gen_requirements.join_requirements()
extras_require = {
    piece: deps for piece, (_, deps) in requirements.items() if piece not in ("all", "core")
}


PKG_NAME = f'mera-tvm-{tvm_mode.name}'

mera_tvm_version = __version__
version_local = os.environ.get("MERA_TVM_LOCAL_VERSION", None)
if version_local:
    mera_tvm_version += f'+{version_local}'

setup(
    name=PKG_NAME,
    version=mera_tvm_version,
    description="Mera TVM: An End to End Tensor IR/DSL Stack for Deep Learning Systems, adapted to Mera (https://github.com/Edgecortix-Inc/mera) environment.",
    zip_safe=False,
    entry_points={"console_scripts": ["tvmc = tvm.driver.tvmc.main:main"]},
    install_requires=requirements["core"][1],
    extras_require=extras_require,
    packages=find_packages(),
    package_dir={PKG_NAME: "tvm"},
    package_data={PKG_NAME: get_package_data_files()},
    distclass=BinaryDistribution,
    url="https://github.com/Edgecortix-Inc/mera-tvm",
    ext_modules=config_cython(),
    **setup_kwargs,
)


if wheel_include_libs:
    # Wheel cleanup
    os.remove("MANIFEST.in")
    for path in LIB_LIST:
        _, libname = os.path.split(path)
        os.remove("tvm/%s" % libname)
    for path in MERA_DNA_LIBS:
        os.remove(f"tvm/{os.path.basename(path)}")
