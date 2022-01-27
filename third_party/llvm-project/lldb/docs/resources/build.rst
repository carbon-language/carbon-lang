Building
========

.. contents::
   :local:

Getting the Sources
-------------------

Please refer to the `LLVM Getting Started Guide
<https://llvm.org/docs/GettingStarted.html#getting-started-with-llvm>`_ for
general instructions on how to check out the LLVM monorepo, which contains the
LLDB sources.

Git browser: https://github.com/llvm/llvm-project/tree/main/lldb

Preliminaries
-------------

LLDB relies on many of the technologies developed by the larger LLVM project.
In particular, it requires both Clang and LLVM itself in order to build. Due to
this tight integration the Getting Started guides for both of these projects
come as prerequisite reading:

* `LLVM <https://llvm.org/docs/GettingStarted.html>`_
* `Clang <http://clang.llvm.org/get_started.html>`_

The following requirements are shared on all platforms.

* `CMake <https://cmake.org>`_
* `Ninja <https://ninja-build.org>`_ (strongly recommended)

If you want to run the test suite, you'll need to build LLDB with Python
scripting support.

* `Python <http://www.python.org/>`_
* `SWIG <http://swig.org/>`_ 3 or later.

Optional Dependencies
*********************

Although the following dependencies are optional, they have a big impact on
LLDB's functionality. It is strongly encouraged to build LLDB with these
dependencies enabled.

By default they are auto-detected: if CMake can find the dependency it will be
used. It is possible to override this behavior by setting the corresponding
CMake flag to ``On`` or ``Off`` to force the dependency to be enabled or
disabled. When a dependency is set to ``On`` and can't be found it will cause a
CMake configuration error.

+-------------------+------------------------------------------------------+--------------------------+
| Feature           | Description                                          | CMake Flag               |
+===================+======================================================+==========================+
| Editline          | Generic line editing, history, Emacs and Vi bindings | ``LLDB_ENABLE_LIBEDIT``  |
+-------------------+------------------------------------------------------+--------------------------+
| Curses            | Text user interface                                  | ``LLDB_ENABLE_CURSES``   |
+-------------------+------------------------------------------------------+--------------------------+
| LZMA              | Lossless data compression                            | ``LLDB_ENABLE_LZMA``     |
+-------------------+------------------------------------------------------+--------------------------+
| Libxml2           | XML                                                  | ``LLDB_ENABLE_LIBXML2``  |
+-------------------+------------------------------------------------------+--------------------------+
| Python            | Python scripting                                     | ``LLDB_ENABLE_PYTHON``   |
+-------------------+------------------------------------------------------+--------------------------+
| Lua               | Lua scripting                                        | ``LLDB_ENABLE_LUA``      |
+-------------------+------------------------------------------------------+--------------------------+

Depending on your platform and package manager, one might run any of the
commands below.

::

  $ yum install libedit-devel libxml2-devel ncurses-devel python-devel swig
  $ sudo apt-get install build-essential swig python3-dev libedit-dev libncurses5-dev
  $ pkg install swig python
  $ pkgin install swig python36 cmake ninja-build
  $ brew install swig cmake ninja

Note that there's an `incompatibility
<https://github.com/swig/swig/issues/1321>`_ between Python version 3.7 and later
and swig versions older than 4.0.0 which makes builds of LLDB using debug
versions of python unusable. This primarily affects Windows, as debug builds of
LLDB must use debug python as well.


Windows
*******

* Visual Studio 2017.
* The latest Windows SDK.
* The Active Template Library (ATL).
* `GnuWin32 <http://gnuwin32.sourceforge.net/>`_ for CoreUtils and Make.
* `Python 3 <https://www.python.org/downloads/windows/>`_.  Make sure to (1) get
  the x64 variant if that's what you're targetting and (2) install the debug
  library if you want to build a debug lldb.
* `Python Tools for Visual Studio
  <https://github.com/Microsoft/PTVS/releases>`_. If you plan to debug test
  failures or even write new tests at all, PTVS is an indispensable debugging
  extension to VS that enables full editing and debugging support for Python
  (including mixed native/managed debugging).

The steps outlined here describes how to set up your system and install the
required dependencies such that they can be found when needed during the build
process. They only need to be performed once.

#. Install Visual Studio with the Windows SDK and ATL components.
#. Install GnuWin32, making sure ``<GnuWin32 install dir>\bin`` is added to
   your PATH environment variable. Verify that utilities like ``dirname`` and
   ``make`` are available from your terminal.
#. Install SWIG for Windows, making sure ``<SWIG install dir>`` is added to
   your PATH environment variable. Verify that ``swig`` is available from your
   terminal.
#. Register the Debug Interface Access DLLs with the Registry from a privileged
   terminal.

::

> regsvr32 "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\DIA SDK\bin\msdia140.dll"
> regsvr32 "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\DIA SDK\bin\amd64\msdia140.dll"

Any command prompt from which you build LLDB should have a valid Visual Studio
environment setup. This means you should run ``vcvarsall.bat`` or open an
appropriate Visual Studio Command Prompt corresponding to the version you wish
to use.

macOS
*****

* To use the in-tree debug server on macOS, lldb needs to be code signed. For
  more information see :ref:`CodeSigning` below.
* If you are building both Clang and LLDB together, be sure to also check out
  libc++, which is a required for testing on macOS.

Building LLDB with CMake
------------------------

The LLVM project is migrating to a single monolithic repository for LLVM and
its subprojects. This is the recommended way to build LLDB. Check out the
source-tree with git:

::

  $ git clone https://github.com/llvm/llvm-project.git

CMake is a cross-platform build-generator tool. CMake does not build the
project, it generates the files needed by your build tool. The recommended
build tool for LLVM is Ninja, but other generators like Xcode or Visual Studio
may be used as well. Please also read `Building LLVM with CMake
<https://llvm.org/docs/CMake.html>`_.

Regular in-tree builds
**********************

Create a new directory for your build-tree. From there run CMake and point it
to the ``llvm`` directory in the source-tree:

::

  $ cmake -G Ninja -DLLVM_ENABLE_PROJECTS="clang;lldb" [<cmake options>] path/to/llvm-project/llvm

We used the ``LLVM_ENABLE_PROJECTS`` option here to tell the build-system which
subprojects to build in addition to LLVM (for more options see
:ref:`CommonCMakeOptions` and :ref:`CMakeCaches`). Parts of the LLDB test suite
require ``lld``. Add it to the list in order to run all tests. Once CMake is done,
run ninja to perform the actual build. We pass ``lldb`` here as the target, so
it only builds what is necessary to run the lldb driver:

::

  $ ninja lldb

Standalone builds
*****************

This is another way to build LLDB. We can use the same source-tree as we
checked out above, but now we will have multiple build-trees:

* the main build-tree for LLDB in ``/path/to/lldb-build``
* one or more provided build-trees for LLVM and Clang; for simplicity we use a
  single one in ``/path/to/llvm-build``

Run CMake with ``-B`` pointing to a new directory for the provided
build-tree\ :sup:`1` and the positional argument pointing to the ``llvm``
directory in the source-tree. Note that we leave out LLDB here and only include
Clang. Then we build the ``ALL`` target with ninja:

::

  $ cmake -B /path/to/llvm-build -G Ninja \
          -DLLVM_ENABLE_PROJECTS=clang \
          [<more cmake options>] /path/to/llvm-project/llvm
  $ ninja

Now run CMake a second time with ``-B`` pointing to a new directory for the
main build-tree and the positional argument pointing to the ``lldb`` directory
in the source-tree. In order to find the provided build-tree, the build system
looks for the path to its CMake modules in ``LLVM_DIR``. If you use a separate
build directory for Clang, remember to pass its module path via ``Clang_DIR``
(CMake variables are case-sensitive!):

::

  $ cmake -B /path/to/lldb-build -G Ninja \
          -DLLVM_DIR=/path/to/llvm-build/lib/cmake/llvm \
          [<more cmake options>] /path/to/llvm-project/lldb
  $ ninja lldb

.. note::

   #. The ``-B`` argument was undocumented for a while and is only officially
      supported since `CMake version 3.14
      <https://cmake.org/cmake/help/v3.14/release/3.14.html#command-line>`_

.. _CommonCMakeOptions:

Common CMake options
********************

Following is a description of some of the most important CMake variables which
you are likely to encounter. A variable FOO is set by adding ``-DFOO=value`` to
the CMake command line.

If you want to debug the lldb that you're building -- that is, build it with
debug info enabled -- pass two additional arguments to cmake before running
ninja:

::

  $ cmake -G Ninja \
      -DLLDB_EXPORT_ALL_SYMBOLS=1 \
      -DCMAKE_BUILD_TYPE=Debug
      <path to root of llvm source tree>

If you want to run the test suite, you will need a compiler to build the test
programs. If you have Clang checked out, that will be used by default.
Alternatively, you can specify a C and C++ compiler to be used by the test
suite.

::

  $ cmake -G Ninja \
      -DLLDB_TEST_COMPILER=<path to C compiler> \
      <path to root of llvm source tree>

It is strongly recommend to use a release build for the compiler to speed up
test execution.

Windows
^^^^^^^

On Windows the LLDB test suite requires lld. Either add ``lld`` to
``LLVM_ENABLE_PROJECTS`` or disable the test suite with
``LLDB_INCLUDE_TESTS=OFF``.

Although the following CMake variables are by no means Windows specific, they
are commonly used on Windows.

* ``LLDB_TEST_DEBUG_TEST_CRASHES`` (Default=0): If set to 1, will cause Windows
  to generate a crash dialog whenever lldb.exe or the python extension module
  crashes while running the test suite. If set to 0, LLDB will silently crash.
  Setting to 1 allows a developer to attach a JIT debugger at the time of a
  crash, rather than having to reproduce a failure or use a crash dump.
* ``PYTHON_HOME`` (Required): Path to the folder where the Python distribution
  is installed. For example, ``C:\Python35``.
* ``LLDB_RELOCATABLE_PYTHON`` (Default=0): When this is 0, LLDB will bind
  statically to the location specified in the ``PYTHON_HOME`` CMake variable,
  ignoring any value of ``PYTHONHOME`` set in the environment. This is most
  useful for developers who simply want to run LLDB after they build it. If you
  wish to move a build of LLDB to a different machine where Python will be in a
  different location, setting ``LLDB_RELOCATABLE_PYTHON`` to 1 will cause
  Python to use its default mechanism for finding the python installation at
  runtime (looking for installed Pythons, or using the ``PYTHONHOME``
  environment variable if it is specified).

Sample command line:

::

  $ cmake -G Ninja^
      -DLLDB_TEST_DEBUG_TEST_CRASHES=1^
      -DPYTHON_HOME=C:\Python35^
      -DLLDB_TEST_COMPILER=d:\src\llvmbuild\ninja_release\bin\clang.exe^
      <path to root of llvm source tree>


Building with ninja is both faster and simpler than building with Visual Studio,
but chances are you still want to debug LLDB with an IDE. One solution is to run
cmake twice and generate the output into two different folders. One for
compiling (the ninja folder), and one for editing, browsing and debugging.

Follow the previous instructions in one directory, and generate a Visual Studio
project in another directory.

::

  $ cmake -G "Visual Studio 15 2017 Win64" -Thost=x64 <cmake variables> <path to root of llvm source tree>

Then you can open the .sln file in Visual Studio, set lldb as the startup
project, and use F5 to run it. You need only edit the project settings to set
the executable and the working directory to point to binaries inside of the
ninja tree.


macOS
^^^^^

On macOS the LLDB test suite requires libc++. Either add
``LLVM_ENABLE_RUNTIMES="libcxx;libcxxabi"`` or disable the test suite with
``LLDB_INCLUDE_TESTS=OFF``. Further useful options:

* ``LLDB_BUILD_FRAMEWORK:BOOL``: Builds the LLDB.framework.
* ``LLDB_CODESIGN_IDENTITY:STRING``: Set the identity to use for code-signing
  all executables. If not explicitly specified, only ``debugserver`` will be
  code-signed with identity ``lldb_codesign`` (see :ref:`CodeSigning`).
* ``LLDB_USE_SYSTEM_DEBUGSERVER:BOOL``: Use the system's debugserver, so lldb is
  functional without setting up code-signing.


.. _CMakeCaches:

CMake caches
************

CMake caches allow to store common sets of configuration options in the form of
CMake scripts and can be useful to reproduce builds for particular use-cases
(see by analogy `usage in LLVM and Clang <https://llvm.org/docs/AdvancedBuilds.html>`_).
A cache is passed to CMake with the ``-C`` flag, following the absolute path to
the file on disk. Subsequent ``-D`` options are still allowed. Please find the
currently available caches in the `lldb/cmake/caches/
<https://github.com/llvm/llvm-project/tree/main/lldb/cmake/caches>`_
directory.

Common configurations on macOS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Build, test and install a distribution of LLDB from the `monorepo
<https://github.com/llvm/llvm-project>`_ (see also `Building a Distribution of
LLVM <https://llvm.org/docs/BuildingADistribution.html>`_):

::

  $ git clone https://github.com/llvm/llvm-project

  $ cmake -B /path/to/lldb-build -G Ninja \
          -C /path/to/llvm-project/lldb/cmake/caches/Apple-lldb-macOS.cmake \
          -DLLVM_ENABLE_PROJECTS="clang;lldb" \
          -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi" \
          llvm-project/llvm

  $ DESTDIR=/path/to/lldb-install ninja -C /path/to/lldb-build check-lldb install-distribution

.. _CMakeGeneratedXcodeProject:

Build LLDB standalone for development with Xcode:

::

  $ git clone https://github.com/llvm/llvm-project

  $ cmake -B /path/to/llvm-build -G Ninja \
          -C /path/to/llvm-project/lldb/cmake/caches/Apple-lldb-base.cmake \
          -DLLVM_ENABLE_PROJECTS="clang" \
          -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi" \
          llvm-project/llvm
  $ ninja -C /path/to/llvm-build

  $ cmake -B /path/to/lldb-build \
          -C /path/to/llvm-project/lldb/cmake/caches/Apple-lldb-Xcode.cmake \
          -DLLVM_DIR=/path/to/llvm-build/lib/cmake/llvm \
          llvm-project/lldb
  $ open lldb.xcodeproj
  $ cmake --build /path/to/lldb-build --target check-lldb

.. note::

   The ``-B`` argument was undocumented for a while and is only officially
   supported since `CMake version 3.14
   <https://cmake.org/cmake/help/v3.14/release/3.14.html#command-line>`_


Building the Documentation
--------------------------

If you wish to build the optional (reference) documentation, additional
dependencies are required:

* Sphinx (for the website)
* Graphviz (for the 'dot' tool)
* doxygen (if you wish to build the C++ API reference)
* epydoc (if you wish to build the Python API reference)

To install the prerequisites for building the documentation (on Debian/Ubuntu)
do:

::

  $ sudo apt-get install doxygen graphviz python3-sphinx
  $ sudo pip install epydoc

To build the documentation, configure with ``LLVM_ENABLE_SPHINX=ON`` and build the desired target(s).

::

  $ ninja docs-lldb-html
  $ ninja docs-lldb-man
  $ ninja lldb-cpp-doc
  $ ninja lldb-python-doc

Cross-compiling LLDB
--------------------

In order to debug remote targets running different architectures than your
host, you will need to compile LLDB (or at least the server component) for the
target. While the easiest solution is to just compile it locally on the target,
this is often not feasible, and in these cases you will need to cross-compile
LLDB on your host.

Cross-compilation is often a daunting task and has a lot of quirks which depend
on the exact host and target architectures, so it is not possible to give a
universal guide which will work on all platforms. However, here we try to
provide an overview of the cross-compilation process along with the main things
you should look out for.

First, you will need a working toolchain which is capable of producing binaries
for the target architecture. Since you already have a checkout of clang and
lldb, you can compile a host version of clang in a separate folder and use
that. Alternatively you can use system clang or even cross-gcc if your
distribution provides such packages (e.g., ``g++-aarch64-linux-gnu`` on
Ubuntu).

Next, you will need a copy of the required target headers and libraries on your
host. The libraries can be usually obtained by copying from the target machine,
however the headers are often not found there, especially in case of embedded
platforms. In this case, you will need to obtain them from another source,
either a cross-package if one is available, or cross-compiling the respective
library from source. Fortunately the list of LLDB dependencies is not big and
if you are only interested in the server component, you can reduce this even
further by passing the appropriate cmake options, such as:

::

  -DLLDB_ENABLE_PYTHON=0
  -DLLDB_ENABLE_LIBEDIT=0
  -DLLDB_ENABLE_CURSES=0
  -DLLVM_ENABLE_TERMINFO=0

In this case you, will often not need anything other than the standard C and
C++ libraries.

Once all of the dependencies are in place, it's just a matter of configuring
the build system with the locations and arguments of all the necessary tools.
The most important cmake options here are:

* ``CMAKE_CROSSCOMPILING`` : Set to 1 to enable cross-compilation.
* ``CMAKE_LIBRARY_ARCHITECTURE`` : Affects the cmake search path when looking
  for libraries. You may need to set this to your architecture triple if you do
  not specify all your include and library paths explicitly.
* ``CMAKE_C_COMPILER``, ``CMAKE_CXX_COMPILER`` : C and C++ compilers for the
  target architecture
* ``CMAKE_C_FLAGS``, ``CMAKE_CXX_FLAGS`` : The flags for the C and C++ target
  compilers. You may need to specify the exact target cpu and abi besides the
  include paths for the target headers.
* ``CMAKE_EXE_LINKER_FLAGS`` : The flags to be passed to the linker. Usually
  just a list of library search paths referencing the target libraries.
* ``LLVM_TABLEGEN``, ``CLANG_TABLEGEN`` : Paths to llvm-tblgen and clang-tblgen
  for the host architecture. If you already have built clang for the host, you
  can point these variables to the executables in your build directory. If not,
  you will need to build the llvm-tblgen and clang-tblgen host targets at
  least.
* ``LLVM_HOST_TRIPLE`` : The triple of the system that lldb (or lldb-server)
  will run on. Not setting this (or setting it incorrectly) can cause a lot of
  issues with remote debugging as a lot of the choices lldb makes depend on the
  triple reported by the remote platform.

You can of course also specify the usual cmake options like
``CMAKE_BUILD_TYPE``, etc.

Example 1: Cross-compiling for linux arm64 on Ubuntu host
*********************************************************

Ubuntu already provides the packages necessary to cross-compile LLDB for arm64.
It is sufficient to install packages ``gcc-aarch64-linux-gnu``,
``g++-aarch64-linux-gnu``, ``binutils-aarch64-linux-gnu``. Then it is possible
to prepare the cmake build with the following parameters:

::

  -DCMAKE_CROSSCOMPILING=1 \
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
  -DLLVM_HOST_TRIPLE=aarch64-unknown-linux-gnu \
  -DLLVM_TABLEGEN=<path-to-host>/bin/llvm-tblgen \
  -DCLANG_TABLEGEN=<path-to-host>/bin/clang-tblgen \
  -DLLDB_ENABLE_PYTHON=0 \
  -DLLDB_ENABLE_LIBEDIT=0 \
  -DLLDB_ENABLE_CURSES=0

An alternative (and recommended) way to compile LLDB is with clang.
Unfortunately, clang is not able to find all the include paths necessary for a
successful cross-compile, so we need to help it with a couple of CFLAGS
options. In my case it was sufficient to add the following arguments to
``CMAKE_C_FLAGS`` and ``CMAKE_CXX_FLAGS`` (in addition to changing
``CMAKE_C(XX)_COMPILER`` to point to clang compilers):

::

  -target aarch64-linux-gnu \
  -I /usr/aarch64-linux-gnu/include/c++/4.8.2/aarch64-linux-gnu \
  -I /usr/aarch64-linux-gnu/include

If you wanted to build a full version of LLDB and avoid passing
``-DLLDB_ENABLE_PYTHON=0`` and other options, you would need to obtain the
target versions of the respective libraries. The easiest way to achieve this is
to use the qemu-debootstrap utility, which can prepare a system image using
qemu and chroot to simulate the target environment. Then you can install the
necessary packages in this environment (python-dev, libedit-dev, etc.) and
point your compiler to use them using the correct -I and -L arguments.

Example 2: Cross-compiling for Android on Linux
***********************************************

In the case of Android, the toolchain and all required headers and libraries
are available in the Android NDK.

The NDK also contains a cmake toolchain file, which makes configuring the build
much simpler. The compiler, include and library paths will be configured by the
toolchain file and all you need to do is to select the architecture
(ANDROID_ABI) and platform level (``ANDROID_PLATFORM``, should be at least 21).
You will also need to set ``ANDROID_ALLOW_UNDEFINED_SYMBOLS=On``, as the
toolchain file defaults to "no undefined symbols in shared libraries", which is
not compatible with some llvm libraries. The first version of NDK which
supports this approach is r14.

For example, the following arguments are sufficient to configure an android
arm64 build:

::

  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-21 \
  -DANDROID_ALLOW_UNDEFINED_SYMBOLS=On \
  -DLLVM_HOST_TRIPLE=aarch64-unknown-linux-android \
  -DCROSS_TOOLCHAIN_FLAGS_NATIVE='-DCMAKE_C_COMPILER=cc;-DCMAKE_CXX_COMPILER=c++'

Note that currently only lldb-server is functional on android. The lldb client
is not supported and unlikely to work.

Verifying Python Support
------------------------

LLDB has a Python scripting capability and supplies its own Python module named
lldb. If a script is run inside the command line lldb application, the Python
module is made available automatically. However, if a script is to be run by a
Python interpreter outside the command line application, the ``PYTHONPATH``
environment variable can be used to let the Python interpreter find the lldb
module.

The correct path can be obtained by invoking the command line lldb tool with
the -P flag:

::

  $ export PYTHONPATH=`$llvm/build/Debug+Asserts/bin/lldb -P`

If you used a different build directory or made a release build, you may need
to adjust the above to suit your needs. To test that the lldb Python module is
built correctly and is available to the default Python interpreter, run:

::

  $ python -c 'import lldb'


Make sure you're using the Python interpreter that matches the Python library
you linked against. For more details please refer to the :ref:`caveats
<python_caveat>`.

.. _CodeSigning:

Code Signing on macOS
---------------------

To use the in-tree debug server on macOS, lldb needs to be code signed. The
Debug, DebugClang and Release builds are set to code sign using a code signing
certificate named ``lldb_codesign``.

Automatic setup, run:

* ``scripts/macos-setup-codesign.sh``

Note that it's possible to build and use lldb on macOS without setting up code
signing by using the system's debug server. To configure lldb in this way with
cmake, specify ``-DLLDB_USE_SYSTEM_DEBUGSERVER=ON``.

If you have re-installed a new OS, please delete all old ``lldb_codesign`` items
from your keychain. There will be a code signing certification and a public
and private key. Reboot after deleting them. You will also need to delete and
build folders that contained old signed items. The darwin kernel will cache
code signing using the executable's file system node, so you will need to
delete the file so the kernel clears its cache.

When you build your LLDB for the first time, the Xcode GUI will prompt you for
permission to use the ``lldb_codesign`` keychain. Be sure to click "Always
Allow" on your first build. From here on out, the ``lldb_codesign`` will be
trusted and you can build from the command line without having to authorize.
Also the first time you debug using a LLDB that was built with this code
signing certificate, you will need to authenticate once.
