====================================
Getting Started with the LLVM System
====================================

.. contents::
   :local:

Overview
========

Welcome to the LLVM project!

The LLVM project has multiple components. The core of the project is
itself called "LLVM". This contains all of the tools, libraries, and header
files needed to process intermediate representations and converts it into
object files.  Tools include an assembler, disassembler, bitcode analyzer, and
bitcode optimizer.  It also contains basic regression tests.

C-like languages use the `Clang <https://clang.llvm.org/>`_ front end.  This
component compiles C, C++, Objective C, and Objective C++ code into LLVM bitcode
-- and from there into object files, using LLVM.

Other components include:
the `libc++ C++ standard library <https://libcxx.llvm.org>`_,
the `LLD linker <https://lld.llvm.org>`_, and more.

Getting the Source Code and Building LLVM
=========================================

The LLVM Getting Started documentation may be out of date.  The `Clang
Getting Started <https://clang.llvm.org/get_started.html>`_ page might have more
accurate information.

This is an example workflow and configuration to get and build the LLVM source:

#. Checkout LLVM (including related subprojects like Clang):

   * ``git clone https://github.com/llvm/llvm-project.git``
   * Or, on windows, ``git clone --config core.autocrlf=false
     https://github.com/llvm/llvm-project.git``
   * To save storage and speed-up the checkout time, you may want to do a
     `shallow clone <https://git-scm.com/docs/git-clone#Documentation/git-clone.txt---depthltdepthgt>`_.
     For example, to get the latest revision of the LLVM project, use
     ``git clone --depth 1 https://github.com/llvm/llvm-project.git``

#. Configure and build LLVM and Clang:

   * ``cd llvm-project``
   * ``mkdir build``
   * ``cd build``
   * ``cmake -G <generator> -DCMAKE_BUILD_TYPE=<type> [options] ../llvm``

     Some common build system generators are:

     * ``Ninja`` --- for generating `Ninja <https://ninja-build.org>`_
       build files. Most llvm developers use Ninja.
     * ``Unix Makefiles`` --- for generating make-compatible parallel makefiles.
     * ``Visual Studio`` --- for generating Visual Studio projects and
       solutions.
     * ``Xcode`` --- for generating Xcode projects.

     Some Common options:

     * ``-DLLVM_ENABLE_PROJECTS='...'`` --- semicolon-separated list of the LLVM
       subprojects you'd like to additionally build. Can include any of: clang,
       clang-tools-extra, lldb, compiler-rt, lld, polly, or cross-project-tests.

       For example, to build LLVM, Clang, libcxx, and libcxxabi, use
       ``-DLLVM_ENABLE_PROJECTS="clang" -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi"``.

     * ``-DCMAKE_INSTALL_PREFIX=directory`` --- Specify for *directory* the full
       pathname of where you want the LLVM tools and libraries to be installed
       (default ``/usr/local``).

     * ``-DCMAKE_BUILD_TYPE=type`` --- Controls optimization level and debug information
       of the build. The default value is ``Debug`` which fits people who want
       to work on LLVM or its libraries. ``Release`` is a better fit for most
       users of LLVM and Clang. For more detailed information see
       :ref:`CMAKE_BUILD_TYPE <cmake_build_type>`.

     * ``-DLLVM_ENABLE_ASSERTIONS=On`` --- Compile with assertion checks enabled
       (default is Yes for Debug builds, No for all other build types).

   * ``cmake --build . [--target <target>]`` or the build system specified
     above directly.

     * The default target (i.e. ``cmake --build .`` or ``make``) will build all of
       LLVM.

     * The ``check-all`` target (i.e. ``ninja check-all``) will run the
       regression tests to ensure everything is in working order.

     * CMake will generate build targets for each tool and library, and most
       LLVM sub-projects generate their own ``check-<project>`` target.

     * Running a serial build will be **slow**.  To improve speed, try running a
       parallel build. That's done by default in Ninja; for ``make``, use the
       option ``-j NN``, where ``NN`` is the number of parallel jobs, e.g. the
       number of available CPUs.

   * For more information see `CMake <CMake.html>`__

   * If you get an "internal compiler error (ICE)" or test failures, see
     `below`_.

Consult the `Getting Started with LLVM`_ section for detailed information on
configuring and compiling LLVM.  Go to `Directory Layout`_ to learn about the
layout of the source code tree.

Requirements
============

Before you begin to use the LLVM system, review the requirements given below.
This may save you some trouble by knowing ahead of time what hardware and
software you will need.

Hardware
--------

LLVM is known to work on the following host platforms:

================== ===================== =============
OS                 Arch                  Compilers
================== ===================== =============
Linux              x86\ :sup:`1`         GCC, Clang
Linux              amd64                 GCC, Clang
Linux              ARM                   GCC, Clang
Linux              Mips                  GCC, Clang
Linux              PowerPC               GCC, Clang
Linux              SystemZ               GCC, Clang
Solaris            V9 (Ultrasparc)       GCC
DragonFlyBSD       amd64                 GCC, Clang
FreeBSD            x86\ :sup:`1`         GCC, Clang
FreeBSD            amd64                 GCC, Clang
NetBSD             x86\ :sup:`1`         GCC, Clang
NetBSD             amd64                 GCC, Clang
OpenBSD            x86\ :sup:`1`         GCC, Clang
OpenBSD            amd64                 GCC, Clang
macOS\ :sup:`2`    PowerPC               GCC
macOS              x86                   GCC, Clang
Cygwin/Win32       x86\ :sup:`1, 3`      GCC
Windows            x86\ :sup:`1`         Visual Studio
Windows x64        x86-64                Visual Studio
================== ===================== =============

.. note::

  #. Code generation supported for Pentium processors and up
  #. Code generation supported for 32-bit ABI only
  #. To use LLVM modules on Win32-based system, you may configure LLVM
     with ``-DBUILD_SHARED_LIBS=On``.

Note that Debug builds require a lot of time and disk space.  An LLVM-only build
will need about 1-3 GB of space.  A full build of LLVM and Clang will need around
15-20 GB of disk space.  The exact space requirements will vary by system.  (It
is so large because of all the debugging information and the fact that the
libraries are statically linked into multiple tools).

If you are space-constrained, you can build only selected tools or only
selected targets.  The Release build requires considerably less space.

The LLVM suite *may* compile on other platforms, but it is not guaranteed to do
so.  If compilation is successful, the LLVM utilities should be able to
assemble, disassemble, analyze, and optimize LLVM bitcode.  Code generation
should work as well, although the generated native code may not work on your
platform.

Software
--------

Compiling LLVM requires that you have several software packages installed. The
table below lists those required packages. The Package column is the usual name
for the software package that LLVM depends on. The Version column provides
"known to work" versions of the package. The Notes column describes how LLVM
uses the package and provides other details.

=========================================================== ============ ==========================================
Package                                                     Version      Notes
=========================================================== ============ ==========================================
`CMake <http://cmake.org/>`__                               >=3.13.4     Makefile/workspace generator
`GCC <http://gcc.gnu.org/>`_                                >=7.1.0      C/C++ compiler\ :sup:`1`
`python <http://www.python.org/>`_                          >=3.6        Automated test suite\ :sup:`2`
`zlib <http://zlib.net>`_                                   >=1.2.3.4    Compression library\ :sup:`3`
`GNU Make <http://savannah.gnu.org/projects/make>`_         3.79, 3.79.1 Makefile/build processor\ :sup:`4`
=========================================================== ============ ==========================================

.. note::

   #. Only the C and C++ languages are needed so there's no need to build the
      other languages for LLVM's purposes. See `below` for specific version
      info.
   #. Only needed if you want to run the automated test suite in the
      ``llvm/test`` directory.
   #. Optional, adds compression / uncompression capabilities to selected LLVM
      tools.
   #. Optional, you can use any other build tool supported by CMake.

Additionally, your compilation host is expected to have the usual plethora of
Unix utilities. Specifically:

* **ar** --- archive library builder
* **bzip2** --- bzip2 command for distribution generation
* **bunzip2** --- bunzip2 command for distribution checking
* **chmod** --- change permissions on a file
* **cat** --- output concatenation utility
* **cp** --- copy files
* **date** --- print the current date/time
* **echo** --- print to standard output
* **egrep** --- extended regular expression search utility
* **find** --- find files/dirs in a file system
* **grep** --- regular expression search utility
* **gzip** --- gzip command for distribution generation
* **gunzip** --- gunzip command for distribution checking
* **install** --- install directories/files
* **mkdir** --- create a directory
* **mv** --- move (rename) files
* **ranlib** --- symbol table builder for archive libraries
* **rm** --- remove (delete) files and directories
* **sed** --- stream editor for transforming output
* **sh** --- Bourne shell for make build scripts
* **tar** --- tape archive for distribution generation
* **test** --- test things in file system
* **unzip** --- unzip command for distribution checking
* **zip** --- zip command for distribution generation

.. _below:
.. _check here:

Host C++ Toolchain, both Compiler and Standard Library
------------------------------------------------------

LLVM is very demanding of the host C++ compiler, and as such tends to expose
bugs in the compiler. We also attempt to follow improvements and developments in
the C++ language and library reasonably closely. As such, we require a modern
host C++ toolchain, both compiler and standard library, in order to build LLVM.

LLVM is written using the subset of C++ documented in :doc:`coding
standards<CodingStandards>`. To enforce this language version, we check the most
popular host toolchains for specific minimum versions in our build systems:

* Clang 5.0
* Apple Clang 9.3
* GCC 7.1
* Visual Studio 2019 16.7

Anything older than these toolchains *may* work, but will require forcing the
build system with a special option and is not really a supported host platform.
Also note that older versions of these compilers have often crashed or
miscompiled LLVM.

For less widely used host toolchains such as ICC or xlC, be aware that a very
recent version may be required to support all of the C++ features used in LLVM.

We track certain versions of software that are *known* to fail when used as
part of the host toolchain. These even include linkers at times.

**GNU ld 2.16.X**. Some 2.16.X versions of the ld linker will produce very long
warning messages complaining that some "``.gnu.linkonce.t.*``" symbol was
defined in a discarded section. You can safely ignore these messages as they are
erroneous and the linkage is correct.  These messages disappear using ld 2.17.

**GNU binutils 2.17**: Binutils 2.17 contains `a bug
<http://sourceware.org/bugzilla/show_bug.cgi?id=3111>`__ which causes huge link
times (minutes instead of seconds) when building LLVM.  We recommend upgrading
to a newer version (2.17.50.0.4 or later).

**GNU Binutils 2.19.1 Gold**: This version of Gold contained `a bug
<http://sourceware.org/bugzilla/show_bug.cgi?id=9836>`__ which causes
intermittent failures when building LLVM with position independent code.  The
symptom is an error about cyclic dependencies.  We recommend upgrading to a
newer version of Gold.

Getting a Modern Host C++ Toolchain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section mostly applies to Linux and older BSDs. On macOS, you should
have a sufficiently modern Xcode, or you will likely need to upgrade until you
do. Windows does not have a "system compiler", so you must install either Visual
Studio 2019 (or later), or a recent version of mingw64. FreeBSD 10.0 and newer
have a modern Clang as the system compiler.

However, some Linux distributions and some other or older BSDs sometimes have
extremely old versions of GCC. These steps attempt to help you upgrade you
compiler even on such a system. However, if at all possible, we encourage you
to use a recent version of a distribution with a modern system compiler that
meets these requirements. Note that it is tempting to install a prior
version of Clang and libc++ to be the host compiler, however libc++ was not
well tested or set up to build on Linux until relatively recently. As
a consequence, this guide suggests just using libstdc++ and a modern GCC as the
initial host in a bootstrap, and then using Clang (and potentially libc++).

The first step is to get a recent GCC toolchain installed. The most common
distribution on which users have struggled with the version requirements is
Ubuntu Precise, 12.04 LTS. For this distribution, one easy option is to install
the `toolchain testing PPA`_ and use it to install a modern GCC. There is
a really nice discussions of this on the `ask ubuntu stack exchange`_ and a
`github gist`_ with updated commands. However, not all users can use PPAs and
there are many other distributions, so it may be necessary (or just useful, if
you're here you *are* doing compiler development after all) to build and install
GCC from source. It is also quite easy to do these days.

.. _toolchain testing PPA:
  https://launchpad.net/~ubuntu-toolchain-r/+archive/test
.. _ask ubuntu stack exchange:
  https://askubuntu.com/questions/466651/how-do-i-use-the-latest-gcc-on-ubuntu/581497#58149
.. _github gist:
  https://gist.github.com/application2000/73fd6f4bf1be6600a2cf9f56315a2d91

Easy steps for installing GCC 7.1.0:

.. code-block:: console

  % gcc_version=7.1.0
  % wget https://ftp.gnu.org/gnu/gcc/gcc-${gcc_version}/gcc-${gcc_version}.tar.bz2
  % wget https://ftp.gnu.org/gnu/gcc/gcc-${gcc_version}/gcc-${gcc_version}.tar.bz2.sig
  % wget https://ftp.gnu.org/gnu/gnu-keyring.gpg
  % signature_invalid=`gpg --verify --no-default-keyring --keyring ./gnu-keyring.gpg gcc-${gcc_version}.tar.bz2.sig`
  % if [ $signature_invalid ]; then echo "Invalid signature" ; exit 1 ; fi
  % tar -xvjf gcc-${gcc_version}.tar.bz2
  % cd gcc-${gcc_version}
  % ./contrib/download_prerequisites
  % cd ..
  % mkdir gcc-${gcc_version}-build
  % cd gcc-${gcc_version}-build
  % $PWD/../gcc-${gcc_version}/configure --prefix=$HOME/toolchains --enable-languages=c,c++
  % make -j$(nproc)
  % make install

For more details, check out the excellent `GCC wiki entry`_, where I got most
of this information from.

.. _GCC wiki entry:
  https://gcc.gnu.org/wiki/InstallingGCC

Once you have a GCC toolchain, configure your build of LLVM to use the new
toolchain for your host compiler and C++ standard library. Because the new
version of libstdc++ is not on the system library search path, you need to pass
extra linker flags so that it can be found at link time (``-L``) and at runtime
(``-rpath``). If you are using CMake, this invocation should produce working
binaries:

.. code-block:: console

  % mkdir build
  % cd build
  % CC=$HOME/toolchains/bin/gcc CXX=$HOME/toolchains/bin/g++ \
    cmake .. -DCMAKE_CXX_LINK_FLAGS="-Wl,-rpath,$HOME/toolchains/lib64 -L$HOME/toolchains/lib64"

If you fail to set rpath, most LLVM binaries will fail on startup with a message
from the loader similar to ``libstdc++.so.6: version `GLIBCXX_3.4.20' not
found``. This means you need to tweak the -rpath linker flag.

This method will add an absolute path to the rpath of all executables. That's
fine for local development. If you want to distribute the binaries you build
so that they can run on older systems, copy ``libstdc++.so.6`` into the
``lib/`` directory.  All of LLVM's shipping binaries have an rpath pointing at
``$ORIGIN/../lib``, so they will find ``libstdc++.so.6`` there.  Non-distributed
binaries don't have an rpath set and won't find ``libstdc++.so.6``. Pass
``-DLLVM_LOCAL_RPATH="$HOME/toolchains/lib64"`` to cmake to add an absolute
path to ``libstdc++.so.6`` as above. Since these binaries are not distributed,
having an absolute local path is fine for them.

When you build Clang, you will need to give *it* access to modern C++
standard library in order to use it as your new host in part of a bootstrap.
There are two easy ways to do this, either build (and install) libc++ along
with Clang and then use it with the ``-stdlib=libc++`` compile and link flag,
or install Clang into the same prefix (``$HOME/toolchains`` above) as GCC.
Clang will look within its own prefix for libstdc++ and use it if found. You
can also add an explicit prefix for Clang to look in for a GCC toolchain with
the ``--gcc-toolchain=/opt/my/gcc/prefix`` flag, passing it to both compile and
link commands when using your just-built-Clang to bootstrap.

.. _Getting Started with LLVM:

Getting Started with LLVM
=========================

The remainder of this guide is meant to get you up and running with LLVM and to
give you some basic information about the LLVM environment.

The later sections of this guide describe the `general layout`_ of the LLVM
source tree, a `simple example`_ using the LLVM tool chain, and `links`_ to find
more information about LLVM or to get help via e-mail.

Terminology and Notation
------------------------

Throughout this manual, the following names are used to denote paths specific to
the local system and working environment.  *These are not environment variables
you need to set but just strings used in the rest of this document below*.  In
any of the examples below, simply replace each of these names with the
appropriate pathname on your local system.  All these paths are absolute:

``SRC_ROOT``

  This is the top level directory of the LLVM source tree.

``OBJ_ROOT``

  This is the top level directory of the LLVM object tree (i.e. the tree where
  object files and compiled programs will be placed.  It can be the same as
  SRC_ROOT).

Unpacking the LLVM Archives
---------------------------

If you have the LLVM distribution, you will need to unpack it before you can
begin to compile it.  LLVM is distributed as a number of different
subprojects. Each one has its own download which is a TAR archive that is
compressed with the gzip program.

The files are as follows, with *x.y* marking the version number:

``llvm-x.y.tar.gz``

  Source release for the LLVM libraries and tools.

``cfe-x.y.tar.gz``

  Source release for the Clang frontend.

.. _checkout:

Checkout LLVM from Git
----------------------

You can also checkout the source code for LLVM from Git.

.. note::

  Passing ``--config core.autocrlf=false`` should not be required in
  the future after we adjust the .gitattribute settings correctly, but
  is required for Windows users at the time of this writing.

Simply run:

.. code-block:: console

  % git clone https://github.com/llvm/llvm-project.git

or on Windows,

.. code-block:: console

  % git clone --config core.autocrlf=false https://github.com/llvm/llvm-project.git

This will create an '``llvm-project``' directory in the current directory and
fully populate it with all of the source code, test directories, and local
copies of documentation files for LLVM and all the related subprojects. Note
that unlike the tarballs, which contain each subproject in a separate file, the
git repository contains all of the projects together.

If you want to get a specific release (as opposed to the most recent revision),
you can check out a tag after cloning the repository. E.g., `git checkout
llvmorg-6.0.1` inside the ``llvm-project`` directory created by the above
command.  Use `git tag -l` to list all of them.

Sending patches
^^^^^^^^^^^^^^^

Please read `Developer Policy <DeveloperPolicy.html#one-off-patches>`_, too.

We don't currently accept github pull requests, so you'll need to send patches
either via emailing to llvm-commits, or, preferably, via :ref:`Phabricator
<phabricator-reviews>`.

You'll generally want to make sure your branch has a single commit,
corresponding to the review you wish to send, up-to-date with the upstream
``origin/main`` branch, and doesn't contain merges. Once you have that, you
can start `a Phabricator review <Phabricator.html>`_ (or use ``git show`` or
``git format-patch`` to output the diff, and attach it to an email message).

However, using the "Arcanist" tool is often easier. After `installing arcanist`_, you
will also need to apply a fix to your arcanist repo in order to submit a patch:

.. code-block:: console

  % cd arcanist
  % git fetch https://github.com/rashkov/arcanist update_cacerts
  % git cherry-pick e3659d43d8911e91739f3b0c5935598bceb859aa

Once this is all done, you can upload the latest commit using:

.. code-block:: console

  % arc diff HEAD~1

Additionally, before sending a patch for review, please also try to ensure it's
formatted properly. We use ``clang-format`` for this, which has git integration
through the ``git-clang-format`` script. On some systems, it may already be
installed (or be installable via your package manager). If so, you can simply
run it -- the following command will format only the code changed in the most
recent commit:

.. code-block:: console

  % git clang-format HEAD~1

Note that this modifies the files, but doesn't commit them -- you'll likely want
to run

.. code-block:: console

  % git commit --amend -a

in order to update the last commit with all pending changes.

.. note::
  If you don't already have ``clang-format`` or ``git clang-format`` installed
  on your system, the ``clang-format`` binary will be built alongside clang, and
  the git integration can be run from
  ``clang/tools/clang-format/git-clang-format``.


.. _commit_from_git:

For developers to commit changes from Git
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once a patch is reviewed, you should rebase it, re-test locally, and commit the
changes to LLVM's main branch. This is done using `git push` if you have the
required access rights. See `committing a change
<Phabricator.html#committing-a-change>`_ for Phabricator based commits or
`obtaining commit access <DeveloperPolicy.html#obtaining-commit-access>`_
for commit access.

Here is an example workflow using git. This workflow assumes you have an
accepted commit on the branch named `branch-with-change`.

.. code-block:: console

  # Go to the branch with your accepted commit.
  % git checkout branch-with-change
  # Rebase your change onto the latest commits on Github.
  % git pull --rebase origin main
  # Rerun the appropriate tests if needed.
  % ninja check-$whatever
  # Check that the list of commits about to be pushed is correct.
  % git log origin/main...HEAD --oneline
  # Push to Github.
  % git push origin HEAD:main

LLVM currently has a linear-history policy, which means that merge commits are
not allowed. The `llvm-project` repo on github is configured to reject pushes
that include merges, so the `git rebase` step above is required.

Please ask for help if you're having trouble with your particular git workflow.


.. _git_pre_push_hook:

Git pre-push hook
^^^^^^^^^^^^^^^^^

We include an optional pre-push hook that run some sanity checks on the revisions
you are about to push and ask confirmation if you push multiple commits at once.
You can set it up (on Unix systems) by running from the repository root:

.. code-block:: console

  % ln -sf ../../llvm/utils/git/pre-push.py .git/hooks/pre-push

Bisecting commits
^^^^^^^^^^^^^^^^^

See `Bisecting LLVM code <GitBisecting.html>`_ for how to use ``git bisect``
on LLVM.

Reverting a change
^^^^^^^^^^^^^^^^^^

When reverting changes using git, the default message will say "This reverts
commit XYZ". Leave this at the end of the commit message, but add some details
before it as to why the commit is being reverted. A brief explanation and/or
links to bots that demonstrate the problem are sufficient.

Local LLVM Configuration
------------------------

Once checked out repository, the LLVM suite source code must be configured
before being built. This process uses CMake.  Unlinke the normal ``configure``
script, CMake generates the build files in whatever format you request as well
as various ``*.inc`` files, and ``llvm/include/Config/config.h``.

Variables are passed to ``cmake`` on the command line using the format
``-D<variable name>=<value>``. The following variables are some common options
used by people developing LLVM.

+-------------------------+----------------------------------------------------+
| Variable                | Purpose                                            |
+=========================+====================================================+
| CMAKE_C_COMPILER        | Tells ``cmake`` which C compiler to use. By        |
|                         | default, this will be /usr/bin/cc.                 |
+-------------------------+----------------------------------------------------+
| CMAKE_CXX_COMPILER      | Tells ``cmake`` which C++ compiler to use. By      |
|                         | default, this will be /usr/bin/c++.                |
+-------------------------+----------------------------------------------------+
| CMAKE_BUILD_TYPE        | Tells ``cmake`` what type of build you are trying  |
|                         | to generate files for. Valid options are Debug,    |
|                         | Release, RelWithDebInfo, and MinSizeRel. Default   |
|                         | is Debug.                                          |
+-------------------------+----------------------------------------------------+
| CMAKE_INSTALL_PREFIX    | Specifies the install directory to target when     |
|                         | running the install action of the build files.     |
+-------------------------+----------------------------------------------------+
| PYTHON_EXECUTABLE       | Forces CMake to use a specific Python version by   |
|                         | passing a path to a Python interpreter. By default |
|                         | the Python version of the interpreter in your PATH |
|                         | is used.                                           |
+-------------------------+----------------------------------------------------+
| LLVM_TARGETS_TO_BUILD   | A semicolon delimited list controlling which       |
|                         | targets will be built and linked into llvm.        |
|                         | The default list is defined as                     |
|                         | ``LLVM_ALL_TARGETS``, and can be set to include    |
|                         | out-of-tree targets. The default value includes:   |
|                         | ``AArch64, AMDGPU, ARM, AVR, BPF, Hexagon, Lanai,  |
|                         | Mips, MSP430, NVPTX, PowerPC, RISCV, Sparc,        |
|                         | SystemZ, WebAssembly, X86, XCore``.                |
|                         |                                                    |
+-------------------------+----------------------------------------------------+
| LLVM_ENABLE_DOXYGEN     | Build doxygen-based documentation from the source  |
|                         | code This is disabled by default because it is     |
|                         | slow and generates a lot of output.                |
+-------------------------+----------------------------------------------------+
| LLVM_ENABLE_PROJECTS    | A semicolon-delimited list selecting which of the  |
|                         | other LLVM subprojects to additionally build. (Only|
|                         | effective when using a side-by-side project layout |
|                         | e.g. via git). The default list is empty. Can      |
|                         | include: clang, clang-tools-extra,                 |
|                         | cross-project-tests, flang, libc, libclc, lld,     |
|                         | lldb, mlir, openmp, polly, or pstl.                |
+-------------------------+----------------------------------------------------+
| LLVM_ENABLE_RUNTIMES    | A semicolon-delimited list selecting which of the  |
|                         | runtimes to build. (Only effective when using the  |
|                         | full monorepo layout). The default list is empty.  |
|                         | Can include: compiler-rt, libc, libcxx, libcxxabi, |
|                         | libunwind, or openmp.                              |
+-------------------------+----------------------------------------------------+
| LLVM_ENABLE_SPHINX      | Build sphinx-based documentation from the source   |
|                         | code. This is disabled by default because it is    |
|                         | slow and generates a lot of output. Sphinx version |
|                         | 1.5 or later recommended.                          |
+-------------------------+----------------------------------------------------+
| LLVM_BUILD_LLVM_DYLIB   | Generate libLLVM.so. This library contains a       |
|                         | default set of LLVM components that can be         |
|                         | overridden with ``LLVM_DYLIB_COMPONENTS``. The     |
|                         | default contains most of LLVM and is defined in    |
|                         | ``tools/llvm-shlib/CMakelists.txt``. This option is|
|                         | not available on Windows.                          |
+-------------------------+----------------------------------------------------+
| LLVM_OPTIMIZED_TABLEGEN | Builds a release tablegen that gets used during    |
|                         | the LLVM build. This can dramatically speed up     |
|                         | debug builds.                                      |
+-------------------------+----------------------------------------------------+

To configure LLVM, follow these steps:

#. Change directory into the object root directory:

   .. code-block:: console

     % cd OBJ_ROOT

#. Run the ``cmake``:

   .. code-block:: console

     % cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=<type> -DCMAKE_INSTALL_PREFIX=/install/path
       [other options] SRC_ROOT

Compiling the LLVM Suite Source Code
------------------------------------

Unlike with autotools, with CMake your build type is defined at configuration.
If you want to change your build type, you can re-run cmake with the following
invocation:

   .. code-block:: console

     % cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=<type> -DCMAKE_BUILD_TYPE=type SRC_ROOT

Between runs, CMake preserves the values set for all options. CMake has the
following build types defined:

Debug

  These builds are the default. The build system will compile the tools and
  libraries unoptimized, with debugging information, and asserts enabled.

Release

  For these builds, the build system will compile the tools and libraries
  with optimizations enabled and not generate debug info. CMakes default
  optimization level is -O3. This can be configured by setting the
  ``CMAKE_CXX_FLAGS_RELEASE`` variable on the CMake command line.

RelWithDebInfo

  These builds are useful when debugging. They generate optimized binaries with
  debug information. CMakes default optimization level is -O2. This can be
  configured by setting the ``CMAKE_CXX_FLAGS_RELWITHDEBINFO`` variable on the
  CMake command line.

Once you have LLVM configured, you can build it by entering the *OBJ_ROOT*
directory and issuing the following command:

.. code-block:: console

  % make

If the build fails, please `check here`_ to see if you are using a version of
GCC that is known not to compile LLVM.

If you have multiple processors in your machine, you may wish to use some of the
parallel build options provided by GNU Make.  For example, you could use the
command:

.. code-block:: console

  % make -j2

There are several special targets which are useful when working with the LLVM
source code:

``make clean``

  Removes all files generated by the build.  This includes object files,
  generated C/C++ files, libraries, and executables.

``make install``

  Installs LLVM header files, libraries, tools, and documentation in a hierarchy
  under ``$PREFIX``, specified with ``CMAKE_INSTALL_PREFIX``, which
  defaults to ``/usr/local``.

``make docs-llvm-html``

  If configured with ``-DLLVM_ENABLE_SPHINX=On``, this will generate a directory
  at ``OBJ_ROOT/docs/html`` which contains the HTML formatted documentation.

Cross-Compiling LLVM
--------------------

It is possible to cross-compile LLVM itself. That is, you can create LLVM
executables and libraries to be hosted on a platform different from the platform
where they are built (a Canadian Cross build). To generate build files for
cross-compiling CMake provides a variable ``CMAKE_TOOLCHAIN_FILE`` which can
define compiler flags and variables used during the CMake test operations.

The result of such a build is executables that are not runnable on the build
host but can be executed on the target. As an example the following CMake
invocation can generate build files targeting iOS. This will work on macOS
with the latest Xcode:

.. code-block:: console

  % cmake -G "Ninja" -DCMAKE_OSX_ARCHITECTURES="armv7;armv7s;arm64"
    -DCMAKE_TOOLCHAIN_FILE=<PATH_TO_LLVM>/cmake/platforms/iOS.cmake
    -DCMAKE_BUILD_TYPE=Release -DLLVM_BUILD_RUNTIME=Off -DLLVM_INCLUDE_TESTS=Off
    -DLLVM_INCLUDE_EXAMPLES=Off -DLLVM_ENABLE_BACKTRACES=Off [options]
    <PATH_TO_LLVM>

Note: There are some additional flags that need to be passed when building for
iOS due to limitations in the iOS SDK.

Check :doc:`HowToCrossCompileLLVM` and `Clang docs on how to cross-compile in general
<https://clang.llvm.org/docs/CrossCompilation.html>`_ for more information
about cross-compiling.

The Location of LLVM Object Files
---------------------------------

The LLVM build system is capable of sharing a single LLVM source tree among
several LLVM builds.  Hence, it is possible to build LLVM for several different
platforms or configurations using the same source tree.

* Change directory to where the LLVM object files should live:

  .. code-block:: console

    % cd OBJ_ROOT

* Run ``cmake``:

  .. code-block:: console

    % cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release SRC_ROOT

The LLVM build will create a structure underneath *OBJ_ROOT* that matches the
LLVM source tree. At each level where source files are present in the source
tree there will be a corresponding ``CMakeFiles`` directory in the *OBJ_ROOT*.
Underneath that directory there is another directory with a name ending in
``.dir`` under which you'll find object files for each source.

For example:

  .. code-block:: console

    % cd llvm_build_dir
    % find lib/Support/ -name APFloat*
    lib/Support/CMakeFiles/LLVMSupport.dir/APFloat.cpp.o

Optional Configuration Items
----------------------------

If you're running on a Linux system that supports the `binfmt_misc
<http://en.wikipedia.org/wiki/binfmt_misc>`_
module, and you have root access on the system, you can set your system up to
execute LLVM bitcode files directly. To do this, use commands like this (the
first command may not be required if you are already using the module):

.. code-block:: console

  % mount -t binfmt_misc none /proc/sys/fs/binfmt_misc
  % echo ':llvm:M::BC::/path/to/lli:' > /proc/sys/fs/binfmt_misc/register
  % chmod u+x hello.bc   (if needed)
  % ./hello.bc

This allows you to execute LLVM bitcode files directly.  On Debian, you can also
use this command instead of the 'echo' command above:

.. code-block:: console

  % sudo update-binfmts --install llvm /path/to/lli --magic 'BC'

.. _Program Layout:
.. _general layout:

Directory Layout
================

One useful source of information about the LLVM source base is the LLVM `doxygen
<http://www.doxygen.org/>`_ documentation available at
`<https://llvm.org/doxygen/>`_.  The following is a brief introduction to code
layout:

``llvm/cmake``
--------------
Generates system build files.

``llvm/cmake/modules``
  Build configuration for llvm user defined options. Checks compiler version and
  linker flags.

``llvm/cmake/platforms``
  Toolchain configuration for Android NDK, iOS systems and non-Windows hosts to
  target MSVC.

``llvm/examples``
-----------------

- Some simple examples showing how to use LLVM as a compiler for a custom
  language - including lowering, optimization, and code generation.

- Kaleidoscope Tutorial: Kaleidoscope language tutorial run through the
  implementation of a nice little compiler for a non-trivial language
  including a hand-written lexer, parser, AST, as well as code generation
  support using LLVM- both static (ahead of time) and various approaches to
  Just In Time (JIT) compilation.
  `Kaleidoscope Tutorial for complete beginner
  <https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html>`_.

- BuildingAJIT: Examples of the `BuildingAJIT tutorial
  <https://llvm.org/docs/tutorial/BuildingAJIT1.html>`_ that shows how LLVM’s
  ORC JIT APIs interact with other parts of LLVM. It also, teaches how to
  recombine them to build a custom JIT that is suited to your use-case.

``llvm/include``
----------------

Public header files exported from the LLVM library. The three main subdirectories:

``llvm/include/llvm``

  All LLVM-specific header files, and  subdirectories for different portions of
  LLVM: ``Analysis``, ``CodeGen``, ``Target``, ``Transforms``, etc...

``llvm/include/llvm/Support``

  Generic support libraries provided with LLVM but not necessarily specific to
  LLVM. For example, some C++ STL utilities and a Command Line option processing
  library store header files here.

``llvm/include/llvm/Config``

  Header files configured by ``cmake``.  They wrap "standard" UNIX and
  C header files.  Source code can include these header files which
  automatically take care of the conditional #includes that ``cmake``
  generates.

``llvm/lib``
------------

Most source files are here. By putting code in libraries, LLVM makes it easy to
share code among the `tools`_.

``llvm/lib/IR/``

  Core LLVM source files that implement core classes like Instruction and
  BasicBlock.

``llvm/lib/AsmParser/``

  Source code for the LLVM assembly language parser library.

``llvm/lib/Bitcode/``

  Code for reading and writing bitcode.

``llvm/lib/Analysis/``

  A variety of program analyses, such as Call Graphs, Induction Variables,
  Natural Loop Identification, etc.

``llvm/lib/Transforms/``

  IR-to-IR program transformations, such as Aggressive Dead Code Elimination,
  Sparse Conditional Constant Propagation, Inlining, Loop Invariant Code Motion,
  Dead Global Elimination, and many others.

``llvm/lib/Target/``

  Files describing target architectures for code generation.  For example,
  ``llvm/lib/Target/X86`` holds the X86 machine description.

``llvm/lib/CodeGen/``

  The major parts of the code generator: Instruction Selector, Instruction
  Scheduling, and Register Allocation.

``llvm/lib/MC/``

  The libraries represent and process code at machine code level. Handles
  assembly and object-file emission.

``llvm/lib/ExecutionEngine/``

  Libraries for directly executing bitcode at runtime in interpreted and
  JIT-compiled scenarios.

``llvm/lib/Support/``

  Source code that corresponding to the header files in ``llvm/include/ADT/``
  and ``llvm/include/Support/``.

``llvm/bindings``
----------------------

Contains bindings for the LLVM compiler infrastructure to allow
programs written in languages other than C or C++ to take advantage of the LLVM
infrastructure.
LLVM project provides language bindings for Go, OCaml and Python.

``llvm/projects``
-----------------

Projects not strictly part of LLVM but shipped with LLVM. This is also the
directory for creating your own LLVM-based projects which leverage the LLVM
build system.

``llvm/test``
-------------

Feature and regression tests and other sanity checks on LLVM infrastructure. These
are intended to run quickly and cover a lot of territory without being exhaustive.

``test-suite``
--------------

A comprehensive correctness, performance, and benchmarking test suite
for LLVM.  This comes in a ``separate git repository
<https://github.com/llvm/llvm-test-suite>``, because it contains a
large amount of third-party code under a variety of licenses. For
details see the :doc:`Testing Guide <TestingGuide>` document.

.. _tools:

``llvm/tools``
--------------

Executables built out of the libraries
above, which form the main part of the user interface.  You can always get help
for a tool by typing ``tool_name -help``.  The following is a brief introduction
to the most important tools.  More detailed information is in
the `Command Guide <CommandGuide/index.html>`_.

``bugpoint``

  ``bugpoint`` is used to debug optimization passes or code generation backends
  by narrowing down the given test case to the minimum number of passes and/or
  instructions that still cause a problem, whether it is a crash or
  miscompilation. See `<HowToSubmitABug.html>`_ for more information on using
  ``bugpoint``.

``llvm-ar``

  The archiver produces an archive containing the given LLVM bitcode files,
  optionally with an index for faster lookup.

``llvm-as``

  The assembler transforms the human readable LLVM assembly to LLVM bitcode.

``llvm-dis``

  The disassembler transforms the LLVM bitcode to human readable LLVM assembly.

``llvm-link``

  ``llvm-link``, not surprisingly, links multiple LLVM modules into a single
  program.

``lli``

  ``lli`` is the LLVM interpreter, which can directly execute LLVM bitcode
  (although very slowly...). For architectures that support it (currently x86,
  Sparc, and PowerPC), by default, ``lli`` will function as a Just-In-Time
  compiler (if the functionality was compiled in), and will execute the code
  *much* faster than the interpreter.

``llc``

  ``llc`` is the LLVM backend compiler, which translates LLVM bitcode to a
  native code assembly file.

``opt``

  ``opt`` reads LLVM bitcode, applies a series of LLVM to LLVM transformations
  (which are specified on the command line), and outputs the resultant
  bitcode.   '``opt -help``'  is a good way to get a list of the
  program transformations available in LLVM.

  ``opt`` can also  run a specific analysis on an input LLVM bitcode
  file and print  the results.  Primarily useful for debugging
  analyses, or familiarizing yourself with what an analysis does.

``llvm/utils``
--------------

Utilities for working with LLVM source code; some are part of the build process
because they are code generators for parts of the infrastructure.


``codegen-diff``

  ``codegen-diff`` finds differences between code that LLC
  generates and code that LLI generates. This is useful if you are
  debugging one of them, assuming that the other generates correct output. For
  the full user manual, run ```perldoc codegen-diff'``.

``emacs/``

   Emacs and XEmacs syntax highlighting  for LLVM   assembly files and TableGen
   description files.  See the ``README`` for information on using them.

``getsrcs.sh``

  Finds and outputs all non-generated source files,
  useful if one wishes to do a lot of development across directories
  and does not want to find each file. One way to use it is to run,
  for example: ``xemacs `utils/getsources.sh``` from the top of the LLVM source
  tree.

``llvmgrep``

  Performs an ``egrep -H -n`` on each source file in LLVM and
  passes to it a regular expression provided on ``llvmgrep``'s command
  line. This is an efficient way of searching the source base for a
  particular regular expression.

``TableGen/``

  Contains the tool used to generate register
  descriptions, instruction set descriptions, and even assemblers from common
  TableGen description files.

``vim/``

  vim syntax-highlighting for LLVM assembly files
  and TableGen description files. See the    ``README`` for how to use them.

.. _simple example:

An Example Using the LLVM Tool Chain
====================================

This section gives an example of using LLVM with the Clang front end.

Example with clang
------------------

#. First, create a simple C file, name it 'hello.c':

   .. code-block:: c

     #include <stdio.h>

     int main() {
       printf("hello world\n");
       return 0;
     }

#. Next, compile the C file into a native executable:

   .. code-block:: console

     % clang hello.c -o hello

   .. note::

     Clang works just like GCC by default.  The standard -S and -c arguments
     work as usual (producing a native .s or .o file, respectively).

#. Next, compile the C file into an LLVM bitcode file:

   .. code-block:: console

     % clang -O3 -emit-llvm hello.c -c -o hello.bc

   The -emit-llvm option can be used with the -S or -c options to emit an LLVM
   ``.ll`` or ``.bc`` file (respectively) for the code.  This allows you to use
   the `standard LLVM tools <CommandGuide/index.html>`_ on the bitcode file.

#. Run the program in both forms. To run the program, use:

   .. code-block:: console

      % ./hello

   and

   .. code-block:: console

     % lli hello.bc

   The second examples shows how to invoke the LLVM JIT, :doc:`lli
   <CommandGuide/lli>`.

#. Use the ``llvm-dis`` utility to take a look at the LLVM assembly code:

   .. code-block:: console

     % llvm-dis < hello.bc | less

#. Compile the program to native assembly using the LLC code generator:

   .. code-block:: console

     % llc hello.bc -o hello.s

#. Assemble the native assembly language file into a program:

   .. code-block:: console

     % /opt/SUNWspro/bin/cc -xarch=v9 hello.s -o hello.native   # On Solaris

     % gcc hello.s -o hello.native                              # On others

#. Execute the native code program:

   .. code-block:: console

     % ./hello.native

   Note that using clang to compile directly to native code (i.e. when the
   ``-emit-llvm`` option is not present) does steps 6/7/8 for you.

Common Problems
===============

If you are having problems building or using LLVM, or if you have any other
general questions about LLVM, please consult the `Frequently Asked
Questions <FAQ.html>`_ page.

If you are having problems with limited memory and build time, please try
building with ninja instead of make. Please consider configuring the
following options with cmake:

 * -G Ninja
   Setting this option will allow you to build with ninja instead of make.
   Building with ninja significantly improves your build time, especially with
   incremental builds, and improves your memory usage.

 * -DLLVM_USE_LINKER
   Setting this option to lld will significantly reduce linking time for LLVM
   executables on ELF-based platforms, such as Linux. If you are building LLVM
   for the first time and lld is not available to you as a binary package, then
   you may want to use the gold linker as a faster alternative to GNU ld.

 * -DCMAKE_BUILD_TYPE
   Controls optimization level and debug information of the build.  This setting
   can affect RAM and disk usage, see :ref:`CMAKE_BUILD_TYPE <cmake_build_type>`
   for more information.

 * -DLLVM_ENABLE_ASSERTIONS
   This option defaults to ON for Debug builds and defaults to OFF for Release
   builds. As mentioned in the previous option, using the Release build type and
   enabling assertions may be a good alternative to using the Debug build type.

 * -DLLVM_PARALLEL_LINK_JOBS
   Set this equal to number of jobs you wish to run simultaneously. This is
   similar to the -j option used with make, but only for link jobs. This option
   can only be used with ninja. You may wish to use a very low number of jobs,
   as this will greatly reduce the amount of memory used during the build
   process. If you have limited memory, you may wish to set this to 1.

 * -DLLVM_TARGETS_TO_BUILD
   Set this equal to the target you wish to build. You may wish to set this to
   X86; however, you will find a full list of targets within the
   llvm-project/llvm/lib/Target directory.

 * -DLLVM_OPTIMIZED_TABLEGEN
   Set this to ON to generate a fully optimized tablegen during your build. This
   will significantly improve your build time. This is only useful if you are
   using the Debug build type.

 * -DLLVM_ENABLE_PROJECTS
   Set this equal to the projects you wish to compile (e.g. clang, lld, etc.) If
   compiling more than one project, separate the items with a semicolon. Should
   you run into issues with the semicolon, try surrounding it with single quotes.

 * -DLLVM_ENABLE_RUNTIMES
   Set this equal to the runtimes you wish to compile (e.g. libcxx, libcxxabi, etc.)
   If compiling more than one runtime, separate the items with a semicolon. Should
   you run into issues with the semicolon, try surrounding it with single quotes.

 * -DCLANG_ENABLE_STATIC_ANALYZER
   Set this option to OFF if you do not require the clang static analyzer. This
   should improve your build time slightly.

 * -DLLVM_USE_SPLIT_DWARF
   Consider setting this to ON if you require a debug build, as this will ease
   memory pressure on the linker. This will make linking much faster, as the
   binaries will not contain any of the debug information; however, this will
   generate the debug information in the form of a DWARF object file (with the
   extension .dwo). This only applies to host platforms using ELF, such as Linux.

.. _links:

Links
=====

This document is just an **introduction** on how to use LLVM to do some simple
things... there are many more interesting and complicated things that you can do
that aren't documented here (but we'll gladly accept a patch if you want to
write something up!).  For more information about LLVM, check out:

* `LLVM Homepage <https://llvm.org/>`_
* `LLVM Doxygen Tree <https://llvm.org/doxygen/>`_
* `Starting a Project that Uses LLVM <https://llvm.org/docs/Projects.html>`_

.. _installing arcanist: https://secure.phabricator.com/book/phabricator/article/arcanist_quick_start/
