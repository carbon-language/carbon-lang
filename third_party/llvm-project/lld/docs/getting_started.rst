.. _getting_started:

Getting Started: Building and Running lld
=========================================

This page gives you the shortest path to checking out and building lld. If you
run into problems, please file bugs in the `LLVM Bugzilla`__

__ https://bugs.llvm.org/

Building lld
------------

On Unix-like Systems
~~~~~~~~~~~~~~~~~~~~

1. Get the required tools.

  * `CMake 2.8`_\+.
  * make (or any build system CMake supports).
  * `Clang 3.1`_\+ or GCC 4.7+ (C++11 support is required).

    * If using Clang, you will also need `libc++`_.
  * `Python 2.4`_\+ (not 3.x) for running tests.

.. _CMake 2.8: http://www.cmake.org/cmake/resources/software.html
.. _Clang 3.1: http://clang.llvm.org/
.. _libc++: http://libcxx.llvm.org/
.. _Python 2.4: http://python.org/download/

2. Check out LLVM and subprojects (including lld)::

     $ git clone https://github.com/llvm/llvm-project.git

4. Build LLVM and lld::

     $ cd llvm-project
     $ mkdir build && cd build
     $ cmake -G "Unix Makefiles" -DLLVM_ENABLE_PROJECTS=lld ../llvm
     $ make

  * If you want to build with clang and it is not the default compiler or
    it is installed in an alternate location, you'll need to tell the cmake tool
    the location of the C and C++ compiler via CMAKE_C_COMPILER and
    CMAKE_CXX_COMPILER. For example::

        $ cmake -DCMAKE_CXX_COMPILER=/path/to/clang++ -DCMAKE_C_COMPILER=/path/to/clang ...

5. Test::

     $ make check-lld

Using Visual Studio
~~~~~~~~~~~~~~~~~~~

#. Get the required tools.

  * `CMake 2.8`_\+.
  * `Visual Studio 12 (2013) or later`_ (required for C++11 support)
  * `Python 2.4`_\+ (not 3.x) for running tests.

.. _CMake 2.8: http://www.cmake.org/cmake/resources/software.html
.. _Visual Studio 12 (2013) or later: http://www.microsoft.com/visualstudio/11/en-us
.. _Python 2.4: http://python.org/download/

#. Check out LLVM as above.

#. Generate Visual Studio project files::

     $ cd llvm-project/build (out of source build required)
     $ cmake -G "Visual Studio 11" -DLLVM_ENABLE_PROJECTS=lld ../llvm

#. Build

  * Open LLVM.sln in Visual Studio.
  * Build the ``ALL_BUILD`` target.

#. Test

  * Build the ``lld-test`` target.

More Information
~~~~~~~~~~~~~~~~

For more information on using CMake see the `LLVM CMake guide`_.

.. _LLVM CMake guide: https://llvm.org/docs/CMake.html
