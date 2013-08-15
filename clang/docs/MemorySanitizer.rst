================
MemorySanitizer
================

.. contents::
   :local:

Introduction
============

MemorySanitizer is a detector of uninitialized reads. It consists of a
compiler instrumentation module and a run-time library.

Typical slowdown introduced by MemorySanitizer is **3x**.

How to build
============

Follow the `clang build instructions <../get_started.html>`_. CMake
build is supported.

Usage
=====

Simply compile and link your program with ``-fsanitize=memory`` flag.
The MemorySanitizer run-time library should be linked to the final
executable, so make sure to use ``clang`` (not ``ld``) for the final
link step. When linking shared libraries, the MemorySanitizer run-time
is not linked, so ``-Wl,-z,defs`` may cause link errors (don't use it
with MemorySanitizer). To get a reasonable performance add ``-O1`` or
higher. To get meaninful stack traces in error messages add
``-fno-omit-frame-pointer``. To get perfect stack traces you may need
to disable inlining (just use ``-O1``) and tail call elimination
(``-fno-optimize-sibling-calls``).

.. code-block:: console

    % cat umr.cc
    #include <stdio.h>

    int main(int argc, char** argv) {
      int* a = new int[10];
      a[5] = 0;
      if (a[argc])
        printf("xx\n");
      return 0;
    }

    % clang -fsanitize=memory -fno-omit-frame-pointer -g -O2 umr.cc

If a bug is detected, the program will print an error message to
stderr and exit with a non-zero exit code. Currently, MemorySanitizer
does not symbolize its output by default, so you may need to use a
separate script to symbolize the result offline (this will be fixed in
future).

.. code-block:: console

    % ./a.out 2>log
    % projects/compiler-rt/lib/asan/scripts/asan_symbolize.py / < log | c++filt
    ==30106==  WARNING: MemorySanitizer: UMR (uninitialized-memory-read)
        #0 0x7f45944b418a in main umr.cc:6
        #1 0x7f45938b676c in __libc_start_main libc-start.c:226
    Exiting

By default, MemorySanitizer exits on the first detected error.

``__has_feature(memory_sanitizer)``
------------------------------------

In some cases one may need to execute different code depending on
whether MemorySanitizer is enabled. :ref:`\_\_has\_feature
<langext-__has_feature-__has_extension>` can be used for this purpose.

.. code-block:: c

    #if defined(__has_feature)
    #  if __has_feature(memory_sanitizer)
    // code that builds only under MemorySanitizer
    #  endif
    #endif

``__attribute__((no_sanitize_memory))``
-----------------------------------------------

Some code should not be checked by MemorySanitizer.
One may use the function attribute
:ref:`no_sanitize_memory <langext-memory_sanitizer>`
to disable uninitialized checks in a particular function.
MemorySanitizer may still instrument such functions to avoid false positives.
This attribute may not be
supported by other compilers, so we suggest to use it together with
``__has_feature(memory_sanitizer)``.

Blacklist
---------

MemorySanitizer supports ``src`` and ``fun`` entity types in
:doc:`SanitizerSpecialCaseList`, that can be used to relax MemorySanitizer
checks for certain source files and functions. All "Use of uninitialized value"
warnings will be suppressed and all values loaded from memory will be
considered fully initialized.

Origin Tracking
===============

MemorySanitizer can track origins of unitialized values, similar to
Valgrind's --track-origins option. This feature is enabled by
``-fsanitize-memory-track-origins`` Clang option. With the code from
the example above,

.. code-block:: console

    % clang -fsanitize=memory -fsanitize-memory-track-origins -fno-omit-frame-pointer -g -O2 umr.cc
    % ./a.out 2>log
    % projects/compiler-rt/lib/asan/scripts/asan_symbolize.py / < log | c++filt
    ==14425==  WARNING: MemorySanitizer: UMR (uninitialized-memory-read)
    ==14425== WARNING: Trying to symbolize code, but external symbolizer is not initialized!
        #0 0x7f8bdda3824b in main umr.cc:6
        #1 0x7f8bdce3a76c in __libc_start_main libc-start.c:226
      raw origin id: 2030043137
      ORIGIN: heap allocation:
        #0 0x7f8bdda4034b in operator new[](unsigned long) msan_new_delete.cc:39
        #1 0x7f8bdda3814d in main umr.cc:4
        #2 0x7f8bdce3a76c in __libc_start_main libc-start.c:226
    Exiting

Origin tracking has proved to be very useful for debugging UMR
reports. It slows down program execution by a factor of 1.5x-2x on top
of the usual MemorySanitizer slowdown.

Handling external code
============================

MemorySanitizer requires that all program code is instrumented. This
also includes any libraries that the program depends on, even libc.
Failing to achieve this may result in false UMR reports.

Full MemorySanitizer instrumentation is very difficult to achieve. To
make it easier, MemorySanitizer runtime library includes 70+
interceptors for the most common libc functions. They make it possible
to run MemorySanitizer-instrumented programs linked with
uninstrumented libc. For example, the authors were able to bootstrap
MemorySanitizer-instrumented Clang compiler by linking it with
self-built instrumented libcxx (as a replacement for libstdc++).

In the case when rebuilding all program dependencies with
MemorySanitizer is problematic, an experimental MSanDR tool can be
used. It is a DynamoRio-based tool that uses dynamic instrumentation
to avoid false positives due to uninstrumented code. The tool simply
marks memory from instrumented libraries as fully initialized. See
`http://code.google.com/p/memory-sanitizer/wiki/Running#Running_with_the_dynamic_tool`
for more information.

Supported Platforms
===================

MemorySanitizer is supported on

* Linux x86\_64 (tested on Ubuntu 10.04 and 12.04);

Limitations
===========

* MemorySanitizer uses 2x more real memory than a native run, 3x with
  origin tracking.
* MemorySanitizer maps (but not reserves) 64 Terabytes of virtual
  address space. This means that tools like ``ulimit`` may not work as
  usually expected.
* Static linking is not supported.
* Non-position-independent executables are not supported.  Therefore, the
  ``fsanitize=memory`` flag will cause Clang to act as though the ``-fPIE``
  flag had been supplied if compiling without ``-fPIC``, and as though the
  ``-pie`` flag had been supplied if linking an executable.
* Depending on the version of Linux kernel, running without ASLR may
  be not supported. Note that GDB disables ASLR by default. To debug
  instrumented programs, use "set disable-randomization off".

Current Status
==============

MemorySanitizer is an experimental tool. It is known to work on large
real-world programs, like Clang/LLVM itself.

More Information
================

`http://code.google.com/p/memory-sanitizer <http://code.google.com/p/memory-sanitizer/>`_

