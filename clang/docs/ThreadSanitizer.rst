ThreadSanitizer
===============

Introduction
------------

ThreadSanitizer is a tool that detects data races.  It consists of a compiler
instrumentation module and a run-time library.  Typical slowdown introduced by
ThreadSanitizer is **5x-15x** (TODO: these numbers are approximate so far).

How to build
------------

Follow the `Clang build instructions <../get_started.html>`_.  CMake build is
supported.

Supported Platforms
-------------------

ThreadSanitizer is supported on Linux x86_64 (tested on Ubuntu 10.04).  Support
for MacOS 10.7 (64-bit only) is planned for late 2012.  Support for 32-bit
platforms is problematic and not yet planned.

Usage
-----

Simply compile your program with ``-fsanitize=thread -fPIE`` and link it with
``-fsanitize=thread -pie``.  To get a reasonable performance add ``-O1`` or
higher.  Use ``-g`` to get file names and line numbers in the warning messages.

Example:

.. code-block:: c++

  % cat projects/compiler-rt/lib/tsan/output_tests/tiny_race.c
  #include <pthread.h>
  int Global;
  void *Thread1(void *x) {
    Global = 42;
    return x;
  }
  int main() {
    pthread_t t;
    pthread_create(&t, NULL, Thread1, NULL);
    Global = 43;
    pthread_join(t, NULL);
    return Global;
  }

  $ clang -fsanitize=thread -g -O1 tiny_race.c -fPIE -pie

If a bug is detected, the program will print an error message to stderr.
Currently, ThreadSanitizer symbolizes its output using an external
``addr2line`` process (this will be fixed in future).

.. code-block:: bash

  % TSAN_OPTIONS=strip_path_prefix=`pwd`/  # Don't print full paths.
  % ./a.out 2> log
  % cat log
  WARNING: ThreadSanitizer: data race (pid=19219)
    Write of size 4 at 0x7fcf47b21bc0 by thread 1:
      #0 Thread1 tiny_race.c:4 (exe+0x00000000a360)
    Previous write of size 4 at 0x7fcf47b21bc0 by main thread:
      #0 main tiny_race.c:10 (exe+0x00000000a3b4)
    Thread 1 (running) created at:
      #0 pthread_create ??:0 (exe+0x00000000c790)
      #1 main tiny_race.c:9 (exe+0x00000000a3a4)

Limitations
-----------

* ThreadSanitizer uses more real memory than a native run. At the default
  settings the memory overhead is 9x plus 9Mb per each thread. Settings with 5x
  and 3x overhead (but less accurate analysis) are also available.
* ThreadSanitizer maps (but does not reserve) a lot of virtual address space.
  This means that tools like ``ulimit`` may not work as usually expected.
* Static linking is not supported.
* ThreadSanitizer requires ``-fPIE -pie``.

Current Status
--------------

ThreadSanitizer is in alpha stage.  It is known to work on large C++ programs
using pthreads, but we do not promise anything (yet).  C++11 threading is not
yet supported.  The test suite is integrated into CMake build and can be run
with ``make check-tsan`` command.

We are actively working on enhancing the tool --- stay tuned.  Any help,
especially in the form of minimized standalone tests is more than welcome.

More Information
----------------
`http://code.google.com/p/thread-sanitizer <http://code.google.com/p/thread-sanitizer/>`_.

