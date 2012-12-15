==========================
Pretokenized Headers (PTH)
==========================

This document first describes the low-level interface for using PTH and
then briefly elaborates on its design and implementation. If you are
interested in the end-user view, please see the :ref:`User's Manual
<usersmanual-precompiled-headers>`.

Using Pretokenized Headers with ``clang`` (Low-level Interface)
===============================================================

The Clang compiler frontend, ``clang -cc1``, supports three command line
options for generating and using PTH files.

To generate PTH files using ``clang -cc1``, use the option ``-emit-pth``:

.. code-block:: console

  $ clang -cc1 test.h -emit-pth -o test.h.pth

This option is transparently used by ``clang`` when generating PTH
files. Similarly, PTH files can be used as prefix headers using the
``-include-pth`` option:

.. code-block:: console

  $ clang -cc1 -include-pth test.h.pth test.c -o test.s

Alternatively, Clang's PTH files can be used as a raw "token-cache" (or
"content" cache) of the source included by the original header file.
This means that the contents of the PTH file are searched as substitutes
for *any* source files that are used by ``clang -cc1`` to process a
source file. This is done by specifying the ``-token-cache`` option:

.. code-block:: console

  $ cat test.h
  #include <stdio.h>
  $ clang -cc1 -emit-pth test.h -o test.h.pth
  $ cat test.c
  #include "test.h"
  $ clang -cc1 test.c -o test -token-cache test.h.pth

In this example the contents of ``stdio.h`` (and the files it includes)
will be retrieved from ``test.h.pth``, as the PTH file is being used in
this case as a raw cache of the contents of ``test.h``. This is a
low-level interface used to both implement the high-level PTH interface
as well as to provide alternative means to use PTH-style caching.

PTH Design and Implementation
=============================

Unlike GCC's precompiled headers, which cache the full ASTs and
preprocessor state of a header file, Clang's pretokenized header files
mainly cache the raw lexer *tokens* that are needed to segment the
stream of characters in a source file into keywords, identifiers, and
operators. Consequently, PTH serves to mainly directly speed up the
lexing and preprocessing of a source file, while parsing and
type-checking must be completely redone every time a PTH file is used.

Basic Design Tradeoffs
~~~~~~~~~~~~~~~~~~~~~~

In the long term there are plans to provide an alternate PCH
implementation for Clang that also caches the work for parsing and type
checking the contents of header files. The current implementation of PCH
in Clang as pretokenized header files was motivated by the following
factors:

**Language independence**
   PTH files work with any language that
   Clang's lexer can handle, including C, Objective-C, and (in the early
   stages) C++. This means development on language features at the
   parsing level or above (which is basically almost all interesting
   pieces) does not require PTH to be modified.

**Simple design**
   Relatively speaking, PTH has a simple design and
   implementation, making it easy to test. Further, because the
   machinery for PTH resides at the lower-levels of the Clang library
   stack it is fairly straightforward to profile and optimize.

Further, compared to GCC's PCH implementation (which is the dominate
precompiled header file implementation that Clang can be directly
compared against) the PTH design in Clang yields several attractive
features:

**Architecture independence**
   In contrast to GCC's PCH files (and
   those of several other compilers), Clang's PTH files are architecture
   independent, requiring only a single PTH file when building an
   program for multiple architectures.

   For example, on Mac OS X one may wish to compile a "universal binary"
   that runs on PowerPC, 32-bit Intel (i386), and 64-bit Intel
   architectures. In contrast, GCC requires a PCH file for each
   architecture, as the definitions of types in the AST are
   architecture-specific. Since a Clang PTH file essentially represents
   a lexical cache of header files, a single PTH file can be safely used
   when compiling for multiple architectures. This can also reduce
   compile times because only a single PTH file needs to be generated
   during a build instead of several.

**Reduced memory pressure**
   Similar to GCC, Clang reads PTH files
   via the use of memory mapping (i.e., ``mmap``). Clang, however,
   memory maps PTH files as read-only, meaning that multiple invocations
   of ``clang -cc1`` can share the same pages in memory from a
   memory-mapped PTH file. In comparison, GCC also memory maps its PCH
   files but also modifies those pages in memory, incurring the
   copy-on-write costs. The read-only nature of PTH can greatly reduce
   memory pressure for builds involving multiple cores, thus improving
   overall scalability.

**Fast generation**
   PTH files can be generated in a small fraction
   of the time needed to generate GCC's PCH files. Since PTH/PCH
   generation is a serial operation that typically blocks progress
   during a build, faster generation time leads to improved processor
   utilization with parallel builds on multicore machines.

Despite these strengths, PTH's simple design suffers some algorithmic
handicaps compared to other PCH strategies such as those used by GCC.
While PTH can greatly speed up the processing time of a header file, the
amount of work required to process a header file is still roughly linear
in the size of the header file. In contrast, the amount of work done by
GCC to process a precompiled header is (theoretically) constant (the
ASTs for the header are literally memory mapped into the compiler). This
means that only the pieces of the header file that are referenced by the
source file including the header are the only ones the compiler needs to
process during actual compilation. While GCC's particular implementation
of PCH mitigates some of these algorithmic strengths via the use of
copy-on-write pages, the approach itself can fundamentally dominate at
an algorithmic level, especially when one considers header files of
arbitrary size.

There are plans to potentially implement an complementary PCH
implementation for Clang based on the lazy deserialization of ASTs. This
approach would theoretically have the same constant-time algorithmic
advantages just mentioned but would also retain some of the strengths of
PTH such as reduced memory pressure (ideal for multi-core builds).

Internal PTH Optimizations
~~~~~~~~~~~~~~~~~~~~~~~~~~

While the main optimization employed by PTH is to reduce lexing time of
header files by caching pre-lexed tokens, PTH also employs several other
optimizations to speed up the processing of header files:

-  ``stat`` caching: PTH files cache information obtained via calls to
   ``stat`` that ``clang -cc1`` uses to resolve which files are included
   by ``#include`` directives. This greatly reduces the overhead
   involved in context-switching to the kernel to resolve included
   files.

-  Fasting skipping of ``#ifdef``... ``#endif`` chains: PTH files
   record the basic structure of nested preprocessor blocks. When the
   condition of the preprocessor block is false, all of its tokens are
   immediately skipped instead of requiring them to be handled by
   Clang's preprocessor.


