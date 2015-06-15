=========
SafeStack
=========

.. contents::
   :local:

Introduction
============

SafeStack is an instrumentation pass that protects programs against attacks
based on stack buffer overflows, without introducing any measurable performance
overhead. It works by separating the program stack into two distinct regions:
the safe stack and the unsafe stack. The safe stack stores return addresses,
register spills, and local variables that are always accessed in a safe way,
while the unsafe stack stores everything else. This separation ensures that
buffer overflows on the unsafe stack cannot be used to overwrite anything
on the safe stack, which includes return addresses.

Performance
-----------

The performance overhead of the SafeStack instrumentation is less than 0.1% on
average across a variety of benchmarks (see the `Code-Pointer Integrity
<http://dslab.epfl.ch/pubs/cpi.pdf>`_ paper for details). This is mainly
because most small functions do not have any variables that require the unsafe
stack and, hence, do not need unsafe stack frames to be created. The cost of
creating unsafe stack frames for large functions is amortized by the cost of
executing the function.

In some cases, SafeStack actually improves the performance. Objects that end up
being moved to the unsafe stack are usually large arrays or variables that are
used through multiple stack frames. Moving such objects away from the safe
stack increases the locality of frequently accessed values on the stack, such
as register spills, return addresses, and small local variables.

Limitations
-----------

SafeStack has not been subjected to a comprehensive security review, and there
exist known weaknesses, including but not limited to the following.

In its current state, the separation of local variables provides protection
against stack buffer overflows, but the safe stack itself is not protected
from being corrupted through a pointer dereference. The Code-Pointer
Integrity paper describes two ways in which we may protect the safe stack:
hardware segmentation on the 32-bit x86 architecture or information hiding
on other architectures.

Even with information hiding, the safe stack would merely be hidden
from attackers by being somewhere in the address space. Depending on the
application, the address could be predictable even on 64-bit address spaces
because not all the bits are addressable, multiple threads each have their
stack, the application could leak the safe stack address to memory via
``__builtin_frame_address``, bugs in the low-level runtime support etc.
Safe stack leaks could be mitigated by writing and deploying a static binary
analysis or a dynamic binary instrumentation based tool to find leaks.

This approach doesn't prevent an attacker from "imbalancing" the safe
stack by say having just one call, and doing two rets (thereby returning
to an address that wasn't meant as a return address). This can be at least
partially mitigated by deploying SafeStack alongside a forward control-flow
integrity mechanism to ensure that calls are made using the correct calling
convention. Clang does not currently implement a comprehensive forward
control-flow integrity protection scheme; there exists one that protects
:doc:`virtual calls <ControlFlowIntegrity>` but not non-virtual indirect calls.

Compatibility
-------------

Most programs, static libraries, or individual files can be compiled
with SafeStack as is. SafeStack requires basic runtime support, which, on most
platforms, is implemented as a compiler-rt library that is automatically linked
in when the program is compiled with SafeStack.

Linking a DSO with SafeStack is not currently supported.

Known compatibility limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Certain code that relies on low-level stack manipulations requires adaption to
work with SafeStack. One example is mark-and-sweep garbage collection
implementations for C/C++ (e.g., Oilpan in chromium/blink), which must be
changed to look for the live pointers on both safe and unsafe stacks.

SafeStack supports linking together modules that are compiled with and without
SafeStack, both statically and dynamically. One corner case that is not
supported is using ``dlopen()`` to load a dynamic library that uses SafeStack into
a program that is not compiled with SafeStack but uses threads.

Signal handlers that use ``sigaltstack()`` must not use the unsafe stack (see
``__attribute__((no_sanitize("safe-stack")))`` below).

Programs that use APIs from ``ucontext.h`` are not supported yet.

Usage
=====

To enable SafeStack, just pass ``-fsanitize=safe-stack`` flag to both compile and link
command lines.

Supported Platforms
-------------------

SafeStack was tested on Linux, FreeBSD and MacOSX.

Low-level API
-------------

``__has_feature(safe_stack)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some rare cases one may need to execute different code depending on
whether SafeStack is enabled. The macro ``__has_feature(safe_stack)`` can
be used for this purpose.

.. code-block:: c

    #if __has_feature(safe_stack)
    // code that builds only under SafeStack
    #endif

``__attribute__((no_sanitize("safe-stack")))``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``__attribute__((no_sanitize("safe-stack")))`` on a function declaration
to specify that the safe stack instrumentation should not be applied to that
function, even if enabled globally (see ``-fsanitize=safe-stack`` flag). This
attribute may be required for functions that make assumptions about the
exact layout of their stack frames.

Care should be taken when using this attribute. The return address is not
protected against stack buffer overflows, and it is easier to leak the
address of the safe stack to memory by taking the address of a local variable.


``__builtin___get_unsafe_stack_ptr()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This builtin function returns current unsafe stack pointer of the current
thread.

``__builtin___get_unsafe_stack_start()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This builtin function returns a pointer to the start of the unsafe stack of the
current thread.

Design
======

Please refer to
`http://dslab.epfl.ch/proj/cpi/ <http://dslab.epfl.ch/proj/cpi/>`_ for more
information about the design of the SafeStack and its related technologies.


Publications
------------

`Code-Pointer Integrity <http://dslab.epfl.ch/pubs/cpi.pdf>`_.
Volodymyr Kuznetsov, Laszlo Szekeres, Mathias Payer, George Candea, R. Sekar, Dawn Song.
USENIX Symposium on Operating Systems Design and Implementation
(`OSDI <https://www.usenix.org/conference/osdi14>`_), Broomfield, CO, October 2014
