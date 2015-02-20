======================
Control Flow Integrity
======================

.. toctree::
   :hidden:

   ControlFlowIntegrityDesign

.. contents::
   :local:

Introduction
============

Clang includes an implementation of a number of control flow integrity (CFI)
schemes, which are designed to abort the program upon detecting certain forms
of undefined behavior that can potentially allow attackers to subvert the
program's control flow. These schemes have been optimized for performance,
allowing developers to enable them in release builds.

To enable Clang's available CFI schemes, use the flag ``-fsanitize=cfi``.
As currently implemented, CFI relies on link-time optimization (LTO); the CFI
schemes imply ``-flto``, and the linker used must support LTO, for example
via the `gold plugin`_. To allow the checks to be implemented efficiently,
the program must be structured such that certain object files are compiled
with CFI enabled, and are statically linked into the program. This may
preclude the use of shared libraries in some cases.

Clang currently implements forward-edge CFI for virtual calls. More schemes
are under development.

.. _gold plugin: http://llvm.org/docs/GoldPlugin.html

Forward-Edge CFI for Virtual Calls
----------------------------------

This scheme checks that virtual calls take place using a vptr of the correct
dynamic type; that is, the dynamic type of the called object must be a
derived class of the static type of the object used to make the call.
This CFI scheme can be enabled on its own using ``-fsanitize=cfi-vptr``.

For this scheme to work, all translation units containing the definition
of a virtual member function (whether inline or not) must be compiled
with ``-fsanitize=cfi-vptr`` enabled and be statically linked into the
program. Classes in the C++ standard library (under namespace ``std``) are
exempted from checking, and therefore programs may be linked against a
pre-built standard library, but this may change in the future.

Performance
~~~~~~~~~~~

A performance overhead of less than 1% has been measured by running the
Dromaeo benchmark suite against an instrumented version of the Chromium
web browser. Another good performance benchmark for this mechanism is the
virtual-call-heavy SPEC 2006 xalancbmk.

Note that this scheme has not yet been optimized for binary size; an increase
of up to 15% has been observed for Chromium.

Design
------

Please refer to the :doc:`design document<ControlFlowIntegrityDesign>`.

Publications
------------

`Control-Flow Integrity: Principles, Implementations, and Applications <http://research.microsoft.com/pubs/64250/ccs05.pdf>`_.
Martin Abadi, Mihai Budiu, Úlfar Erlingsson, Jay Ligatti.

`Enforcing Forward-Edge Control-Flow Integrity in GCC & LLVM <http://www.pcc.me.uk/~peter/acad/usenix14.pdf>`_.
Caroline Tice, Tom Roeder, Peter Collingbourne, Stephen Checkoway,
Úlfar Erlingsson, Luis Lozano, Geoff Pike.
