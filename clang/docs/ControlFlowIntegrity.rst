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

Bad Cast Checking
-----------------

This scheme checks that pointer casts are made to an object of the correct
dynamic type; that is, the dynamic type of the object must be a derived class
of the pointee type of the cast. The checks are currently only introduced
where the class being casted to is a polymorphic class.

Bad casts are not in themselves control flow integrity violations, but they
can also create security vulnerabilities, and the implementation uses many
of the same mechanisms.

There are two types of bad cast that may be forbidden: bad casts
from a base class to a derived class (which can be checked with
``-fsanitize=cfi-derived-cast``), and bad casts from a pointer of
type ``void*`` or another unrelated type (which can be checked with
``-fsanitize=cfi-unrelated-cast``).

The difference between these two types of casts is that the first is defined
by the C++ standard to produce an undefined value, while the second is not
in itself undefined behavior (it is well defined to cast the pointer back
to its original type).

If a program as a matter of policy forbids the second type of cast, that
restriction can normally be enforced. However it may in some cases be necessary
for a function to perform a forbidden cast to conform with an external API
(e.g. the ``allocate`` member function of a standard library allocator). Such
functions may be blacklisted using a :doc:`SanitizerSpecialCaseList`.

For this scheme to work, all translation units containing the definition
of a virtual member function (whether inline or not) must be compiled with
``-fsanitize=cfi-derived-cast`` or ``-fsanitize=cfi-unrelated-cast`` enabled
and be statically linked into the program. Classes in the C++ standard library
(under namespace ``std``) are exempted from checking, and therefore programs
may be linked against a pre-built standard library, but this may change in
the future.

.. _cfi-strictness:

Strictness
~~~~~~~~~~

If a class has a single non-virtual base and does not introduce or override
virtual member functions or fields other than an implicitly defined virtual
destructor, it will have the same layout and virtual function semantics as
its base. By default, casts to such classes are checked as if they were made
to the least derived such class.

Casting an instance of a base class to such a derived class is technically
undefined behavior, but it is a relatively common hack for introducing
member functions on class instances with specific properties that works under
most compilers and should not have security implications, so we allow it by
default. It can be disabled with ``-fsanitize=cfi-cast-strict``.

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
