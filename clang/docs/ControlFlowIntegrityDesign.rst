===========================================
Control Flow Integrity Design Documentation
===========================================

This page documents the design of the :doc:`ControlFlowIntegrity` schemes
supported by Clang.

Forward-Edge CFI for Virtual Calls
----------------------------------

This scheme works by allocating, for each static type used to make a virtual
call, a region of read-only storage in the object file holding a bit vector
that maps onto to the region of storage used for those virtual tables. Each
set bit in the bit vector corresponds to the `address point`_ for a virtual
table compatible with the static type for which the bit vector is being built.

For example, consider the following three C++ classes:

.. code-block:: c++

  struct A {
    virtual void f();
  };

  struct B : A {
    virtual void f();
  };

  struct C : A {
    virtual void f();
  };

The scheme will cause the virtual tables for A, B and C to be laid out
consecutively:

.. csv-table:: Virtual Table Layout for A, B, C
  :header: 0, 1, 2, 3, 4, 5, 6, 7, 8

  A::offset-to-top, &A::rtti, &A::f, B::offset-to-top, &B::rtti, &B::f, C::offset-to-top, &C::rtti, &C::f

The bit vector for static types A, B and C will look like this:

.. csv-table:: Bit Vectors for A, B, C
  :header: Class, 0, 1, 2, 3, 4, 5, 6, 7, 8

  A, 0, 0, 1, 0, 0, 1, 0, 0, 1
  B, 0, 0, 0, 0, 0, 1, 0, 0, 0
  C, 0, 0, 0, 0, 0, 0, 0, 0, 1

To emit a virtual call, the compiler will assemble code that checks that
the object's virtual table pointer is in-bounds and aligned and that the
relevant bit is set in the bit vector.

The compiler relies on co-operation from the linker in order to assemble
the bit vector for the whole program. It currently does this using LLVM's
`bit sets`_ mechanism together with link-time optimization.

.. _address point: https://mentorembedded.github.io/cxx-abi/abi.html#vtable-general
.. _bit sets: http://llvm.org/docs/BitSets.html
