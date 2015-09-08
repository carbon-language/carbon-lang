=======
Bitsets
=======

This is a mechanism that allows IR modules to co-operatively build pointer
sets corresponding to addresses within a given set of globals. One example
of a use case for this is to allow a C++ program to efficiently verify (at
each call site) that a vtable pointer is in the set of valid vtable pointers
for the type of the class or its derived classes.

To use the mechanism, a client creates a global metadata node named
``llvm.bitsets``.  Each element is a metadata node with three elements:

1. a metadata object representing an identifier for the bitset
2. either a global variable or a function
3. a byte offset into the global (generally zero for functions)

Each bitset must exclusively contain either global variables or functions.

.. admonition:: Limitation

  The current implementation only supports functions as members of bitsets on
  the x86-32 and x86-64 architectures.

This will cause a link-time optimization pass to generate bitsets from the
memory addresses referenced from the elements of the bitset metadata. The
pass will lay out referenced global variables consecutively, so their
definitions must be available at LTO time.

A bit set containing functions is transformed into a jump table, which
is a block of code consisting of one branch instruction for each of the
functions in the bit set that branches to the target function, and redirect
any taken function addresses to the corresponding jump table entry. In the
object file's symbol table, the jump table entries take the identities of
the original functions, so that addresses taken outside the module will pass
any verification done inside the module.

Jump tables may call external functions, so their definitions need not
be available at LTO time. Note that if an externally defined function is a
member of a bitset, there is no guarantee that its identity within the module
will be the same as its identity outside of the module, as the former will
be the jump table entry if a jump table is necessary.

The `GlobalLayoutBuilder`_ class is responsible for laying out the globals
efficiently to minimize the sizes of the underlying bitsets. An intrinsic,
:ref:`llvm.bitset.test <bitset.test>`, generates code to test whether a
given pointer is a member of a bitset.

:Example:

::

    target datalayout = "e-p:32:32"

    @a = internal global i32 0
    @b = internal global i32 0
    @c = internal global i32 0
    @d = internal global [2 x i32] [i32 0, i32 0]

    define void @e() {
      ret void
    }

    define void @f() {
      ret void
    }

    declare void @g()

    !llvm.bitsets = !{!0, !1, !2, !3, !4, !5, !6}

    !0 = !{!"bitset1", i32* @a, i32 0}
    !1 = !{!"bitset1", i32* @b, i32 0}
    !2 = !{!"bitset2", i32* @b, i32 0}
    !3 = !{!"bitset2", i32* @c, i32 0}
    !4 = !{!"bitset2", i32* @d, i32 4}
    !5 = !{!"bitset3", void ()* @e, i32 0}
    !6 = !{!"bitset3", void ()* @g, i32 0}

    declare i1 @llvm.bitset.test(i8* %ptr, metadata %bitset) nounwind readnone

    define i1 @foo(i32* %p) {
      %pi8 = bitcast i32* %p to i8*
      %x = call i1 @llvm.bitset.test(i8* %pi8, metadata !"bitset1")
      ret i1 %x
    }

    define i1 @bar(i32* %p) {
      %pi8 = bitcast i32* %p to i8*
      %x = call i1 @llvm.bitset.test(i8* %pi8, metadata !"bitset2")
      ret i1 %x
    }

    define i1 @baz(void ()* %p) {
      %pi8 = bitcast void ()* %p to i8*
      %x = call i1 @llvm.bitset.test(i8* %pi8, metadata !"bitset3")
      ret i1 %x
    }

    define void @main() {
      %a1 = call i1 @foo(i32* @a) ; returns 1
      %b1 = call i1 @foo(i32* @b) ; returns 1
      %c1 = call i1 @foo(i32* @c) ; returns 0
      %a2 = call i1 @bar(i32* @a) ; returns 0
      %b2 = call i1 @bar(i32* @b) ; returns 1
      %c2 = call i1 @bar(i32* @c) ; returns 1
      %d02 = call i1 @bar(i32* getelementptr ([2 x i32]* @d, i32 0, i32 0)) ; returns 0
      %d12 = call i1 @bar(i32* getelementptr ([2 x i32]* @d, i32 0, i32 1)) ; returns 1
      %e = call i1 @baz(void ()* @e) ; returns 1
      %f = call i1 @baz(void ()* @f) ; returns 0
      %g = call i1 @baz(void ()* @g) ; returns 1
      ret void
    }

.. _GlobalLayoutBuilder: http://llvm.org/klaus/llvm/blob/master/include/llvm/Transforms/IPO/LowerBitSets.h
