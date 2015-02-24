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
the first is a metadata string containing an identifier for the bitset,
the second is a global variable and the third is a byte offset into the
global variable.

This will cause a link-time optimization pass to generate bitsets from the
memory addresses referenced from the elements of the bitset metadata. The pass
will lay out the referenced globals consecutively, so their definitions must
be available at LTO time. The `GlobalLayoutBuilder`_ class is responsible for
laying out the globals efficiently to minimize the sizes of the underlying
bitsets. An intrinsic, :ref:`llvm.bitset.test <bitset.test>`, generates code
to test whether a given pointer is a member of a bitset.

:Example:

::

    target datalayout = "e-p:32:32"

    @a = internal global i32 0
    @b = internal global i32 0
    @c = internal global i32 0
    @d = internal global [2 x i32] [i32 0, i32 0]

    !llvm.bitsets = !{!0, !1, !2, !3, !4}

    !0 = !{!"bitset1", i32* @a, i32 0}
    !1 = !{!"bitset1", i32* @b, i32 0}
    !2 = !{!"bitset2", i32* @b, i32 0}
    !3 = !{!"bitset2", i32* @c, i32 0}
    !4 = !{!"bitset2", i32* @d, i32 4}

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

    define void @main() {
      %a1 = call i1 @foo(i32* @a) ; returns 1
      %b1 = call i1 @foo(i32* @b) ; returns 1
      %c1 = call i1 @foo(i32* @c) ; returns 0
      %a2 = call i1 @bar(i32* @a) ; returns 0
      %b2 = call i1 @bar(i32* @b) ; returns 1
      %c2 = call i1 @bar(i32* @c) ; returns 1
      %d02 = call i1 @bar(i32* getelementptr ([2 x i32]* @d, i32 0, i32 0)) ; returns 0
      %d12 = call i1 @bar(i32* getelementptr ([2 x i32]* @d, i32 0, i32 1)) ; returns 1
      ret void
    }

.. _GlobalLayoutBuilder: http://llvm.org/klaus/llvm/blob/master/include/llvm/Transforms/IPO/LowerBitSets.h
