//===- README_ALTIVEC.txt - Notes for improving Altivec code gen ----------===//

Implement PPCInstrInfo::isLoadFromStackSlot/isStoreToStackSlot for vector
registers, to generate better spill code.

//===----------------------------------------------------------------------===//

Altivec support.  The first should be a single lvx from the constant pool, the
second should be a xor/stvx:

void foo(void) {
  int x[8] __attribute__((aligned(128))) = { 1, 1, 1, 17, 1, 1, 1, 1 };
  bar (x);
}

#include <string.h>
void foo(void) {
  int x[8] __attribute__((aligned(128)));
  memset (x, 0, sizeof (x));
  bar (x);
}

//===----------------------------------------------------------------------===//

Altivec: Codegen'ing MUL with vector FMADD should add -0.0, not 0.0:
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=8763

When -ffast-math is on, we can use 0.0.

//===----------------------------------------------------------------------===//

  Consider this:
  v4f32 Vector;
  v4f32 Vector2 = { Vector.X, Vector.X, Vector.X, Vector.X };

Since we know that "Vector" is 16-byte aligned and we know the element offset 
of ".X", we should change the load into a lve*x instruction, instead of doing
a load/store/lve*x sequence.

//===----------------------------------------------------------------------===//

There are a wide range of vector constants we can generate with combinations of
altivec instructions.  Examples
 GCC does: "t=vsplti*, r = t+t"  for constants it can't generate with one vsplti

 -0.0 (sign bit):  vspltisw v0,-1 / vslw v0,v0,v0

//===----------------------------------------------------------------------===//

FABS/FNEG can be codegen'd with the appropriate and/xor of -0.0.

//===----------------------------------------------------------------------===//

Codegen the constant here with something better than a constant pool load.

void %test_f(<4 x float>* %P, <4 x float>* %Q, float %X) {
        %tmp = load <4 x float>* %Q
        %tmp = cast <4 x float> %tmp to <4 x int>
        %tmp1 = and <4 x int> %tmp, < int 2147483647, int 2147483647, int 2147483647, int 2147483647 > 
        %tmp2 = cast <4 x int> %tmp1 to <4 x float>
        store <4 x float> %tmp2, <4 x float>* %P
        ret void
}

//===----------------------------------------------------------------------===//

For functions that use altivec AND have calls, we are VRSAVE'ing all call
clobbered regs.

//===----------------------------------------------------------------------===//

Implement passing/returning vectors by value.

//===----------------------------------------------------------------------===//

GCC apparently tries to codegen { C1, C2, Variable, C3 } as a constant pool load
of C1/C2/C3, then a load and vperm of Variable.

//===----------------------------------------------------------------------===//

We currently codegen SCALAR_TO_VECTOR as a store of the scalar to a 16-byte
aligned stack slot, followed by a lve*x/vperm.  We should probably just store it
to a scalar stack slot, then use lvsl/vperm to load it.  If the value is already
in memory, this is a huge win.

//===----------------------------------------------------------------------===//

Do not generate the MFCR/RLWINM sequence for predicate compares when the
predicate compare is used immediately by a branch.  Just branch on the right
cond code on CR6.

//===----------------------------------------------------------------------===//

SROA should turn "vector unions" into the appropriate insert/extract element
instructions.
 
//===----------------------------------------------------------------------===//

We need an LLVM 'shuffle' instruction, that corresponds to the VECTOR_SHUFFLE
node.

//===----------------------------------------------------------------------===//

We need a way to teach tblgen that some operands of an intrinsic are required to
be constants.  The verifier should enforce this constraint.

//===----------------------------------------------------------------------===//

Instead of writting a pattern for type-agnostic operations (e.g. gen-zero, load,
store, and, ...) in every supported type, make legalize do the work.  We should
have a canonical type that we want operations changed to (e.g. v4i32 for
build_vector) and legalize should change non-identical types to thse.  This is
similar to what it does for operations that are only supported in some types,
e.g. x86 cmov (not supported on bytes).

This would fix two problems:
1. Writing patterns multiple times.
2. Identical operations in different types are not getting CSE'd (e.g. 
   { 0U, 0U, 0U, 0U } and {0.0, 0.0, 0.0, 0.0}.

//===----------------------------------------------------------------------===//

Implement multiply for vector integer types, to avoid the horrible scalarized
code produced by legalize.

void test(vector int *X, vector int *Y) {
  *X = *X * *Y;
}

//===----------------------------------------------------------------------===//

There are a wide variety of vector_shuffle operations that we can do with a pair
of instructions (e.g. a vsldoi + vpkuhum).  We should pattern match these, but
there are a huge number of these.

//===----------------------------------------------------------------------===//
