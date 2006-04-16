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
altivec instructions.

Examples, these work with all widths:
  Splat(+/- 16,18,20,22,24,28,30):  t = vspliti I/2,  r = t+t
  Splat(+/- 17,19,21,23,25,29):     t = vsplti +/-15, t2 = vsplti I-15, r=t + t2
  Splat(31):                        t = vsplti FB,  r = srl t,t
  Splat(256):  t = vsplti 1, r = vsldoi t, t, 1

Lots more are listed here:
http://www.informatik.uni-bremen.de/~hobold/AltiVec.html

This should be added to the ISD::BUILD_VECTOR case in 
PPCTargetLowering::LowerOperation.

//===----------------------------------------------------------------------===//

FABS/FNEG can be codegen'd with the appropriate and/xor of -0.0.

//===----------------------------------------------------------------------===//

For functions that use altivec AND have calls, we are VRSAVE'ing all call
clobbered regs.

//===----------------------------------------------------------------------===//

Implement passing vectors by value.

//===----------------------------------------------------------------------===//

GCC apparently tries to codegen { C1, C2, Variable, C3 } as a constant pool load
of C1/C2/C3, then a load and vperm of Variable.

//===----------------------------------------------------------------------===//

We currently codegen SCALAR_TO_VECTOR as a store of the scalar to a 16-byte
aligned stack slot, followed by a load/vperm.  We should probably just store it
to a scalar stack slot, then use lvsl/vperm to load it.  If the value is already
in memory, this is a huge win.

//===----------------------------------------------------------------------===//

Do not generate the MFCR/RLWINM sequence for predicate compares when the
predicate compare is used immediately by a branch.  Just branch on the right
cond code on CR6.

//===----------------------------------------------------------------------===//

We need a way to teach tblgen that some operands of an intrinsic are required to
be constants.  The verifier should enforce this constraint.

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

Specific examples:

C = vector_shuffle A, B, <0, 1, 2, 4>
->  t = vsldoi A, A, 12
->  C = vsldoi A, B, 4

//===----------------------------------------------------------------------===//

extract_vector_elt of an arbitrary constant vector can be done with the 
following instructions:

vTemp = vec_splat(v0,2);    // 2 is the element the src is in.
vec_ste(&destloc,0,vTemp);

We can do an arbitrary non-constant value by using lvsr/perm/ste.

//===----------------------------------------------------------------------===//
