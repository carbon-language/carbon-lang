//===- README_ALTIVEC.txt - Notes for improving Altivec code gen ----------===//

Implement TargetConstantVec, and set up PPC to custom lower ConstantVec into
TargetConstantVec's if it's one of the many forms that are algorithmically
computable using the spiffy altivec instructions.

//===----------------------------------------------------------------------===//

Implement PPCInstrInfo::isLoadFromStackSlot/isStoreToStackSlot for vector
registers, to generate better spill code.

//===----------------------------------------------------------------------===//

Altivec support.  The first should be a single lvx from the constant pool, the
second should be a xor/stvx:

void foo(void) {
  int x[8] __attribute__((aligned(128))) = { 1, 1, 1, 1, 1, 1, 1, 1 };
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

We need to codegen -0.0 vector efficiently (no constant pool load).

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
altivec instructions.  For example, GCC does: t=vsplti*, r = t+t.

//===----------------------------------------------------------------------===//

