// RUN: %clang_cc1 %s -emit-llvm -o - -triple=armv7-unknown-freebsd -std=c11 | FileCheck %s

// Test that we are generating atomicrmw instructions, rather than
// compare-exchange loops for common atomic ops.  This makes a big difference
// on RISC platforms, where the compare-exchange loop becomes a ll/sc pair for
// the load and then another ll/sc in the loop, expanding to about 30
// instructions when it should be only 4.  It has a smaller, but still
// noticeable, impact on platforms like x86 and RISC-V, where there are atomic
// RMW instructions.
//
// We currently emit cmpxchg loops for most operations on _Bools, because
// they're sufficiently rare that it's not worth making sure that the semantics
// are correct.

typedef int __attribute__((vector_size(16))) vector;

_Atomic(_Bool) b;
_Atomic(int) i;
_Atomic(long long) l;
_Atomic(short) s;
_Atomic(char*) p;
_Atomic(float) f;
_Atomic(vector) v;

// CHECK-NOT: cmpxchg

// CHECK: testinc
void testinc(void)
{
  // Special case for suffix bool++, sets to true and returns the old value.
  // CHECK: atomicrmw xchg i8* @b, i8 1 seq_cst
  b++;
  // CHECK: atomicrmw add i32* @i, i32 1 seq_cst
  i++;
  // CHECK: atomicrmw add i64* @l, i64 1 seq_cst
  l++;
  // CHECK: atomicrmw add i16* @s, i16 1 seq_cst
  s++;
  // Prefix increment
  // Special case for bool: set to true and return true
  // CHECK: store atomic i8 1, i8* @b seq_cst, align 1
  ++b;
  // Currently, we have no variant of atomicrmw that returns the new value, so
  // we have to generate an atomic add, which returns the old value, and then a
  // non-atomic add.
  // CHECK: atomicrmw add i32* @i, i32 1 seq_cst
  // CHECK: add i32 
  ++i;
  // CHECK: atomicrmw add i64* @l, i64 1 seq_cst
  // CHECK: add i64
  ++l;
  // CHECK: atomicrmw add i16* @s, i16 1 seq_cst
  // CHECK: add i16
  ++s;
}
// CHECK: testdec
void testdec(void)
{
  // CHECK: cmpxchg i8* @b
  b--;
  // CHECK: atomicrmw sub i32* @i, i32 1 seq_cst
  i--;
  // CHECK: atomicrmw sub i64* @l, i64 1 seq_cst
  l--;
  // CHECK: atomicrmw sub i16* @s, i16 1 seq_cst
  s--;
  // CHECK: cmpxchg i8* @b
  --b;
  // CHECK: atomicrmw sub i32* @i, i32 1 seq_cst
  // CHECK: sub i32
  --i;
  // CHECK: atomicrmw sub i64* @l, i64 1 seq_cst
  // CHECK: sub i64
  --l;
  // CHECK: atomicrmw sub i16* @s, i16 1 seq_cst
  // CHECK: sub i16
  --s;
}
// CHECK: testaddeq
void testaddeq(void)
{
  // CHECK: cmpxchg i8* @b
  // CHECK: atomicrmw add i32* @i, i32 42 seq_cst
  // CHECK: atomicrmw add i64* @l, i64 42 seq_cst
  // CHECK: atomicrmw add i16* @s, i16 42 seq_cst
  b += 42;
  i += 42;
  l += 42;
  s += 42;
}
// CHECK: testsubeq
void testsubeq(void)
{
  // CHECK: cmpxchg i8* @b
  // CHECK: atomicrmw sub i32* @i, i32 42 seq_cst
  // CHECK: atomicrmw sub i64* @l, i64 42 seq_cst
  // CHECK: atomicrmw sub i16* @s, i16 42 seq_cst
  b -= 42;
  i -= 42;
  l -= 42;
  s -= 42;
}
// CHECK: testxoreq
void testxoreq(void)
{
  // CHECK: cmpxchg i8* @b
  // CHECK: atomicrmw xor i32* @i, i32 42 seq_cst
  // CHECK: atomicrmw xor i64* @l, i64 42 seq_cst
  // CHECK: atomicrmw xor i16* @s, i16 42 seq_cst
  b ^= 42;
  i ^= 42;
  l ^= 42;
  s ^= 42;
}
// CHECK: testoreq
void testoreq(void)
{
  // CHECK: cmpxchg i8* @b
  // CHECK: atomicrmw or i32* @i, i32 42 seq_cst
  // CHECK: atomicrmw or i64* @l, i64 42 seq_cst
  // CHECK: atomicrmw or i16* @s, i16 42 seq_cst
  b |= 42;
  i |= 42;
  l |= 42;
  s |= 42;
}
// CHECK: testandeq
void testandeq(void)
{
  // CHECK: cmpxchg i8* @b
  // CHECK: atomicrmw and i32* @i, i32 42 seq_cst
  // CHECK: atomicrmw and i64* @l, i64 42 seq_cst
  // CHECK: atomicrmw and i16* @s, i16 42 seq_cst
  b &= 42;
  i &= 42;
  l &= 42;
  s &= 42;
}

