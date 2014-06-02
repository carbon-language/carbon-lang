// RUN: %clang_cc1 -g -triple %itanium_abi_triple -emit-llvm -o - %s | FileCheck %s

// Make sure that clang outputs distinct debug info for a function
// that is inlined twice on the same line. Otherwise it would appear
// as if the function was only inlined once.

#define INLINE inline __attribute__((always_inline))

int i;

INLINE void sum(int a, int b) {
  i = a + b;
}

void noinline(int x, int y) {
  i = x + y;
}

#define CALLS sum(9, 10), sum(11, 12)

inline void inlsum(int t, int u) {
  i = t + u;
}

int main() {
  sum(1, 2), sum(3, 4);
  noinline(5, 6), noinline(7, 8);
  CALLS;
  inlsum(13, 14), inlsum(15, 16);
}

// CHECK-LABEL: @main
// CHECK: = add {{.*}} !dbg [[FIRST_INLINE:![0-9]*]]
// CHECK: = add {{.*}} !dbg [[SECOND_INLINE:![0-9]*]]

// Check that we don't give column information (and thus end up with distinct
// line entries) for two non-inlined calls on the same line.
// CHECK: call {{.*}}noinline{{.*}}({{i32[ ]?[a-z]*}} 5, {{i32[ ]?[a-z]*}} 6), !dbg [[NOINLINE:![0-9]*]]
// CHECK: call {{.*}}noinline{{.*}}({{i32[ ]?[a-z]*}} 7, {{i32[ ]?[a-z]*}} 8), !dbg [[NOINLINE]]

// FIXME: These should be separate locations but because the two calls have the
// same line /and/ column, they get coalesced into a single inlined call by
// accident. We need discriminators or some other changes to LLVM to cope with
// this. (this is, unfortunately, an LLVM test disguised as a Clang test - since
// inlining is forced to happen here). It's possible this could be fixed in
// Clang, but I doubt it'll be the right place for the fix.
// CHECK: = add {{.*}} !dbg [[FIRST_MACRO_INLINE:![0-9]*]]
// CHECK: = add {{.*}} !dbg [[FIRST_MACRO_INLINE]]

// Even if the functions are marked inline but do not get inlined, they
// shouldn't use column information, and thus should be at the same debug
// location.
// CHECK: call {{.*}}inlsum{{.*}}({{i32[ ]?[a-z]*}} 13, {{i32[ ]?[a-z]*}} 14), !dbg [[INL_FIRST:![0-9]*]]
// CHECK: call {{.*}}inlsum{{.*}}({{i32[ ]?[a-z]*}} 15, {{i32[ ]?[a-z]*}} 16), !dbg [[INL_SECOND:![0-9]*]]

// [[FIRST_INLINE]] =
// [[SECOND_INLINE]] =

// FIXME: These should be the same location since the functions appear on the
// same line and were not inlined - they needlessly have column information
// intended to disambiguate inlined calls, which is going to confuse GDB as it
// doesn't cope well with column information.
// [[INL_FIRST]] =
// [[INL_SECOND]] =
