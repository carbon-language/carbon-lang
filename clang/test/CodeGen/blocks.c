// RUN: %clang_cc1 -triple i386-unknown-unknown %s -emit-llvm -o %t -fblocks
void (^f)(void) = ^{};

// rdar://6768379
int f0(int (^a0)()) {
  return a0(1, 2, 3);
}

// Verify that attributes on blocks are set correctly.
typedef struct s0 T;
struct s0 {
  int a[64];
};

// RUN: grep 'internal void @__f2_block_invoke_0(.struct.s0\* sret .*, .*, .* byval .*)' %t
struct s0 f2(struct s0 a0) {
  return ^(struct s0 a1){ return a1; }(a0);
}

// This should not crash: rdar://6808051
void *P = ^{
  void *Q = __func__;
};

void (^test1)(void) = ^(void) {
  __block int i;
  ^ { i = 1; }();
};

typedef double ftype(double);
// It's not clear that we *should* support this syntax, but until that decision
// is made, we should support it properly and not crash.
ftype ^test2 = ^ftype {
  return 0;
};

// rdar://problem/8605032
void f3_helper(void (^)(void));
void f3() {
  _Bool b = 0;
  f3_helper(^{ if (b) {} });
}
