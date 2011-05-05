// RUN: %clang_cc1 -triple x86_64-unk-unk -o %t -emit-llvm -g %s
// RUN: FileCheck --input-file=%t %s

// PR3023
void convert(void) {
  struct { typeof(0) f0; } v0;
}


// PR2784
struct OPAQUE;
typedef struct OPAQUE *PTR;
PTR p;


// PR2950
struct s0;
struct s0 { struct s0 *p; } g0;

struct s0 *f0(struct s0 *a0) {
  return a0->p;
}


// PR3134
char xpto[];


// PR3427
struct foo {
  int a;
  void *ptrs[];
};
struct foo bar;


// PR4143
struct foo2 {
  enum bar *bar;
};

struct foo2 foo2;


// Radar 7325611
// CHECK: "barfoo"
typedef int barfoo;
barfoo foo() {
}

// CHECK: __uint128_t
__uint128_t foo128 ()
{
  __uint128_t int128 = 44;
  return int128;
}
