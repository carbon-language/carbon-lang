// RUN: clang-cc -o %t --emit-llvm -g %s

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
