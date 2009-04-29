// RUN: clang -fsyntax-only -Xclang -verify %s
typedef union {
  int *ip;
  float *fp;
} TU __attribute__((transparent_union));

void f(TU);

void g(int *ip, float *fp, char *cp) {
  f(ip);
  f(fp);
  f(cp); // expected-error{{incompatible type}}
  f(0);

  TU tu_ip = ip; // expected-error{{incompatible type}}
  TU tu;
  tu.ip = ip;
}

/* FIXME: we'd like to just use an "int" here and align it differently
   from the normal "int", but if we do so we lose the alignment
   information from the typedef within the compiler. */
typedef struct { int x, y; } __attribute__((aligned(8))) aligned_struct8;

typedef struct { int x, y; } __attribute__((aligned(4))) aligned_struct4;
typedef union {
  aligned_struct4 s4; // expected-note{{alignment of first field}}
  aligned_struct8 s8; // expected-warning{{alignment of field}}
} TU1 __attribute__((transparent_union));

typedef union {
  char c; // expected-note{{size of first field is 8 bits}}
  int i; // expected-warning{{size of field}}
} TU2 __attribute__((transparent_union));

typedef union {
  float f; // expected-warning{{floating}}
} TU3 __attribute__((transparent_union));

typedef union { } TU4 __attribute__((transparent_union)); // expected-warning{{field}} 
