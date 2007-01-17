// RUN: %llvmgcc -S %s -o /dev/null

// XFAIL: llvmgcc3

struct istruct {
  unsigned char C;
};

struct foo {
  unsigned int I:1;
  struct istruct J;
  unsigned char L[1];
  unsigned int K:1;
};

struct foo F = { 1, { 7 }, { 123 } , 1 };


