// RUN: %clang_cc1 -fsyntax-only -verify %s -triple x86_64-linux-gnu
// expected-no-diagnostics

// PR15216
// Don't crash when taking computing the offset of structs with large arrays.
const unsigned long Size = (1l << 62);

struct Chunk {
  char padding[Size];
  char more_padding[1][Size];
  char data;
};

int test1 = __builtin_offsetof(struct Chunk, data);

