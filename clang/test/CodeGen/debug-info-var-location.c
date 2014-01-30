// FIXME: Check IR rather than asm, then triple is not needed.
// RUN: %clang -Xclang -triple=%itanium_abi_triple -S -g -fverbose-asm %s -o - | FileCheck %s
// Radar 8461032
// CHECK: DW_AT_location
// CHECK-NEXT: byte 145

// 145 is DW_OP_fbreg
struct s {
  int a;
  struct s *next;
};

int foo(struct  s *s) {
  switch (s->a) {
  case 1:
  case 2: {
    struct s *sp = s->next;
  }
    break;
  }
  return 1;
}
