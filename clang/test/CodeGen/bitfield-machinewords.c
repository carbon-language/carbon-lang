// RUN: %clang_cc1 -triple i386-unknown-unknown %s -emit-llvm -o - | FileCheck %s -check-prefix=CHECK32
// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -emit-llvm -o - | FileCheck %s -check-prefix=CHECK64

typedef unsigned long long uint64_t;

struct thirty_two_bit_fields {
  unsigned int ttbf1 : 32;
  unsigned int ttbf2 : 32;
  unsigned int ttbf3 : 32;
  unsigned int ttbf4 : 32;
};
void ttbf(struct thirty_two_bit_fields *x) {}
// CHECK32: %struct.thirty_two_bit_fields = type { i32, i32, i32, i32 }
// CHECK64: %struct.thirty_two_bit_fields = type { i64, i64 }

struct thirty_two_in_sixty_four {
  uint64_t ttisf1 : 32;
  uint64_t ttisf2 : 32;
  uint64_t ttisf3 : 32;
  uint64_t ttisf4 : 32;
};
void ttisf(struct thirty_two_in_sixty_four *x) {}
// CHECK32: %struct.thirty_two_in_sixty_four = type { i32, i32, i32, i32 }
// CHECK64: %struct.thirty_two_in_sixty_four = type { i64, i64 }

struct everything_fits {
  unsigned int ef1 : 2;
  unsigned int ef2 : 29;
  unsigned int ef3 : 1;

  unsigned int ef4 : 16;
  unsigned int ef5 : 16;

  unsigned int ef6 : 7;
  unsigned int ef7 : 25;
};
void ef(struct everything_fits *x) {}
// CHECK32: %struct.everything_fits = type { i32, i32, i32 }
// CHECK64: %struct.everything_fits = type <{ i64, i32 }>

struct not_lined_up {
  uint64_t nlu1 : 31;
  uint64_t nlu2 : 2;
  uint64_t nlu3 : 32;
  uint64_t nlu4 : 31;
};
void nlu(struct not_lined_up *x) {}
// CHECK32: %struct.not_lined_up = type { i96 }
// CHECK64: %struct.not_lined_up = type { i40, i64 }

struct padding_between_words {
  unsigned int pbw1 : 16;
  unsigned int pbw2 : 14;

  unsigned int pbw3 : 12;
  unsigned int pbw4 : 16;

  unsigned int pbw5 : 8;
  unsigned int pbw6 : 10;

  unsigned int pbw7 : 20;
  unsigned int pbw8 : 10;
};
void pbw(struct padding_between_words *x) {}
// CHECK32: %struct.padding_between_words = type { i32, i32, i24, i32 }
// CHECK64: %struct.padding_between_words = type { i32, i32, i24, i32 }

struct unaligned_are_coalesced {
  uint64_t uac1 : 16;
  uint64_t uac2 : 32;
  uint64_t uac3 : 16;
  uint64_t uac4 : 48;
  uint64_t uac5 : 64;
  uint64_t uac6 : 16;
  uint64_t uac7 : 32;
};
void uac(struct unaligned_are_coalesced *x) {}
// CHECK32: %struct.unaligned_are_coalesced = type { i112, i112 }
// CHECK64: %struct.unaligned_are_coalesced = type { i64, i48, i64, i48 }
