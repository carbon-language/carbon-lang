// RUN: %clang -emit-llvm -S -g %s -o %t
// RUN: grep DW_TAG_lexical_block %t | count 3

// Radar 8396182
// There are three lexical blocks in this test case.

int foo() {
  int i = 1;
# 4 "m.c"
# 1 "m.h" 1
  int j = 2;
# 2 "m.h"
# 5 "m.c" 2
  return i + j;
}
