// RUN: %clang -emit-llvm -S -g %s -o - | FileCheck %s

// Radar 8396182
// There are no lexical blocks, but we need two DILexicalBlockFiles to
// correctly represent file info.

int foo() {
  int i = 1;
# 4 "m.c"
# 1 "m.h" 1
  int j = 2;
# 2 "m.h"
# 5 "m.c" 2
  return i + j;
}

// CHECK: DW_TAG_lexical_block
// CHECK: !"m.h"
// CHECK: DW_TAG_lexical_block
// CHECK: !"m.c"
// CHECK-NOT: DW_TAG_lexical_block
