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

// CHECK-NOT: !MDLexicalBlock
// CHECK: !MDLexicalBlockFile({{.*}}file: ![[MH:[0-9]+]]
// CHECK: !MDFile(filename: "m.h"
// CHECK: !MDLexicalBlockFile({{.*}}file: ![[MC:[0-9]+]]
// CHECK: !MDFile(filename: "m.c"
// CHECK-NOT: !MDLexicalBlock
