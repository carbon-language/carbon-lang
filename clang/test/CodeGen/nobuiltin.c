// RUN: %clang_cc1 -fno-builtin -O1 -S -o - %s | FileCheck %s

void fn() {
  char content[2];
  // CHECK: __strcpy_chk
  __builtin___strcpy_chk(content, "", 1);
}
