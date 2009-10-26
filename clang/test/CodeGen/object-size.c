// RUN: clang-cc -triple x86_64-apple-darwin -S -D_FORTIFY_SOURCE=2 %s -o %t.s &&
// RUN: FileCheck --input-file=%t.s %s
#include <string.h>

char gbuf[63];
char *gp;

void test1() {
  // CHECK:       movabsq $59, %rdx
  // CHECK-NEXT:  movq    %rax, %rdi
  // CHECK-NEXT:  movq    %rcx, %rsi
  // CHECK-NEXT:  call    ___strcpy_chk
  strcpy(&gbuf[4], "Hi there");
}

void test2() {
  // CHECK:       movabsq $63, %rdx
  // CHECK-NEXT:  movq    %rax, %rdi
  // CHECK-NEXT:  movq    %rcx, %rsi
  // CHECK-NEXT:  call    ___strcpy_chk
  strcpy(gbuf, "Hi there");
}

void test4() {
  // CHECK:       call    ___inline_strcpy_chk
  strcpy(gp, "Hi");
}

void test3() {
  int i;
  // CHECK:       call    ___inline_strcpy_chk
  strcpy((++i, gbuf), "Hi");
}
