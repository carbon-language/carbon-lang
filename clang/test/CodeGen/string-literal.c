// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s

int main() {
  // CHECK: internal constant [10 x i8] c"abc\00\00\00\00\00\00\00", align 1
  char a[10] = "abc";

  // This should convert to utf8.
  // CHECK: internal constant [10 x i8] c"\E1\84\A0\C8\A0\F4\82\80\B0\00", align 1
  char b[10] = "\u1120\u0220\U00102030";

  // CHECK: private unnamed_addr constant [12 x i8] c"A\00\00\00B\00\00\00\00\00\00\00", align 1
  void *foo = L"AB";

  // CHECK: private unnamed_addr constant [12 x i8] c"4\12\00\00\0B\F0\10\00\00\00\00\00", align 1
  void *bar = L"\u1234\U0010F00B";
}
