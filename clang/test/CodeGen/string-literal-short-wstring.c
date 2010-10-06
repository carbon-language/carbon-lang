// RUN: %clang_cc1 -emit-llvm -fshort-wchar %s -o - | FileCheck %s

int main() {
  // This should convert to utf8.
  // CHECK: internal constant [10 x i8] c"\E1\84\A0\C8\A0\F4\82\80\B0\00", align 1
  char b[10] = "\u1120\u0220\U00102030";

  // CHECK: private constant [6 x i8] c"A\00B\00\00\00"
  void *foo = L"AB";

  // This should convert to utf16.
  // CHECK: private constant [10 x i8] c" \11 \02\C8\DB0\DC\00\00"
  void *bar = L"\u1120\u0220\U00102030";
}
