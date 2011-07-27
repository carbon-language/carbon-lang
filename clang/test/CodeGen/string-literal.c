// RUN: %clang_cc1 -x c++ -std=c++0x -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// Runs in c++0x mode so that wchar_t, char16_t, and char32_t are available.

int main() {
  // CHECK: internal unnamed_addr constant [10 x i8] c"abc\00\00\00\00\00\00\00", align 1
  char a[10] = "abc";

  // This should convert to utf8.
  // CHECK: internal unnamed_addr constant [10 x i8] c"\E1\84\A0\C8\A0\F4\82\80\B0\00", align 1
  char b[10] = "\u1120\u0220\U00102030";

  // CHECK: private unnamed_addr constant [12 x i8] c"A\00\00\00B\00\00\00\00\00\00\00", align 1
  const wchar_t *foo = L"AB";

  // CHECK: private unnamed_addr constant [12 x i8] c"4\12\00\00\0B\F0\10\00\00\00\00\00", align 1
  const wchar_t *bar = L"\u1234\U0010F00B";

  // CHECK: private unnamed_addr constant [12 x i8] c"C\00\00\00D\00\00\00\00\00\00\00", align 1
  const char32_t *c = U"CD";

  // CHECK: private unnamed_addr constant [12 x i8] c"5\12\00\00\0C\F0\10\00\00\00\00\00", align 1
  const char32_t *d = U"\u1235\U0010F00C";

  // CHECK: private unnamed_addr constant [6 x i8] c"E\00F\00\00\00", align 1
  const char16_t *e = u"EF";

  // This should convert to utf16.
  // CHECK: private unnamed_addr constant [10 x i8] c" \11 \02\C8\DB0\DC\00\00", align 1
  const char16_t *f = u"\u1120\u0220\U00102030";

  // CHECK: private unnamed_addr constant [4 x i8] c"def\00", align 1
  const char *g = u8"def";
}
