// RUN: %clang_cc1 -x c++ -emit-llvm -fshort-wchar %s -o - | FileCheck %s
// Runs in c++ mode so that wchar_t is available.

int main() {
  // This should convert to utf8.
  // CHECK: internal unnamed_addr constant [10 x i8] c"\E1\84\A0\C8\A0\F4\82\80\B0\00", align 1
  char b[10] = "\u1120\u0220\U00102030";

  // CHECK: private unnamed_addr constant [6 x i8] c"A\00B\00\00\00"
  const wchar_t *foo = L"AB";

  // This should convert to utf16.
  // CHECK: private unnamed_addr constant [10 x i8] c" \11 \02\C8\DB0\DC\00\00"
  const wchar_t *bar = L"\u1120\u0220\U00102030";



  // Should pick second character.
  // CHECK: store i8 98
  char c = 'ab';

  // CHECK: store i16 97
  wchar_t wa = L'a';

  // Should pick second character.
  // CHECK: store i16 98
  wchar_t wb = L'ab';

  // -4085 == 0xf00b
  // CHECK: store i16 -4085
  wchar_t wc = L'\uF00B';

  // Should take lower word of the 4byte UNC sequence. This does not match
  // gcc. I don't understand what gcc does (it looks like it converts to utf16,
  // then takes the second (!) utf16 word, swaps the lower two nibbles, and
  // stores that?).
  // CHECK: store i16 -4085
  wchar_t wd = L'\U0010F00B';  // has utf16 encoding dbc8 dcb0

  // Should pick second character. (gcc: -9205)
  // CHECK: store i16 -4085
  wchar_t we = L'\u1234\U0010F00B';
}
