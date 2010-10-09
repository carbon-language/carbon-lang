// RUN: %clang_cc1 -x c++ -triple i386-unknown-unkown -emit-llvm %s -o - | FileCheck %s
// Runs in c++ mode so that wchar_t is available.

int main() {
  // CHECK: store i8 97
  char a = 'a';

  // Should pick second character.
  // CHECK: store i8 98
  char b = 'ab';

  // CHECK: store i32 97
  wchar_t wa = L'a';

  // Should pick second character.
  // CHECK: store i32 98
  wchar_t wb = L'ab';

  // Should pick last character and store its lowest byte.
  // This does not match gcc, which takes the last character, converts it to
  // utf8, and then picks the second-lowest byte of that (they probably store
  // the utf8 in uint16_ts internally and take the lower byte of that).
  // CHECK: store i8 48
  char c = '\u1120\u0220\U00102030';

  // CHECK: store i32 61451
  wchar_t wc = L'\uF00B';

  // CHECK: store i32 1110027
  wchar_t wd = L'\U0010F00B';

  // Should pick second character.
  // CHECK: store i32 1110027
  wchar_t we = L'\u1234\U0010F00B';
}
