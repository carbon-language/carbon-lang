// RUN: %clang_cc1 -x c++ -std=c++0x -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// Runs in c++0x mode so that wchar_t, char16_t, and char32_t are available.

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

  // CHECK: store i16 97
  char16_t ua = u'a';

  // Should pick second character.
  // CHECK: store i16 98
  char16_t ub = u'ab';

  // CHECK: store i32 97
  char32_t Ua = U'a';

  // Should pick second character.
  // CHECK: store i32 98
  char32_t Ub = U'ab';

  // Should pick last character and store its lowest byte.
  // This does not match gcc, which takes the last character, converts it to
  // utf8, and then picks the second-lowest byte of that (they probably store
  // the utf8 in uint16_ts internally and take the lower byte of that).
  // CHECK: store i8 48
  char c = '\u1120\u0220\U00102030';

  // CHECK: store i32 61451
  wchar_t wc = L'\uF00B';

  // -4085 == 0xf00b
  // CHECK: store i16 -4085
  char16_t uc = u'\uF00B';

  // CHECK: store i32 61451
  char32_t Uc = U'\uF00B';

  // CHECK: store i32 1110027
  wchar_t wd = L'\U0010F00B';

  // Should take lower word of the 4byte UNC sequence. This does not match
  // gcc. I don't understand what gcc does (it looks like it converts to utf16,
  // then takes the second (!) utf16 word, swaps the lower two nibbles, and
  // stores that?).
  // CHECK: store i16 -4085
  char16_t ud = u'\U0010F00B';  // has utf16 encoding dbc8 dcb0

  // CHECK: store i32 1110027
  char32_t Ud = U'\U0010F00B';

  // Should pick second character.
  // CHECK: store i32 1110027
  wchar_t we = L'\u1234\U0010F00B';

  // Should pick second character.
  // CHECK: store i16 -4085
  char16_t ue = u'\u1234\U0010F00B';

  // Should pick second character.
  // CHECK: store i32 1110027
  char32_t Ue = U'\u1234\U0010F00B';

}
