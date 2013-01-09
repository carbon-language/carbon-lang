// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-apple-darwin10 %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -triple x86_64-apple-darwin10 %s
// rdar://11577384

int f(int i) {
  switch (i) {
    case 2147483647 + 2: // expected-note {{value 2147483649 is outside the range of representable values of type 'int'}}  \
                      // expected-warning {{overflow converting case value to switch condition type}} 
      return 1;
    case 9223372036854775807L * 4 : // expected-note {{value 36893488147419103228 is outside the range of representable values of type 'long'}}   \
                        // expected-warning {{overflow converting case value to switch condition type}} 
      return 2;
    case 2147483647:
      return 0;
  }
  return 0;
}
