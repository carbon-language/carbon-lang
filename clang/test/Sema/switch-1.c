// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-apple-darwin10 %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -triple x86_64-apple-darwin10 %s
// rdar://11577384

int f(int i) {
  switch (i) {
    case 2147483647 + 2: // expected-warning {{overflow of constant expression results in value -2147483647 of type 'int'}}
      return 1;
    case 9223372036854775807L * 4: // expected-warning {{overflow of constant expression results in value -4 of type 'long'}}
      return 2;
    case (123456 *789012) + 1:  // expected-warning {{overflow of constant expression results in value -1375982336 of type 'int'}}
      return 3;
    case 2147483647:
      return 0;
  }
  return 0;
}
