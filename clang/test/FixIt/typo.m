// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -fixit -o - | %clang_cc1 -fsyntax-only -pedantic -Werror -x objective-c -

@interface NSString
@end

void test() {
  NSstring *str = @"A string"; // expected-error{{use of undeclared identifier 'NSstring'; did you mean 'NSString'?}}
}
