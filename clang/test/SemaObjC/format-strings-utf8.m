// REQUIRES: shell
// RUN: env -i LC_ALL=C \
// RUN: %clang_cc1 -triple x86_64-apple-darwin -Wformat-nonliteral -fsyntax-only -verify -Wno-objc-root-class %s

#include <stdarg.h>
int printf(const char *restrict, ...);
int scanf(const char * restrict, ...);
@class NSString, Protocol;
extern void NSLog(NSString *format, ...);

void testInvalidNoPrintable(int *a) {
  printf("%\u25B9"); // expected-warning {{invalid conversion specifier '\u25b9'}}
  printf("%\xE2\x96\xB9"); // expected-warning {{invalid conversion specifier '\u25b9'}}
  printf("%\U00010348"); // expected-warning {{invalid conversion specifier '\U00010348'}}
  printf("%\xF0\x90\x8D\x88"); // expected-warning {{invalid conversion specifier '\U00010348'}}
  printf("%\xe2"); // expected-warning {{invalid conversion specifier '\xe2'}}
  NSLog(@"%\u25B9"); // expected-warning {{invalid conversion specifier '\u25b9'}}
  NSLog(@"%\xE2\x96\xB9"); // expected-warning {{invalid conversion specifier '\u25b9'}}
  NSLog(@"%\U00010348"); // expected-warning {{invalid conversion specifier '\U00010348'}}
  NSLog(@"%\xF0\x90\x8D\x88"); // expected-warning {{invalid conversion specifier '\U00010348'}}
  NSLog(@"%\xe2"); // expected-warning {{input conversion stopped}} expected-warning {{invalid conversion specifier '\xe2'}}
  scanf("%\u25B9", a); // expected-warning {{invalid conversion specifier '\u25b9'}}
  scanf("%\xE2\x96\xB9", a); // expected-warning {{invalid conversion specifier '\u25b9'}}
  scanf("%\U00010348", a); // expected-warning {{invalid conversion specifier '\U00010348'}}
  scanf("%\xF0\x90\x8D\x88", a); // expected-warning {{invalid conversion specifier '\U00010348'}}
  scanf("%\xe2", a); // expected-warning {{invalid conversion specifier '\xe2'}}
}
