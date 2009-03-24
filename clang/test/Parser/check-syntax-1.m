// RUN: clang-cc -fsyntax-only -verify %s

int @interface bla  ; // expected-error {{cannot combine with previous 'int' declaration specifier}}
@end
