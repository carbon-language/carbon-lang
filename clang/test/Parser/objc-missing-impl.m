// RUN: clang -cc1 -fsyntax-only -verify %s
@end // expected-warning {{@end must appear in an @implementation context}}
