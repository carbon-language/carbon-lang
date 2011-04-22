// RUN: %clang_cc1 -fsyntax-only -verify %s
@end // expected-error {{@end must appear in an @implementation context}}
