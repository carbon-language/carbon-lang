// RUN: %clang_cc1 -verify -fsyntax-only %s

// expected-error@+1 {{'swift_attr' attribute takes one argument}}
__attribute__((swift_attr))
@interface I
@end

// expected-error@+1 {{'swift_attr' attribute requires a string}}
__attribute__((swift_attr(1)))
@interface J
@end
