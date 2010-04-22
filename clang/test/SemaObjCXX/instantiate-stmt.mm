// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface NSException
@end

// @throw
template<typename T>
void throw_test(T value) {
  @throw value; // expected-error{{@throw requires an Objective-C object type ('int' invalid)}}
}

template void throw_test(NSException *);
template void throw_test(int); // expected-note{{in instantiation of}}


