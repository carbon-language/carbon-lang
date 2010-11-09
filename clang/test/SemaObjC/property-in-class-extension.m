// RUN: %clang_cc1  -fsyntax-only -verify %s
// rdar://7766184

@interface Foo @end

@interface Foo ()
  @property (readonly) int bar;
@end

void FUNC () {
    Foo *foo;
    foo.bar = 0; // expected-error {{assigning to property with 'readonly' attribute not allowed}}
}


