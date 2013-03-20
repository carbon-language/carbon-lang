// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar: //6854840
@interface A {@end // expected-error {{'@end' appears where closing brace '}' is expected}}


@interface B {
@public
  int ivar;
@protected
  id pp;
@property (copy) id PROP; // expected-error {{illegal visibility specification}}
@end
