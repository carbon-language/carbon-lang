// RUN: clang-cc -fsyntax-only -verify %s
// XFAIL
// <rdar://problem/6211479>

typedef int T[2];

@interface A
@property(assign) T p2; // expected-error {{FIXME: property has invalid type}}
@end
