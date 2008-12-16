// RUN: clang -fsyntax-only -verify %s

typedef int T[2];
typedef void (F)(void);

@interface A
@property(assign) T p2;  // expected-error {{property cannot have type 'T' (array or function type)}}

@property(assign) F f2; // expected-error {{property cannot have type 'F' (array or function type)}}

@end

