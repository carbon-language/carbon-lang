// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

@interface X {}
+ (X*) alloc;
- (X*) init;
- (int) getSize;
- (void) setSize: (int) size;
- (X*) getSelf;
@end

void f(X *noreturn) {
  // An array size which is computed by a message send is OK.
  int a[ [noreturn getSize] ];

  // ... but is interpreted as an attribute where possible.
  int b[ [noreturn] ]; // expected-warning {{'noreturn' only applies to function types}}

  int c[ [noreturn getSize] + 1 ];

  // An array size which is computed by a lambda is not OK.
  int d[ [noreturn] { return 3; } () ]; // expected-error {{expected ']'}} expected-warning {{'noreturn' only applies}}

  // A message send which contains a message send is OK.
  [ [ X alloc ] init ];
  [ [ int(), noreturn getSelf ] getSize ]; // expected-warning {{unused}}

  // A message send which contains a lambda is OK.
  [ [noreturn] { return noreturn; } () setSize: 4 ];
  [ [bitand] { return noreturn; } () setSize: 5 ];
  [[[[] { return [ X alloc ]; } () init] getSelf] getSize];

  // An attribute is OK.
  [[]];
  [[int(), noreturn]];
  [[class, test(foo 'x' bar),,,]];
  [[bitand, noreturn]];

  [[noreturn]]int(e)();

  // A function taking a noreturn function.
  int(f)([[noreturn]] int());
  f(e);

  // Variables initialized by a message send.
  int(g)([[noreturn getSelf] getSize]);
  int(h)([[noreturn]{return noreturn;}() getSize]);

  int i = g + h;
}

template<typename...Ts> void f(Ts ...x) {
  [[test::foo(bar, baz)...]];
  [[used(x)...]];
  [[x...] { return [ X alloc ]; }() init];
}
