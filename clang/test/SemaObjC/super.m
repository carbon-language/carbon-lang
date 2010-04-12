// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface Foo
- iMethod;
+ cMethod;
@end

@interface A
@end

@interface B : A
- (void)instanceMethod;
+ classMethod;
@end

@implementation B

- (void)instanceMethod {
  [super iMethod]; // expected-warning{{'A' may not respond to 'iMethod')}}
}

+ classMethod {
  [super cMethod]; // expected-warning{{method '+cMethod' not found (return type defaults to 'id')}}
  return 0;
}
@end

@interface XX
- m;
@end

void f(id super) {
  [super m];
}
void f0(int super) {
  [super m]; // expected-warning{{receiver type 'int' is not 'id'}} \
                expected-warning {{method '-m' not found (return type defaults to 'id')}}
}
void f1(id puper) {  // expected-note {{'puper' declared here}}
  [super m]; // expected-error{{use of undeclared identifier 'super'; did you mean 'puper'?}}
}

// radar 7400691
typedef Foo super;

typedef Foo FooTD;

void test() {
  [FooTD cMethod];
  [super cMethod];
}

struct SomeStruct {
  int X;
};

int test2() {
  struct SomeStruct super = { 0 };
  return super.X;
}

int test3() {
  id super = 0;
  [(B*)super instanceMethod];
  int *s1 = (int*)super;
  return 0;
}
