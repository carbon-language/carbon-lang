// RUN: clang-cc -analyze -checker-cfref -analyzer-constraints=basic -analyzer-store=basic %s -verify &&
// RUN: clang-cc -analyze -checker-cfref -analyzer-constraints=basic -analyzer-store=region %s -verify

typedef struct Foo { int x; } Bar;

@interface MyClass {}
- (Bar)foo;
@end
@implementation MyClass
- (Bar)foo { 
  struct Foo f = { 0 };
  return f;
}
@end

void createFoo() {
  MyClass *obj = 0;  
  Bar f = [obj foo]; // expected-warning{{The receiver in the message expression is 'nil' and results in the returned value (of type 'Bar') to be garbage or otherwise undefined.}}
}

void createFoo2() {
  MyClass *obj = 0;  
  [obj foo]; // no-warning
  Bar f = [obj foo]; // expected-warning{{The receiver in the message expression is 'nil' and results in the returned value (of type 'Bar') to be garbage or otherwise undefined.}}
}

