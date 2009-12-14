// RUN: clang -cc1 -fsyntax-only -verify %s

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
  [super iMethod]; // expected-warning{{method '-iMethod' not found (return type defaults to 'id')}}
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
void f1(int puper) {
  [super m]; // expected-error{{use of undeclared identifier 'super'}}
}
