// RUN: clang -fsyntax-only -verify %s

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
}
@end
