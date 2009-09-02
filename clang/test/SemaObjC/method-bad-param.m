// RUN: clang-cc -triple i386-unknown-unknown -fsyntax-only -verify %s

@interface foo
@end

@implementation foo
@end

@interface bar
-(void) my_method:(foo) my_param; // expected-error {{Objective-C interface type 'foo' cannot be passed by value}}
- (foo)cccccc:(long)ddddd;  // expected-error {{Objective-C interface type 'foo' cannot be returned by value}}
@end

@implementation bar
-(void) my_method:(foo) my_param  // expected-error {{Objective-C interface type 'foo' cannot be passed by value}}
{
}
- (foo)cccccc:(long)ddddd // expected-error {{Objective-C interface type 'foo' cannot be returned by value}}
{
}
@end

void somefunc(foo x) {} // expected-error {{Objective-C interface type 'foo' cannot be passed by value}}
foo somefunc2() {} // expected-error {{Objective-C interface type 'foo' cannot be returned by value}}

// rdar://6780761
void f0(foo *a0) {
  extern void g0(int x, ...);
  g0(1, *(foo*)0);  // expected-error {{cannot pass object with interface type 'foo' by-value through variadic function}}
}
