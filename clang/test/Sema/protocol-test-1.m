// RUN: clang -fsyntax-only -verify %s

@protocol PROTO1
@required 
- (int) FooBar;
@optional
- (void) MyMethod1;
+ (int) S;
@end

@interface INTF1
@required  // expected-error {{@required may be specified in protocols only}}
- (int) FooBar;
- (int) FooBar1;
- (int) FooBar2;
@optional  // expected-error {{@optional may be specified in protocols only}}
+ (int) C;

- (int)I;
@end
