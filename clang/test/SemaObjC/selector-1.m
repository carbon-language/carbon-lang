// RUN: %clang_cc1  -fsyntax-only -Wselector -verify %s
// rdar://8851684

@interface Foo
- (void) foo;
- (void) bar;
@end

@implementation Foo
- (void) bar
{
}

- (void) foo
{
  SEL a,b,c;
  a = @selector(b1ar);  // expected-warning {{unimplemented selector 'b1ar'}}
  b = @selector(bar);
}
@end

@interface I
- length;
@end

SEL func()
{
    return  @selector(length);  // expected-warning {{unimplemented selector 'length'}}
}
