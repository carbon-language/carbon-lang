// RUN: clang -cc1  -fsyntax-only -verify %s

@interface Foo
- (char*) foo;
- (void) bar;
@end

@implementation Foo
- (void) bar
{
}

- (char*) foo
{
  char* a,b,c;
  a = (char*)@selector(bar);  // expected-error {{cannot type cast @selector expression}}
  return (char*)@selector(bar);  // expected-error {{cannot type cast @selector expression}}
}
@end

