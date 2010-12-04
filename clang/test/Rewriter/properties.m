// RUN: %clang_cc1 -rewrite-objc %s -o -

// Fariborz approved this being xfail'ed during the addition
// of explicit lvalue-to-rvalue conversions.
// XFAIL: *

@interface Foo {
    int i;
    int rrrr;
    Foo *o;
}
@property int i;
@property(readonly) int rrrr;
@property int d;
@property(retain) Foo *o;

- (void)foo;
@end

@implementation Foo
@synthesize i;
@synthesize rrrr;
@synthesize o;

@dynamic d;

- (void)foo {
    i = 99;
}

- (int)bar {
  return i;
}
@end

@interface Bar {
}
@end

@implementation Bar

static int func(int i);

- (void)baz {
    Foo *obj1, *obj2;
    int i;
    if (obj1.i == obj2.rrrr)
      obj1.i = 33;
    obj1.i = func(obj2.rrrr);
    obj1.i = obj2.rrrr;
    obj1.i = (obj2.rrrr);
    [obj1 setI:[obj2 rrrr]];
    obj1.i = [obj2 rrrr];
    obj1.i = 3 + [obj2 rrrr];
    i = obj1.o.i;
    obj1.o.i = 77;
}
@end
