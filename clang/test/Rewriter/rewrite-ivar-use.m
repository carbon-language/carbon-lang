// RUN: %clang_cc1 -rewrite-objc -fms-extensions %s -o -
// radar 7490331

@interface Foo {
        int a;
        id b;
}
- (void)bar;
- (void)baz:(id)q;
@end

@implementation Foo
// radar 7522803
static void foo(id bar) {
        int i = ((Foo *)bar)->a;
}

- (void)bar {
        a = 42;
        [self baz:b];
}
- (void)baz:(id)q {
}
@end

