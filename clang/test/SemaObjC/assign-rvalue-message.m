// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -fobjc-nonfragile-abi %s
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin10 -fsyntax-only -verify -fobjc-nonfragile-abi %s
// rdar://9005189

@interface Foo 
@end

struct Bar {
    int x;
};

@implementation Foo {
    struct Bar bar;
}

- (const struct Bar)bar {
    return bar;
}

- (void)baz {
    bar.x = 0;
    [self bar].x = 10; // expected-error {{assigning to 'readonly' return result of an objective-c message not allowed}}
}
@end
