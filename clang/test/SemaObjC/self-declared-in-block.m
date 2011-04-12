// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-darwin10  -fblocks -fobjc-nonfragile-abi -verify %s 
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -triple x86_64-apple-darwin10  -fblocks -fobjc-nonfragile-abi -verify %s 
// rdar://9154582

@interface Blocky @end

@implementation Blocky {
    int _a;
}
- (void)doAThing {
    ^{
        char self; // expected-note {{declared here}}
        _a; // expected-error {{instance variable '_a' cannot be accessed because 'self' has been redeclared}}
    }();
}

@end

