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


// rdar://9284603
@interface ShadowSelf
{
    int _anIvar;
}
@end

@interface C {
  int _cIvar;
}
@end

@implementation ShadowSelf 
- (void)doSomething {
    __typeof(self) newSelf = self;
    {
        __typeof(self) self = newSelf;
        (void)_anIvar;
    }
    {
      C* self;	// expected-note {{declared here}}
      (void) _anIvar; // expected-error {{instance variable '_anIvar' cannot be accessed because 'self' has been redeclared}}
    }
}
- (void)doAThing {
    ^{
        id self;	// expected-note {{declared here}}
	(void)_anIvar; // expected-error {{instance variable '_anIvar' cannot be accessed because 'self' has been redeclared}}
    }();
}
@end

