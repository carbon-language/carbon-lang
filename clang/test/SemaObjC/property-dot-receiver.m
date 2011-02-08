// RUN: %clang_cc1 -fsyntax-only -verify %s 
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify %s 
// rdar://8962253

@interface Singleton {
}
+ (Singleton*) instance;
@end

@implementation Singleton

- (void) someSelector { }

+ (Singleton*) instance { return 0; }

+ (void) compileError
{
     [Singleton.instance  someSelector]; // clang issues error here
}

@end

