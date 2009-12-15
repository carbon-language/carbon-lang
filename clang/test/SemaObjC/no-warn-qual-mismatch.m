// RUN: %clang_cc1  -fsyntax-only -verify %s
// radar 7211563

@interface X

+ (void)prototypeWithScalar:(int)aParameter;
+ (void)prototypeWithPointer:(void *)aParameter;

@end

@implementation X

+ (void)prototypeWithScalar:(const int)aParameter {}
+ (void)prototypeWithPointer:(void * const)aParameter {}

@end
