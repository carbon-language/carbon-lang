// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify %s
// rdar://10565506

@protocol P @end

@interface I
@property Class<P> MyClass;
@property Class MyClass1;
@property void * VOIDSTAR;
@end

@implementation I
@synthesize MyClass, MyClass1, VOIDSTAR;
@end
