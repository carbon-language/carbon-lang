// RUN: %clang_cc1 -x objective-c -emit-pch -o %t

// Avoid infinite loop because of method redeclarations.

@interface Foo
-(void)meth;
-(void)meth;
-(void)meth;
@end

@implementation Foo
-(void)meth { }
@end
