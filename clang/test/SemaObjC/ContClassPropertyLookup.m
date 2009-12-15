// RUN: %clang_cc1  -fsyntax-only -verify %s

@interface MyObject {
    int _foo;
}
@end

@interface MyObject(whatever)
@property (assign) int foo;
@end

@interface MyObject()
@property (assign) int foo;
@end

@implementation MyObject
@synthesize foo = _foo;
@end
