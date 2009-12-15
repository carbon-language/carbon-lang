// RUN: %clang_cc1 -rewrite-objc %s -o -

@interface MyDerived
- (void) instanceMethod;
@end

@implementation MyDerived
- (void) instanceMethod {
}
@end

