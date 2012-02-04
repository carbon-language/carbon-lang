// RUN: %clang_cc1 -rewrite-objc -fobjc-fragile-abi  %s -o -

@interface MyDerived
- (void) instanceMethod;
@end

@implementation MyDerived
- (void) instanceMethod {
}
@end

