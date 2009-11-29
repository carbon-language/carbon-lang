// RUN: clang-cc -rewrite-objc %s -o -

@interface MyDerived
- (void) instanceMethod;
@end

@implementation MyDerived
- (void) instanceMethod {
}
@end

