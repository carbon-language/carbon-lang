// RUN: clang -rewrite-test %s

#include <objc/objc.h>

@interface MyDerived
- (void) instanceMethod;
@end

@implementation MyDerived
- (void) instanceMethod {
}
@end

