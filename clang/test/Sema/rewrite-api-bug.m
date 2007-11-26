// RUN: clang -rewrite-test %s

#include <Objc/objc.h>

@interface MyDerived
- (void) instanceMethod;
@end

@implementation MyDerived
- (void) instanceMethod {
}
@end

