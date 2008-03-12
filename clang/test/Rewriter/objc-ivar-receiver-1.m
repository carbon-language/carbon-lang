// RUN: clang -rewrite-test %s 
// RUN: clang -rewrite-test %s | grep 'newInv->_container'

@interface NSMutableArray 
- (void)addObject:(id)addObject;
@end

@interface NSInvocation {
@private
    id _container;
}
+ (NSInvocation *)invocationWithMethodSignature;

@end

@implementation NSInvocation

+ (NSInvocation *)invocationWithMethodSignature {
    NSInvocation *newInv;
    id obj = newInv->_container;
    [newInv->_container addObject:0];
   return 0;
}
@end
