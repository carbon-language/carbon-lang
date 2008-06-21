// RUN: clang -rewrite-objc %s -o - | grep objc_msgSendSuper | grep MainMethod

@interface SUPER
- (int) MainMethod;
@end

@interface MyDerived : SUPER
- (int) instanceMethod;
@end

@implementation MyDerived 
- (int) instanceMethod {
  return [super MainMethod];
}
@end
