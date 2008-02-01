// RUN: clang -rewrite-test %s | clang

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
