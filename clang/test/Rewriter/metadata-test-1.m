// RUN: %clang_cc1 -rewrite-objc -fobjc-fragile-abi  %s -o -

@interface Intf 
@end

@implementation Intf(Category)
- (void) CatMeth {}
@end

@implementation Another
- (void) CatMeth {}
@end
