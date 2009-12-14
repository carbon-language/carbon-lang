// RUN: clang -cc1 -rewrite-objc %s -o -

@interface Intf 
@end

@implementation Intf(Category)
- (void) CatMeth {}
@end

@implementation Another
- (void) CatMeth {}
@end
