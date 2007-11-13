// RUN: clang -rewrite-test %s

@interface Intf 
@end

@implementation Intf(Category)
- (void) CatMeth {}
@end

@implementation Another
- (void) CatMeth {}
@end
