/* For use with the methods.m test */

@interface TestPCH
+ alloc;
- (instancetype)instMethod;
@end

@class TestForwardClassDecl;

// FIXME: @compatibility_alias  AliasForTestPCH TestPCH;

// http://llvm.org/PR12689
@interface PR12689
@end

@implementation PR12689
-(void)mugi:(int)x {
  switch(x) {
    case 23: {}
  }
}
-(void)bonk:(int)x {
  switch(x) {
    case 42: {}
  }
}
@end
