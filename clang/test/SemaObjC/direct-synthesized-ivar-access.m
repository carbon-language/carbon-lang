// RUN: %clang_cc1 -Wnonfragile-abi2 -fsyntax-only -fobjc-default-synthesize-properties -verify -Wno-objc-root-class %s
// rdar://8673791
// rdar://9943851

@interface I {
}

@property int IVAR; 
- (int) OK;
@end

@implementation I
- (int) Meth { return _IVAR; }
- (int) OK { return self.IVAR; }
@end
