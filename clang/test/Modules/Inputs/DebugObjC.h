@class FwdDecl;

@interface ObjCClass {
  int ivar;
}
+ classMethod;
- instanceMethodWithInt:(int)i;
- (struct OpaqueData*) getSomethingOpaque;
@property int property;
@end

@interface ObjCClass (Category)
- categoryMethod;
@end
