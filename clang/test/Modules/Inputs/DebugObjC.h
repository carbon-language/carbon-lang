@interface ObjCClass {
  int ivar;
}
+ classMethod;
- instanceMethodWithInt:(int)i;
@property int property;
@end

@interface ObjCClass (Category)
- categoryMethod;
@end
