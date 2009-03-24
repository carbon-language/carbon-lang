// RUN: clang-cc %s -fsyntax-only -verify
@interface A
{
  int ivar;
}
@end

@interface B : A
- (int)ivar;
@end

@implementation B
- (int)ivar {
  return ivar;
} 
@end
