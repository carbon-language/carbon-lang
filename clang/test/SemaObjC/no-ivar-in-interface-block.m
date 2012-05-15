// RUN: %clang_cc1  -fsyntax-only -verify -Winterface-block-ivar %s
// rdar://10763173

@interface I
{
  @protected  int P_IVAR; // expected-warning {{declaration of ivar in the interface block is deprecated}}

  @public     int PU_IVAR; // expected-warning {{declaration of ivar in the interface block is deprecated}}

  @private    int PRV_IVAR; // expected-warning {{declaration of ivar in the interface block is deprecated}}
}
@end

@interface I()
{
  int I1;
  int I2;
}
@end

@interface I()
{
  int I3, I4;
}
@end

@implementation I
{
  int I5;
  int I6;
}
@end 
