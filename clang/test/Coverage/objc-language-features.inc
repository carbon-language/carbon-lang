//-*- ObjC -*-

@protocol P0;

@protocol P1 
-(void) fm0;
@end

@class B;

@interface Root
@end

@interface A : Root <P1> {
  int iv0;
  B *iv1;
}

@property(readonly) int p0;
@property(assign,nonatomic,readwrite) int p1;
@property(copy) id p2;
@property(retain) id p3;
@property(assign, getter=getme, setter=setme:) id p4;
@end

@implementation A
@dynamic p0;
@synthesize p1 = iv0;
+(void) fm0 {
  [super fm0];
}
-(void) im0 {
  const char *s0 = __func__;
  const char *s1 = __FUNCTION__;
  const char *s2 = __PRETTY_FUNCTION__;
  [super im0];
  int x = super.p0;
}
-(void) im1: (int) x, ... {
}
@end

@implementation C : A
@end

@interface A (Cat)
@end

@implementation A (Cat)
@end

@interface B
@end

int f0(id x) {
  @synchronized(x) {
  }

  @try {
    @throw x;

  } @catch(A *e) {
    @throw;

    // @catch param doesn't require name.
  } @catch(B *) {

  } @finally {
    ;
  }

  for (id y in x) {
    break;
  }
}

#ifndef __OBJC2__
struct s0 {
  @defs(A);
};
#endif
