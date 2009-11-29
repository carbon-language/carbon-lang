// RUN: clang-cc -emit-llvm -o %t %s

int printf(const char *, ...);

@interface Root
-(id) alloc;
-(id) init;
@end

@interface A : Root {
  int x;
  int y, ro, z;
  id ob0, ob1, ob2, ob3, ob4;
}
@property int x;
@property int y;
@property int z;
@property(readonly) int ro;
@property(assign) id ob0;
@property(retain) id ob1;
@property(copy) id ob2;
@property(retain, nonatomic) id ob3;
@property(copy, nonatomic) id ob4;
@end

@implementation A
@dynamic x;
@synthesize y;
@synthesize z = z;
@synthesize ro;
@synthesize ob0;
@synthesize ob1;
@synthesize ob2;
@synthesize ob3;
@synthesize ob4;
-(int) y {
  return x + 1;
}
-(void) setZ: (int) arg {
  x = arg - 1;
}
@end

@interface A (Cat)
@property int dyn;
@end

@implementation A (Cat)
-(int) dyn {
  return 10;
}
@end
