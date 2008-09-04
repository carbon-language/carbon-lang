// RUN: clang -fnext-runtime --emit-llvm -o %t %s

#include <stdio.h>

@interface Root
-(id) alloc;
-(id) init;
@end

@interface A : Root {
  int x;
}
@property int x;
@property int y;
@property int z;
@property(readonly) int ro;
@end

@implementation A
@dynamic x;
@synthesize x;
@synthesize y = x;
@synthesize z = x;
@synthesize ro = x;
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
