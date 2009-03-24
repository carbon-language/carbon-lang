// RUN: clang-cc %s -fsyntax-only -verify

@interface Test {
   int x;
}

-(void) setX: (int) d;
@end

extern struct foo x;

@implementation Test

-(void) setX: (int) n {
   x = n;
}

@end
