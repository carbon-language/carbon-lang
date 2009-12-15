// RUN: %clang_cc1 -emit-llvm -o %t %s

int printf(const char *, ...);

@interface Root
@end

typedef struct {
  int x, y, z[10];
} MyPoint;
typedef struct {
  float width, height;
} MySize;

@interface A : Root
+(void) printThisInt: (int) arg0 andThatFloat: (float) arg1 andADouble: (double) arg2 andAPoint: (MyPoint) arg3;
+(float) returnAFloat;
+(double) returnADouble;
+(MyPoint) returnAPoint;
+(void) printThisSize: (MySize) arg0;
+(MySize) returnASize;

-(void) printThisInt: (int) arg0 andThatFloat: (float) arg1 andADouble: (double) arg2 andAPoint: (MyPoint) arg3;
-(float) returnAFloat;
-(double) returnADouble;
-(MyPoint) returnAPoint;
-(void) printThisSize: (MySize) arg0;
-(MySize) returnASize;
@end
@interface B : A
@end

@implementation A
+(void) printThisInt: (int) arg0 andThatFloat: (float) arg1 andADouble: (double) arg2 andAPoint: (MyPoint) arg3 {
  printf("(CLASS) theInt: %d, theFloat: %f, theDouble: %f, thePoint: { %d, %d }\n",
         arg0, arg1, arg2, arg3.x, arg3.y);
}
+(float) returnAFloat {
  return 15.;
}
+(double) returnADouble {
  return 25.;
}
+(MyPoint) returnAPoint {
  MyPoint x = { 35, 45 };
  return x;
}
+(void) printThisSize: (MySize) arg0 {
  printf("(CLASS) theSize: { %f, %f }\n",
         arg0.width, arg0.height);
}
+(MySize) returnASize {
  MySize x = { 32, 44 };
  return x;
}

-(void) printThisInt: (int) arg0 andThatFloat: (float) arg1 andADouble: (double) arg2 andAPoint: (MyPoint) arg3 {
  printf("theInt: %d, theFloat: %f, theDouble: %f, thePoint: { %d, %d }\n",
         arg0, arg1, arg2, arg3.x, arg3.y);
}
-(float) returnAFloat {
  return 10.;
}
-(double) returnADouble {
  return 20.;
}
-(MyPoint) returnAPoint {
  MyPoint x = { 30, 40 };
  return x;
}
-(void) printThisSize: (MySize) arg0 {
  printf("theSize: { %f, %f }\n",
         arg0.width, arg0.height);
}
-(MySize) returnASize {
  MySize x = { 22, 34 };
  return x;
}
@end

@implementation B
+(void) printThisInt: (int) arg0 andThatFloat: (float) arg1 andADouble: (double) arg2 andAPoint: (MyPoint) arg3 {
  arg3.x *= 2;
  arg3.y *= 2;
  [ super printThisInt: arg0*2 andThatFloat: arg1*2 andADouble: arg2*2 andAPoint: arg3 ];
}
+(void) printThisSize: (MySize) arg0 {
  arg0.width *= 2;
  arg0.height *= 2;
  [ super printThisSize: arg0 ];
}
+(float) returnAFloat {
  return [ super returnAFloat ]*2;
}
+(double) returnADouble {
  return [ super returnADouble ]*2;
}
+(MyPoint) returnAPoint {
  MyPoint x = [ super returnAPoint ];
  x.x *= 2;
  x.y *= 2;
  return x;
}
+(MySize) returnASize {
  MySize x = [ super returnASize ];
  x.width *= 2;
  x.height *= 2;
  return x;
}

-(void) printThisInt: (int) arg0 andThatFloat: (float) arg1 andADouble: (double) arg2 andAPoint: (MyPoint) arg3 {
  arg3.x *= 2;
  arg3.y *= 2;
  [ super printThisInt: arg0*2 andThatFloat: arg1*2 andADouble: arg2*2 andAPoint: arg3 ];
}
-(void) printThisSize: (MySize) arg0 {
  arg0.width *= 2;
  arg0.height *= 2;
  [ super printThisSize: arg0 ];
}
-(float) returnAFloat {
  return [ super returnAFloat ]*2;
}
-(double) returnADouble {
  return [ super returnADouble ]*2;
}
-(MyPoint) returnAPoint {
  MyPoint x = [ super returnAPoint ];
  x.x *= 2;
  x.y *= 2;
  return x;
}
-(MySize) returnASize {
  MySize x = [ super returnASize ];
  x.width *= 2;
  x.height *= 2;
  return x;
}
@end
