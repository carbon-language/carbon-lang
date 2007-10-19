#import <stdio.h>

@interface Foo {
  char c;
  short s;
  int i;
  long l;
  float f;
  double d;
}
-(Foo*)retain;
@end

struct Foo *bork(Foo *FooArray) {
  struct Foo *result = 0;
  @try {
    result = [FooArray retain];
  } @catch(id any) {
    printf("hello world\n");
  }

  return result;
}
