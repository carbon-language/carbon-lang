// RUN: clang %s -fsyntax-only
#import <Foundation/NSObject.h>

struct D {
  double d;
};

@interface Foo : NSObject 

- method:(int)a;
- method:(int)a;

@end

@interface Bar : NSObject 

- method:(void *)a;

@end

@interface Car : NSObject 

- method:(struct D)a;

@end

@interface Zar : NSObject 

- method:(float)a;

@end

@interface Rar : NSObject 

- method:(float)a;

@end

int main() {
  id xx = [[Car alloc] init];

  [xx method:4];
}
