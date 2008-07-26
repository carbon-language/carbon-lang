// RUN: clang -fsyntax-only -verify %s

@interface foo
- (void)meth;
@end

@implementation foo
- (void) contents {}			// No declaration in @interface!
- (void) meth { [self contents]; } 
@end

typedef struct _NSPoint {
    float x;
    float y;
} NSPoint;

typedef struct _NSSize {
    float width; 
    float height;
} NSSize;

typedef struct _NSRect {
    NSPoint origin;
    NSSize size;
} NSRect;

@interface AnyClass
- (NSRect)rect;
@end

@class Helicopter;

static void func(Helicopter *obj) {
  // Note that the proto for "rect" is found in the global pool even when
  // a statically typed object's class interface isn't in scope! This 
  // behavior isn't very desirable, however wee need it for GCC compatibility.
  NSRect r = [obj rect];
}

@interface NSObject @end

extern Class NSClassFromObject(id object);

@interface XX : NSObject 
@end

@implementation XX

+ _privateMethod {
  return self;
}

- (void) xx {
  [NSClassFromObject(self) _privateMethod];
}
@end

@implementation XX (Private)
- (void) yy {
  [NSClassFromObject(self) _privateMethod];
}
@end

