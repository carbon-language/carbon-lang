// RUN: clang -cc1 -verify %s 

typedef struct objc_class *Class;
@interface XX

- (void)addObserver:(XX*)o;

@end

@interface YY

+ (void)classMethod;

@end

@implementation YY

static XX *obj;

+ (void)classMethod {
  [obj addObserver:self];
  Class whatever;
  [obj addObserver:whatever]; // GCC warns about this.
}
@end

