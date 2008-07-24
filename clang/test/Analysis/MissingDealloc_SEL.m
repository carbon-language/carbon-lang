// RUN: clang -warn-objc-missing-dealloc -verify %s

typedef struct objc_selector *SEL;
typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@protocol NSObject
- (BOOL)isEqual:(id)object;
@end
@interface NSObject <NSObject> {}
- (id)init;
@end

@interface TestSELs : NSObject {
  SEL a;
  SEL b;
}

@end

@implementation TestSELs // no-warning
- (id)init {
  if( (self = [super init]) ) {
    a = @selector(a);
    b = @selector(b);
  }

  return self;
}
@end
