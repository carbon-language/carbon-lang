// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-ipa=dynamic-bifurcate -verify %s

typedef signed char BOOL;
typedef struct objc_class *Class;
typedef struct objc_object {
    Class isa;
} *id;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@interface NSObject <NSObject> {}
+(id)alloc;
-(id)init;
-(id)autorelease;
-(id)copy;
- (Class)class;
-(id)retain;
@end

@interface MyParent : NSObject
- (int)getZero;
@end
@implementation MyParent
- (int)getZero {
    return 0;
}
@end

@interface MyClass : MyParent
- (int)getZero;
@end

MyClass *getObj();

// Check that we explore both paths - on one the calla are inlined and they are 
// not inlined on the other.
// In this case, p can be either the object of type MyParent* or MyClass*:
// - If it's MyParent*, getZero returns 0.
// - If it's MyClass*, getZero returns 1 and 'return 5/m' is reachable.
@implementation MyClass
+ (int) testTypeFromParam:(MyParent*) p {
  int m = 0;
  int z = [p getZero];
  if (z)
    return 5/m; // expected-warning {{Division by zero}}
  return 5/[p getZero];// expected-warning {{Division by zero}}
}

- (int)getZero {
    return 1;
}

@end