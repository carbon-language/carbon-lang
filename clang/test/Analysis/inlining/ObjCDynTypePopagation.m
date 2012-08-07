// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -analyzer-ipa=dynamic -verify %s

typedef signed char BOOL;
typedef struct objc_class *Class;
typedef struct objc_object {
    Class isa;
} *id;

void clang_analyzer_eval(BOOL);

@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@interface NSObject <NSObject> {}
+(id)alloc;
-(id)init;
+(id)new;
-(id)autorelease;
-(id)copy;
- (Class)class;
-(id)retain;
@end

@interface MyParent : NSObject
- (int)getZeroOverridden;
@end
@implementation MyParent
- (int) getZeroOverridden {
   return 1;
}
- (int) getZero {
   return 0;
}
@end

@interface MyClass : MyParent
- (int) getZeroOverridden;
@end

MyClass *getObj();

@implementation MyClass
- (int) getZeroOverridden {
   return 0;
}

/* Test that we get the right type from call to alloc. */

+ (void) testAllocSelf {
  id a = [self alloc];
  clang_analyzer_eval([a getZeroOverridden] == 0); // expected-warning{{TRUE}}
}


+ (void) testAllocClass {
  id a = [MyClass alloc];
  clang_analyzer_eval([a getZeroOverridden] == 0); // expected-warning{{TRUE}}
}

+ (void) testAllocSuperOverriden {
  id a = [super alloc];
  // Evaluates to 1 in the parent.
  clang_analyzer_eval([a getZeroOverridden] == 0); // expected-warning{{FALSE}} 
}

+ (void) testAllocSuper {
  id a = [super alloc];
  clang_analyzer_eval([a getZero] == 0); // expected-warning{{TRUE}}
}

+ (void) testAllocInit {
  id a = [[self alloc] init];
  clang_analyzer_eval([a getZeroOverridden] == 0); // expected-warning{{TRUE}}
}

+ (void) testNewSelf {
  id a = [self new];
  clang_analyzer_eval([a getZeroOverridden] == 0); // expected-warning{{TRUE}}
}

// Casting to parent should not pessimize the dynamic type. 
+ (void) testCastToParent {
 id a = [[self alloc] init];
  MyParent *p = a;  
  clang_analyzer_eval([p getZeroOverridden] == 0); // expected-warning{{TRUE}}
}

// The type of parameter gets used.
+ (void)testTypeFromParam:(MyParent*) p {
  clang_analyzer_eval([p getZero] == 0); // expected-warning{{TRUE}}
}

// Test implicit cast.
// Note, in this case, p could also be a subclass of MyParent.
+ (void) testCastFromId:(id) a {
  MyParent *p = a;  
  clang_analyzer_eval([p getZero] == 0); // expected-warning{{TRUE}}
}
@end

// TODO: Would be nice to handle the case of dynamically obtained class info
// as well. We need a MemRegion for class types for this.
int testDynamicClass(BOOL coin) {
 Class AllocClass = (coin ? [NSObject class] : [MyClass class]);
 id x = [[AllocClass alloc] init];
 if (coin)
   return [x getZero];
 return 1;
}
