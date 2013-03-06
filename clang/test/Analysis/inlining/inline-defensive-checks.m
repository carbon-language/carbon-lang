// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-config suppress-inlined-defensive-checks=true -verify %s

typedef signed char BOOL;
typedef struct objc_class *Class;
typedef struct objc_object {
  Class isa;
} *id;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@interface NSObject <NSObject> {}
+(id)alloc;
+(id)new;
-(id)init;
-(id)autorelease;
-(id)copy;
- (Class)class;
-(id)retain;
@end

// expected-no-diagnostics
// Check that inline defensive checks is triggered for null expressions
// within CompoundLiteralExpr.
typedef union {
  struct dispatch_object_s *_do;
  struct dispatch_source_s *_ds;
} dispatch_object_t __attribute__((__transparent_union__));
typedef struct dispatch_source_s *dispatch_source_t;

extern __attribute__((visibility("default"))) __attribute__((__nonnull__)) __attribute__((__nothrow__))
void
dispatch_resume(dispatch_object_t object);

@interface AppDelegate : NSObject {
@protected
	dispatch_source_t p;
}
@end
@implementation AppDelegate
- (void)updateDeleteTimer {
	if (p != ((void*)0))
		;
}
- (void)createAndStartDeleteTimer {
  [self updateDeleteTimer];
  dispatch_resume(p); // no warning
}
@end
