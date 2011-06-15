// RUN: arcmt-test -check-only -verify --args %s

#if __has_feature(objc_arr)
#define NS_AUTOMATED_REFCOUNT_UNAVAILABLE __attribute__((unavailable("not available in automatic reference counting mode")))
#else
#define NS_AUTOMATED_REFCOUNT_UNAVAILABLE
#endif

typedef struct _NSZone NSZone;
typedef int BOOL;
typedef unsigned NSUInteger;

@protocol NSObject
- (BOOL)isEqual:(id)object;
- (id)retain NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
- (NSUInteger)retainCount NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
- (oneway void)release NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
- (id)autorelease NS_AUTOMATED_REFCOUNT_UNAVAILABLE;

- (NSZone *)zone NS_AUTOMATED_REFCOUNT_UNAVAILABLE; // expected-note {{marked unavailable here}}
@end

@protocol NSCopying
- (id)copyWithZone:(NSZone *)zone;
@end

@protocol NSMutableCopying
- (id)mutableCopyWithZone:(NSZone *)zone;
@end

@interface NSObject <NSObject> {}
- (id)init;

+ (id)allocWithZone:(NSZone *)zone NS_AUTOMATED_REFCOUNT_UNAVAILABLE; // expected-note 2 {{marked unavailable here}}
+ (id)alloc;
- (void)dealloc;

- (void)finalize;

- (id)copy;
- (id)mutableCopy;

+ (id)copyWithZone:(NSZone *)zone NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
+ (id)mutableCopyWithZone:(NSZone *)zone NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
@end

extern void NSRecycleZone(NSZone *zone);

id IhaveSideEffect();

@interface Foo : NSObject <NSCopying, NSMutableCopying> {
  id bar;
}
@property (retain) id bar;
-(id)test:(id)obj;
@end

@implementation Foo

@synthesize bar;

-(id)test:(id)obj {
  Foo *foo1 = [[Foo allocWithZone:[self zone]] init];
  Foo *foo2 = [[Foo allocWithZone:[super zone]] init];
  Foo *foo3 = [[Foo allocWithZone:[IhaveSideEffect() zone]] init]; // expected-error {{not available}}
  NSRecycleZone([self zone]); // expected-error {{not available}}
  
  foo1 = [foo1 copyWithZone:[self zone]];
  foo2 = [foo1 copyWithZone:[super zone]];
  foo3 = [foo1 copyWithZone:[IhaveSideEffect() zone]];
  foo1 = [foo1 mutableCopyWithZone:[self zone]];

  return foo1;
}

+(id)allocWithZone:(NSZone *)zone {
  return [super allocWithZone:zone]; // expected-error {{not available in automatic reference counting mode}}
}

@end
