// RUN: %clang_cc1 -fsyntax-only -x objective-c -fobjc-arc -verify -Wno-objc-root-class %s
// rdar://18768214


@class NSArray;
typedef const struct __attribute__((objc_bridge(NSArray))) __CFArray * CFArrayRef;
@class NSString;
typedef const void * CFTypeRef;

typedef const struct __attribute__((objc_bridge(NSString))) __CFString * CFStringRef;

typedef long NSInteger;
typedef unsigned long NSUInteger;

@interface NSObject {
    Class isa __attribute__((deprecated));
}
+ (void)initialize;
- (instancetype)init;
+ (instancetype)new;
+ (instancetype)alloc;
- (void)dealloc;
@end

@interface NSArray : NSObject
@property (readonly) NSUInteger count;
- (id)objectAtIndex:(NSUInteger)index;
- (instancetype)init __attribute__((objc_designated_initializer));
- (instancetype)initWithObjects:(const id [])objects count:(NSUInteger)cnt __attribute__((objc_designated_initializer));
+ (instancetype)array;
+ (instancetype)arrayWithObject:(id)anObject;
+ (instancetype)arrayWithObjects:(const id [])objects count:(NSUInteger)cnt;
+ (instancetype)arrayWithObjects:(id)firstObj, ... __attribute__((sentinel(0,1)));
@end

static CFStringRef _s;

CFArrayRef _array()
{
    return (__bridge CFArrayRef)@[(__bridge NSString *)_s]; // expected-warning {{__bridge cast of collection literal of type 'NSArray *' to "bridgeable" C type 'CFArrayRef' (aka 'const struct __CFArray *') causes early release of the collection}}
}
