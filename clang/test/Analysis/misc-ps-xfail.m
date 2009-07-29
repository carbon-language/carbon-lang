// RUN: clang-cc -analyze -checker-cfref --analyzer-store=basic -analyzer-constraints=basic --verify -fblocks %s &&
// RUN: clang-cc -analyze -checker-cfref --analyzer-store=basic -analyzer-constraints=range --verify -fblocks %s &&
// RUN: clang-cc -analyze -checker-cfref --analyzer-store=region -analyzer-constraints=basic --verify -fblocks %s &&
// RUN: clang-cc -analyze -checker-cfref --analyzer-store=region -analyzer-constraints=range --verify -fblocks %s
// XFAIL
typedef struct objc_ivar *Ivar;
typedef struct objc_selector *SEL;
typedef signed char BOOL;
typedef int NSInteger;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSArray, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject
- (BOOL)isEqual:(id)object;
- (id)autorelease;
@end
@protocol NSCopying
- (id)copyWithZone:(NSZone *)zone;
@end
@protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone; @end
@protocol NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@interface NSObject <NSObject> {}
- (id)init;
+ (id)allocWithZone:(NSZone *)zone;
@end
extern id NSAllocateObject(Class aClass, NSUInteger extraBytes, NSZone *zone);
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>
- (NSUInteger)length;
+ (id)stringWithUTF8String:(const char *)nullTerminatedCString;
@end extern NSString * const NSBundleDidLoadNotification;
@interface NSValue : NSObject <NSCopying, NSCoding>
- (void)getValue:(void *)value;
@end
@interface NSNumber : NSValue
- (char)charValue;
- (id)initWithBool:(BOOL)value;
@end
@interface NSAssertionHandler : NSObject {}
+ (NSAssertionHandler *)currentHandler;
- (void)handleFailureInMethod:(SEL)selector object:(id)object file:(NSString *)fileName lineNumber:(NSInteger)line description:(NSString *)format,...;
@end
extern NSString * const NSConnectionReplyMode;
typedef float CGFloat;
typedef struct _NSPoint {
    CGFloat x;
    CGFloat y;
} NSPoint;
typedef struct _NSSize {
    CGFloat width;
    CGFloat height;
} NSSize;
typedef struct _NSRect {
    NSPoint origin;
    NSSize size;
} NSRect;

// *** This case currently crashes for RegionStore ***
// Reduced from a crash involving the cast of an Objective-C symbolic region to
// 'char *'
static NSNumber *test_ivar_offset(id self, SEL _cmd, Ivar inIvar) {
  return [[[NSNumber allocWithZone:((void*)0)] initWithBool:*(_Bool *)((char *)self + ivar_getOffset(inIvar))] autorelease];
}
