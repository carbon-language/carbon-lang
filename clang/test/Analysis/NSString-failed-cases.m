// RUN: clang -cc1 -triple i386-apple-darwin10 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=region -analyzer-constraints=basic -verify %s
// RUN: clang -cc1 -triple i386-apple-darwin10 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=region -analyzer-constraints=range -verify %s
// RUN: clang -cc1 -DTEST_64 -triple x86_64-apple-darwin10 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=region -analyzer-constraints=basic -verify %s
// RUN: clang -cc1 -DTEST_64 -triple x86_64-apple-darwin10 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=region -analyzer-constraints=range -verify %s
// XFAIL: *

//===----------------------------------------------------------------------===//
// The following code is reduced using delta-debugging from
// Foundation.h (Mac OS X).
//
// It includes the basic definitions for the test cases below.
// Not directly including Foundation.h directly makes this test case 
// both svelte and portable to non-Mac platforms.
//===----------------------------------------------------------------------===//

#ifdef TEST_64
typedef long long int64_t;
_Bool OSAtomicCompareAndSwap64Barrier( int64_t __oldValue, int64_t __newValue, volatile int64_t *__theValue );
#define COMPARE_SWAP_BARRIER OSAtomicCompareAndSwap64Barrier
typedef int64_t intptr_t;
#else
typedef int int32_t;
_Bool OSAtomicCompareAndSwap32Barrier( int32_t __oldValue, int32_t __newValue, volatile int32_t *__theValue );
#define COMPARE_SWAP_BARRIER OSAtomicCompareAndSwap32Barrier
typedef int32_t intptr_t;
#endif

typedef const void * CFTypeRef;
typedef const struct __CFString * CFStringRef;
typedef const struct __CFAllocator * CFAllocatorRef;
extern const CFAllocatorRef kCFAllocatorDefault;
extern CFTypeRef CFRetain(CFTypeRef cf);
void CFRelease(CFTypeRef cf);
typedef const struct __CFDictionary * CFDictionaryRef;
const void *CFDictionaryGetValue(CFDictionaryRef theDict, const void *key);
extern CFStringRef CFStringCreateWithFormat(CFAllocatorRef alloc, CFDictionaryRef formatOptions, CFStringRef format, ...);
typedef signed char BOOL;
typedef int NSInteger;
typedef unsigned int NSUInteger;
@class NSString, Protocol;
extern void NSLog(NSString *format, ...) __attribute__((format(__NSString__, 1, 2)));
typedef NSInteger NSComparisonResult;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject
- (BOOL)isEqual:(id)object;
- (oneway void)release;
- (id)retain;
- (id)autorelease;
@end
@protocol NSCopying
- (id)copyWithZone:(NSZone *)zone;
@end
@protocol NSMutableCopying
- (id)mutableCopyWithZone:(NSZone *)zone;
@end
@protocol NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@interface NSObject <NSObject> {}
- (id)init;
+ (id)alloc;
@end
extern id NSAllocateObject(Class aClass, NSUInteger extraBytes, NSZone *zone);
typedef struct {} NSFastEnumerationState;
@protocol NSFastEnumeration
- (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id *)stackbuf count:(NSUInteger)len;
@end
@class NSString;
typedef struct _NSRange {} NSRange;
@interface NSArray : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>
- (NSUInteger)count;
@end
@interface NSMutableArray : NSArray
- (void)addObject:(id)anObject;
- (id)initWithCapacity:(NSUInteger)numItems;
@end
typedef unsigned short unichar;
@class NSData, NSArray, NSDictionary, NSCharacterSet, NSData, NSURL, NSError, NSLocale;
typedef NSUInteger NSStringCompareOptions;
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>    - (NSUInteger)length;
- (NSComparisonResult)compare:(NSString *)string;
- (NSComparisonResult)compare:(NSString *)string options:(NSStringCompareOptions)mask;
- (NSComparisonResult)compare:(NSString *)string options:(NSStringCompareOptions)mask range:(NSRange)compareRange;
- (NSComparisonResult)compare:(NSString *)string options:(NSStringCompareOptions)mask range:(NSRange)compareRange locale:(id)locale;
- (NSComparisonResult)caseInsensitiveCompare:(NSString *)string;
- (NSArray *)componentsSeparatedByCharactersInSet:(NSCharacterSet *)separator;
+ (id)stringWithFormat:(NSString *)format, ... __attribute__((format(__NSString__, 1, 2)));
@end
@interface NSSimpleCString : NSString {} @end
@interface NSConstantString : NSSimpleCString @end
extern void *_NSConstantStringClassReference;

//===----------------------------------------------------------------------===//
// Test cases.  These should all be merged into NSString.m once these tests
//  stop reporting leaks.
//===----------------------------------------------------------------------===//

// FIXME: THIS TEST CASE INCORRECTLY REPORTS A LEAK.
void testOSCompareAndSwapXXBarrier_parameter(NSString **old) {
  NSString *s = [[NSString alloc] init]; // no-warning
  if (!COMPARE_SWAP_BARRIER((intptr_t) 0, (intptr_t) s, (intptr_t*) old))
    [s release];
  else    
    [*old release];
}

// FIXME: THIS TEST CASE INCORRECTLY REPORTS A LEAK.
void testOSCompareAndSwapXXBarrier_parameter_no_direct_release(NSString **old) {
  NSString *s = [[NSString alloc] init]; // no-warning
  if (!COMPARE_SWAP_BARRIER((intptr_t) 0, (intptr_t) s, (intptr_t*) old))
    return;
  else    
    [*old release];
}
