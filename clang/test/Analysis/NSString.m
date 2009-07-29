// RUN: clang-cc -triple i386-pc-linux-gnu -analyze -checker-cfref -analyzer-store=basic -analyzer-constraints=basic -verify %s &&
// RUN: clang-cc -triple i386-pc-linux-gnu -analyze -checker-cfref -analyzer-store=basic -analyzer-constraints=range -verify %s &&
// RUN: clang-cc -triple i386-pc-linux-gnu -analyze -checker-cfref -analyzer-store=region -analyzer-constraints=basic -verify %s &&
// RUN: clang-cc -triple i386-pc-linux-gnu -analyze -checker-cfref -analyzer-store=region -analyzer-constraints=range -verify %s

//===----------------------------------------------------------------------===//
// The following code is reduced using delta-debugging from
// Foundation.h (Mac OS X).
//
// It includes the basic definitions for the test cases below.
// Not directly including Foundation.h directly makes this test case 
// both svelte and portable to non-Mac platforms.
//===----------------------------------------------------------------------===//

typedef int int32_t;
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
// Test cases.
//===----------------------------------------------------------------------===//

NSComparisonResult f1(NSString* s) {
  NSString *aString = 0;
  return [s compare:aString]; // expected-warning {{Argument to 'NSString' method 'compare:' cannot be nil.}}
}

NSComparisonResult f2(NSString* s) {
  NSString *aString = 0;
  return [s caseInsensitiveCompare:aString]; // expected-warning {{Argument to 'NSString' method 'caseInsensitiveCompare:' cannot be nil.}}
}

NSComparisonResult f3(NSString* s, NSStringCompareOptions op) {
  NSString *aString = 0;
  return [s compare:aString options:op]; // expected-warning {{Argument to 'NSString' method 'compare:options:' cannot be nil.}}
}

NSComparisonResult f4(NSString* s, NSStringCompareOptions op, NSRange R) {
  NSString *aString = 0;
  return [s compare:aString options:op range:R]; // expected-warning {{Argument to 'NSString' method 'compare:options:range:' cannot be nil.}}
}

NSComparisonResult f5(NSString* s, NSStringCompareOptions op, NSRange R) {
  NSString *aString = 0;
  return [s compare:aString options:op range:R locale:0]; // expected-warning {{Argument to 'NSString' method 'compare:options:range:locale:' cannot be nil.}}
}

NSArray *f6(NSString* s) {
  return [s componentsSeparatedByCharactersInSet:0]; // expected-warning {{Argument to 'NSString' method 'componentsSeparatedByCharactersInSet:' cannot be nil.}}
}

NSString* f7(NSString* s1, NSString* s2, NSString* s3) {

  NSString* s4 = (NSString*)
    CFStringCreateWithFormat(kCFAllocatorDefault, 0,  // expected-warning{{leak}}
                             (CFStringRef) __builtin___CFStringMakeConstantString("%@ %@ (%@)"), 
                             s1, s2, s3);

  CFRetain(s4);
  return s4;
}

NSMutableArray* f8() {
  
  NSString* s = [[NSString alloc] init];
  NSMutableArray* a = [[NSMutableArray alloc] initWithCapacity:2];
  [a addObject:s];
  [s release]; // no-warning
  return a;
}

void f9() {
  
  NSString* s = [[NSString alloc] init];
  NSString* q = s;
  [s release];
  [q release]; // expected-warning {{used after it is released}}
}

NSString* f10() {
  static NSString* s = 0;
  if (!s) s = [[NSString alloc] init];
  return s; // no-warning
}

// Test case for regression reported in <rdar://problem/6452745>.
// Essentially 's' should not be considered allocated on the false branch.
// This exercises the 'EvalAssume' logic in GRTransferFuncs (CFRefCount.cpp).
NSString* f11(CFDictionaryRef dict, const char* key) {
  NSString* s = (NSString*) CFDictionaryGetValue(dict, key);
  [s retain];
  if (s) {
    [s release];
  }
  return 0;
}

// Test case for passing a tracked object by-reference to a function we
// don't understand.
void unknown_function_f12(NSString** s);
void f12() {
  NSString *string = [[NSString alloc] init];
  unknown_function_f12(&string); // no-warning
}

// Test double release of CFString (PR 4014).
void f13(void) {
  CFStringRef ref = CFStringCreateWithFormat(kCFAllocatorDefault, ((void*)0), ((CFStringRef) __builtin___CFStringMakeConstantString ("" "%d" "")), 100);
  CFRelease(ref);
  CFRelease(ref); // expected-warning{{Reference-counted object is used after it is released}}
}

// Test regular use of -autorelease
@interface TestAutorelease
-(NSString*) getString;
@end
@implementation TestAutorelease
-(NSString*) getString {
  NSString *str = [[NSString alloc] init];
  return [str autorelease]; // no-warning
}
- (void)m1
{
 NSString *s = [[NSString alloc] init]; // expected-warning{{leak}}
 [s retain];
 [s autorelease];
}
- (void)m2
{
 NSString *s = [[[NSString alloc] init] autorelease]; // expected-warning{{leak}}
 [s retain];
}
- (void)m3
{
 NSString *s = [[[NSString alloc] init] autorelease];
 [s retain];
 [s autorelease];
}
- (void)m4
{
 NSString *s = [[NSString alloc] init]; // expected-warning{{leak}}
 [s retain];
}
- (void)m5
{
 NSString *s = [[NSString alloc] init];
 [s autorelease];
}
@end

@interface C1 : NSObject {}
- (NSString*) getShared;
+ (C1*) sharedInstance;
@end
@implementation C1 : NSObject {}
- (NSString*) getShared {
  static NSString* s = 0;
  if (!s) s = [[NSString alloc] init];    
  return s; // no-warning  
}
+ (C1 *)sharedInstance {
  static C1 *sharedInstance = 0;
  if (!sharedInstance) {
    sharedInstance = [[C1 alloc] init];
  }
  return sharedInstance; // no-warning
}
@end

@interface SharedClass : NSObject
+ (id)sharedInstance;
- (id)notShared;
@end

@implementation SharedClass

- (id)_init {
    if ((self = [super init])) {
        NSLog(@"Bar");
    }
    return self;
}

- (id)notShared {
  return [[SharedClass alloc] _init]; // expected-warning{{leak}}
}

+ (id)sharedInstance {
    static SharedClass *_sharedInstance = 0;
    if (!_sharedInstance) {
        _sharedInstance = [[SharedClass alloc] _init];
    }
    return _sharedInstance; // no-warning
}
@end

id testSharedClassFromFunction() {
  return [[SharedClass alloc] _init]; // no-warning
}

// Test OSCompareAndSwap
_Bool OSAtomicCompareAndSwapPtr( void *__oldValue, void *__newValue, void * volatile *__theValue );
_Bool OSAtomicCompareAndSwap32Barrier( int32_t __oldValue, int32_t __newValue, volatile int32_t *__theValue );
extern BOOL objc_atomicCompareAndSwapPtr(id predicate, id replacement, volatile id *objectLocation);

void testOSCompareAndSwap() {
  NSString *old = 0;
  NSString *s = [[NSString alloc] init]; // no-warning
  if (!OSAtomicCompareAndSwapPtr(0, s, (void**) &old))
    [s release];
  else    
    [old release];
}

void testOSCompareAndSwap32Barrier() {
  NSString *old = 0;
  NSString *s = [[NSString alloc] init]; // no-warning
  if (!OSAtomicCompareAndSwap32Barrier((int32_t) 0, (int32_t) s, (int32_t*) &old))
    [s release];
  else    
    [old release];
}

int testOSCompareAndSwap32Barrier_id(Class myclass, id xclass) {
  if (OSAtomicCompareAndSwap32Barrier(0, (int32_t) myclass, (int32_t*) &xclass))
    return 1;
  return 0;
}  

void test_objc_atomicCompareAndSwap() {
  NSString *old = 0;
  NSString *s = [[NSString alloc] init]; // no-warning
  if (!objc_atomicCompareAndSwapPtr(0, s, &old))
    [s release];
  else    
    [old release];
}

// Test stringWithFormat (<rdar://problem/6815234>)
void test_stringWithFormat() {  
  NSString *string = [[NSString stringWithFormat:@"%ld", (long) 100] retain];
  [string release];
  [string release]; // expected-warning{{Incorrect decrement of the reference count}}
}

// Test isTrackedObjectType().
typedef NSString* WonkyTypedef;
@interface TestIsTracked
+ (WonkyTypedef)newString;
@end

void test_isTrackedObjectType(void) {
  NSString *str = [TestIsTracked newString]; // expected-warning{{Potential leak}}
}

// Test isTrackedCFObjectType().
@interface TestIsCFTracked
+ (CFStringRef) badNewCFString;
+ (CFStringRef) newCFString;
@end

@implementation TestIsCFTracked
+ (CFStringRef) newCFString {
  return CFStringCreateWithFormat(kCFAllocatorDefault, ((void*)0), ((CFStringRef) __builtin___CFStringMakeConstantString ("" "%d" "")), 100); // no-warning
}
+ (CFStringRef) badNewCFString {
  return CFStringCreateWithFormat(kCFAllocatorDefault, ((void*)0), ((CFStringRef) __builtin___CFStringMakeConstantString ("" "%d" "")), 100); // expected-warning{{leak}}
}

// Test @synchronized
void test_synchronized(id x) {
  @synchronized(x) {
    NSString *string = [[NSString stringWithFormat:@"%ld", (long) 100] retain]; // expected-warning {{leak}}
  }
}


