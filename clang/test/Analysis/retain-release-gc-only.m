// RUN: clang-cc -analyze -checker-cfref -verify -fobjc-gc-only %s &&
// RUN: clang-cc -analyze -checker-cfref -analyzer-store=region -fobjc-gc-only -verify %s

//===----------------------------------------------------------------------===//
// Header stuff.
//===----------------------------------------------------------------------===//

typedef struct objc_class *Class;

typedef unsigned int __darwin_natural_t;
typedef struct {} div_t;
typedef unsigned long UInt32;
typedef signed long CFIndex;
typedef const void * CFTypeRef;
typedef const struct __CFString * CFStringRef;
typedef const struct __CFAllocator * CFAllocatorRef;
extern const CFAllocatorRef kCFAllocatorDefault;
extern CFTypeRef CFRetain(CFTypeRef cf);
extern void CFRelease(CFTypeRef cf);
typedef struct {
}
CFArrayCallBacks;
extern const CFArrayCallBacks kCFTypeArrayCallBacks;
typedef const struct __CFArray * CFArrayRef;
typedef struct __CFArray * CFMutableArrayRef;
extern CFMutableArrayRef CFArrayCreateMutable(CFAllocatorRef allocator, CFIndex capacity, const CFArrayCallBacks *callBacks);
extern const void *CFArrayGetValueAtIndex(CFArrayRef theArray, CFIndex idx);
extern void CFArrayAppendValue(CFMutableArrayRef theArray, const void *value);
typedef const struct __CFDictionary * CFDictionaryRef;
typedef UInt32 CFStringEncoding;
enum {
kCFStringEncodingMacRoman = 0,     kCFStringEncodingWindowsLatin1 = 0x0500,     kCFStringEncodingISOLatin1 = 0x0201,     kCFStringEncodingNextStepLatin = 0x0B01,     kCFStringEncodingASCII = 0x0600,     kCFStringEncodingUnicode = 0x0100,     kCFStringEncodingUTF8 = 0x08000100,     kCFStringEncodingNonLossyASCII = 0x0BFF      ,     kCFStringEncodingUTF16 = 0x0100,     kCFStringEncodingUTF16BE = 0x10000100,     kCFStringEncodingUTF16LE = 0x14000100,      kCFStringEncodingUTF32 = 0x0c000100,     kCFStringEncodingUTF32BE = 0x18000100,     kCFStringEncodingUTF32LE = 0x1c000100  };
extern CFStringRef CFStringCreateWithCString(CFAllocatorRef alloc, const char *cStr, CFStringEncoding encoding);
typedef double CFTimeInterval;
typedef CFTimeInterval CFAbsoluteTime;
extern CFAbsoluteTime CFAbsoluteTimeGetCurrent(void);
typedef const struct __CFDate * CFDateRef;
extern CFDateRef CFDateCreate(CFAllocatorRef allocator, CFAbsoluteTime at);
extern CFAbsoluteTime CFDateGetAbsoluteTime(CFDateRef theDate);
typedef __darwin_natural_t natural_t;
typedef natural_t mach_port_name_t;
typedef mach_port_name_t mach_port_t;
typedef struct {
}
CFRunLoopObserverContext;
typedef signed char BOOL;
typedef unsigned int NSUInteger;
@class NSString, Protocol;
extern void NSLog(NSString *format, ...) __attribute__((format(__NSString__, 1, 2)));
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object;
- (id)retain;
- (oneway void)release;
- (id)autorelease;
@end  @protocol NSCopying  - (id)copyWithZone:(NSZone *)zone;
@end  @protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone;
@end  @protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@interface NSObject <NSObject> {}
- (Class)class;
+ (id)alloc;
+ (id)allocWithZone:(NSZone *)zone;
@end   typedef float CGFloat;
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>    - (NSUInteger)length;
- (const char *)UTF8String;
- (id)initWithUTF8String:(const char *)nullTerminatedCString;
+ (id)stringWithUTF8String:(const char *)nullTerminatedCString;
- (id)init;
- (void)dealloc;
@end   extern NSString * const NSCurrentLocaleDidChangeNotification ;
@protocol NSLocking  - (void)lock;
@end  extern NSString * const NSUndoManagerCheckpointNotification;
typedef enum {
ACL_READ_DATA = (1<<1),  ACL_LIST_DIRECTORY = (1<<1),  ACL_WRITE_DATA = (1<<2),  ACL_ADD_FILE = (1<<2),  ACL_EXECUTE = (1<<3),  ACL_SEARCH = (1<<3),  ACL_DELETE = (1<<4),  ACL_APPEND_DATA = (1<<5),  ACL_ADD_SUBDIRECTORY = (1<<5),  ACL_DELETE_CHILD = (1<<6),  ACL_READ_ATTRIBUTES = (1<<7),  ACL_WRITE_ATTRIBUTES = (1<<8),  ACL_READ_EXTATTRIBUTES = (1<<9),  ACL_WRITE_EXTATTRIBUTES = (1<<10),  ACL_READ_SECURITY = (1<<11),  ACL_WRITE_SECURITY = (1<<12),  ACL_CHANGE_OWNER = (1<<13) }
acl_entry_id_t;
typedef int kern_return_t;
typedef kern_return_t mach_error_t;
typedef mach_port_t io_object_t;
typedef io_object_t io_service_t;
typedef struct __DASession * DASessionRef;
extern DASessionRef DASessionCreate( CFAllocatorRef allocator );
typedef struct __DADisk * DADiskRef;
extern DADiskRef DADiskCreateFromBSDName( CFAllocatorRef allocator, DASessionRef session, const char * name );
extern DADiskRef DADiskCreateFromIOMedia( CFAllocatorRef allocator, DASessionRef session, io_service_t media );
extern CFDictionaryRef DADiskCopyDescription( DADiskRef disk );
extern DADiskRef DADiskCopyWholeDisk( DADiskRef disk );
@interface NSResponder : NSObject <NSCoding> {
}
@end  @class NSColor, NSFont, NSNotification;
typedef struct __CFlags {
}
_CFlags;
@interface NSCell : NSObject <NSCopying, NSCoding> {
}
@end  @class NSDate, NSDictionary, NSError, NSException, NSNotification;
@interface NSManagedObjectContext : NSObject <NSCoding, NSLocking> {
}
@end enum {
kDAReturnSuccess = 0,     kDAReturnError = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x01,     kDAReturnBusy = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x02,     kDAReturnBadArgument = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x03,     kDAReturnExclusiveAccess = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x04,     kDAReturnNoResources = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x05,     kDAReturnNotFound = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x06,     kDAReturnNotMounted = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x07,     kDAReturnNotPermitted = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x08,     kDAReturnNotPrivileged = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x09,     kDAReturnNotReady = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x0A,     kDAReturnNotWritable = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x0B,     kDAReturnUnsupported = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x0C };
typedef mach_error_t DAReturn;
typedef const struct __DADissenter * DADissenterRef;
extern DADissenterRef DADissenterCreate( CFAllocatorRef allocator, DAReturn status, CFStringRef string );

CFTypeRef CFMakeCollectable(CFTypeRef cf) ;

static __inline__ __attribute__((always_inline)) id NSMakeCollectable(CFTypeRef 
cf) {
    return cf ? (id)CFMakeCollectable(cf) : ((void*)0);
}

//===----------------------------------------------------------------------===//
// Test cases.
//===----------------------------------------------------------------------===//

void f1() {
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks); // no-warning
  id x = [(id) A autorelease];
  CFRelease((CFMutableArrayRef) x);
}

void f2() {
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks); // expected-warning{{leak}}
  id x = [(id) A retain];
  [x release];
  [x release];
}

void f3() {
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks); // expected-warning{{leak}}
  CFMakeCollectable(A);
  CFRetain(A);
}

void f3b() {
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks); // no-warning
  CFMakeCollectable(A);
}


void f4() {
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks); // expected-warning{{leak}}
  NSMakeCollectable(A);
  CFRetain(A);
}

void f4b() {
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks); // no-warning
  NSMakeCollectable(A);
}

void f5() {
  id x = [NSMakeCollectable(CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks)) autorelease]; // no-warning
}

void f5b() {
  id x = [(id) CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks) autorelease]; // expected-warning{{leak}}
}

// Test return of non-owned objects in contexts where an owned object
// is expected.
@interface TestReturnNotOwnedWhenExpectedOwned
- (NSString*)newString;
- (CFMutableArrayRef)newArray;
@end

@implementation TestReturnNotOwnedWhenExpectedOwned
- (NSString*)newString {
  NSString *s = [NSString stringWithUTF8String:"hello"]; // expected-warning{{Potential leak (when using garbage collection) of an object allocated}}
  CFRetain(s);
  return s;
}
- (CFMutableArrayRef)newArray{
   return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks); // no-warning
}
@end

//===----------------------------------------------------------------------===//
// <rdar://problem/6948053> False positive: object substitution during -init*
//   methods warns about returning +0 when using -fobjc-gc-only
//===----------------------------------------------------------------------===//

@interface MyClassRdar6948053 : NSObject
- (id) init;
+ (id) shared;
@end

@implementation MyClassRdar6948053
+(id) shared {
  return (id) 0;
}
- (id) init
{
  Class myClass = [self class];  
  [self release];
  return [[myClass shared] retain]; // no-warning
}
@end

//===----------------------------------------------------------------------===//
// Tests of ownership attributes.
//===----------------------------------------------------------------------===//

@interface TestOwnershipAttr : NSObject
- (NSString*) returnsAnOwnedString __attribute__((ns_returns_retained));
- (NSString*) returnsAnOwnedCFString  __attribute__((cf_returns_retained));
@end

void test_attr_1(TestOwnershipAttr *X) {
  NSString *str = [X returnsAnOwnedString]; // no-warning
}

void test_attr_1b(TestOwnershipAttr *X) {
  NSString *str = [X returnsAnOwnedCFString]; // expected-warning{{leak}}
}

@interface MyClassTestCFAttr : NSObject {}
- (NSDate*) returnsCFRetained __attribute__((cf_returns_retained));
- (NSDate*) alsoReturnsRetained;
- (NSDate*) returnsNSRetained __attribute__((ns_returns_retained));
@end

__attribute__((cf_returns_retained))
CFDateRef returnsRetainedCFDate()  {
  return CFDateCreate(0, CFAbsoluteTimeGetCurrent());
}

@implementation MyClassTestCFAttr
- (NSDate*) returnsCFRetained {
  return (NSDate*) returnsRetainedCFDate(); // No leak.
}

- (NSDate*) alsoReturnsRetained {
  return (NSDate*) returnsRetainedCFDate(); // expected-warning{{leak}}
}

- (NSDate*) returnsNSRetained {
  return (NSDate*) returnsRetainedCFDate(); // expected-warning{{leak}}
}
@end