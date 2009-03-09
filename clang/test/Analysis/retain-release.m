// RUN: clang -analyze -checker-cfref -verify %s &&
// RUN: clang -analyze -checker-cfref -analyzer-store=region -verify %s


//===----------------------------------------------------------------------===//
// The following code is reduced using delta-debugging from
// Foundation.h (Mac OS X).
//
// It includes the basic definitions for the test cases below.
// Not including Foundation.h directly makes this test case both svelte and
// portable to non-Mac platforms.
//===----------------------------------------------------------------------===//

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
@end    @interface NSObject <NSObject> {
}
+ (id)alloc;
@end   typedef float CGFloat;
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>    - (NSUInteger)length;
- (const char *)UTF8String;
- (id)initWithUTF8String:(const char *)nullTerminatedCString;
+ (id)stringWithUTF8String:(const char *)nullTerminatedCString;
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




//===----------------------------------------------------------------------===//
// Test cases.
//===----------------------------------------------------------------------===//

CFAbsoluteTime f1() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t);
  CFRetain(date);
  CFRelease(date);
  CFDateGetAbsoluteTime(date); // no-warning
  CFRelease(date);
  t = CFDateGetAbsoluteTime(date);   // expected-warning{{Reference-counted object is used after it is released.}}
  return t;
}

CFAbsoluteTime f2() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t);  
  [((NSDate*) date) retain];
  CFRelease(date);
  CFDateGetAbsoluteTime(date); // no-warning
  [((NSDate*) date) release];
  t = CFDateGetAbsoluteTime(date);   // expected-warning{{Reference-counted object is used after it is released.}}
  return t;
}


NSDate* global_x;

// Test to see if we supresss an error when we store the pointer
// to a global.

CFAbsoluteTime f3() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t);  
  [((NSDate*) date) retain];
  CFRelease(date);
  CFDateGetAbsoluteTime(date); // no-warning
  global_x = (NSDate*) date;  
  [((NSDate*) date) release];
  t = CFDateGetAbsoluteTime(date);   // no-warning
  return t;
}

//---------------------------------------------------------------------------
// Test case 'f4' differs for region store and basic store.  See
// retain-release-region-store.m and retain-release-basic-store.m.
//---------------------------------------------------------------------------

// Test a leak.

CFAbsoluteTime f5(int x) {  
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t); // expected-warning{{leak}}
  
  if (x)
    CFRelease(date);
  
  return t;
}

// Test a leak involving the return.

CFDateRef f6(int x) {  
  CFDateRef date = CFDateCreate(0, CFAbsoluteTimeGetCurrent());  // expected-warning{{leak}}
  CFRetain(date);
  return date;
}

// Test a leak involving an overwrite.

CFDateRef f7() {
  CFDateRef date = CFDateCreate(0, CFAbsoluteTimeGetCurrent());  //expected-warning{{leak}}
  CFRetain(date);
  date = CFDateCreate(0, CFAbsoluteTimeGetCurrent());
  return date;
}

// Generalization of Create rule.  MyDateCreate returns a CFXXXTypeRef, and
// has the word create.
CFDateRef MyDateCreate();

CFDateRef f8() {
  CFDateRef date = MyDateCreate(); // expected-warning{{leak}}
  CFRetain(date);  
  return date;
}

CFDateRef f9() {
  CFDateRef date = CFDateCreate(0, CFAbsoluteTimeGetCurrent());
  int *p = 0;
  // test that the checker assumes that CFDateCreate returns a non-null
  // pointer
  if (!date) *p = 1; // no-warning
  return date;
}

// Handle DiskArbitration API:
//
// http://developer.apple.com/DOCUMENTATION/DARWIN/Reference/DiscArbitrationFramework/
//
void f10(io_service_t media, DADiskRef d, CFStringRef s) {
  DADiskRef disk = DADiskCreateFromBSDName(kCFAllocatorDefault, 0, "hello"); // expected-warning{{leak}}
  if (disk) NSLog(@"ok");
  
  disk = DADiskCreateFromIOMedia(kCFAllocatorDefault, 0, media); // expected-warning{{leak}}
  if (disk) NSLog(@"ok");

  CFDictionaryRef dict = DADiskCopyDescription(d);  // expected-warning{{leak}}
  if (dict) NSLog(@"ok"); 
  
  disk = DADiskCopyWholeDisk(d); // expected-warning{{leak}}
  if (disk) NSLog(@"ok");
    
  DADissenterRef dissenter = DADissenterCreate(kCFAllocatorDefault,   // expected-warning{{leak}}
                                                kDAReturnSuccess, s);
  if (dissenter) NSLog(@"ok");
  
  DASessionRef session = DASessionCreate(kCFAllocatorDefault);  // expected-warning{{leak}}
  if (session) NSLog(@"ok");
}

// Test retain/release checker with CFString and CFMutableArray.
void f11() {
  // Create the array.
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);

  // Create a string.
  CFStringRef s1 = CFStringCreateWithCString(0, "hello world",
                                             kCFStringEncodingUTF8);

  // Add the string to the array.
  CFArrayAppendValue(A, s1);
  
  // Decrement the reference count.
  CFRelease(s1); // no-warning
  
  // Get the string.  We don't own it.
  s1 = (CFStringRef) CFArrayGetValueAtIndex(A, 0);
  
  // Release the array.
  CFRelease(A); // no-warning
  
  // Release the string.  This is a bug.
  CFRelease(s1); // expected-warning{{Incorrect decrement of the reference count}}
}

// PR 3337: Handle functions declared using typedefs.
typedef CFTypeRef CREATEFUN();
CREATEFUN MyCreateFun;

void f12() {
  CFTypeRef o = MyCreateFun(); // expected-warning {{leak}}
}

void f13_autorelease() {
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
  [(id) A autorelease]; // no-warning
}

// This case exercises the logic where the leak site is the same as the allocation site.
void f14_leakimmediately() {
  CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks); // expected-warning{{leak}}
}

// Test basic tracking of ivars associated with 'self'.  For the retain/release
// checker we currently do not want to flag leaks associated with stores
// of tracked objects to ivars.
@interface SelfIvarTest : NSObject {
  id myObj;
}
- (void)test_self_tracking;
@end

@implementation SelfIvarTest
- (void)test_self_tracking {
  myObj = (id) CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks); // no-warning
}
@end

// <rdar://problem/6659160>
int isFoo(char c);

static void rdar_6659160(char *inkind, char *inname)
{
  // We currently expect that [NSObject alloc] cannot fail.  This
  // will be a toggled flag in the future.  It can indeed return null, but
  // Cocoa programmers generally aren't expected to reason about out-of-memory
  // conditions.
  NSString *kind = [[NSString alloc] initWithUTF8String:inkind];  // expected-warning{{leak}}
  
  // We do allow stringWithUTF8String to fail.  This isn't really correct, as
  // far as returning nil.  In most error conditions it will throw an exception.
  // If allocation fails it could return nil, but again this
  // isn't expected.
  NSString *name = [NSString stringWithUTF8String:inname];
  if(!name)
    return;

  const char *kindC = 0;
  const char *nameC = 0;
  
  // In both cases, we cannot reach a point down below where we
  // dereference kindC or nameC with either being null.  This is because
  // we assume that [NSObject alloc] doesn't fail and that we have the guard
  // up above.
  
  if(kind)
    kindC = [kind UTF8String];
  if(name)
    nameC = [name UTF8String];
  if(!isFoo(kindC[0])) // no-warning
    return;
  if(!isFoo(nameC[0])) // no-warning
    return;

  [kind release];
  [name release];
}

