// RUN: clang-cc -analyze -checker-cfref -verify %s &&
// RUN: clang-cc -analyze -checker-cfref -analyzer-store=region -verify %s


//===----------------------------------------------------------------------===//
// The following code is reduced using delta-debugging from
// Foundation.h (Mac OS X).
//
// It includes the basic definitions for the test cases below.
// Not including Foundation.h directly makes this test case both svelte and
// portable to non-Mac platforms.
//===----------------------------------------------------------------------===//

typedef unsigned int __darwin_natural_t;
typedef unsigned int UInt32;
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
typedef int kern_return_t;
typedef kern_return_t mach_error_t;
typedef struct objc_selector *SEL;
typedef signed char BOOL;
typedef unsigned long NSUInteger;
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
+ (id)allocWithZone:(NSZone *)zone;
+ (id)alloc;
- (void)dealloc;
@end      extern id NSAllocateObject(Class aClass, NSUInteger extraBytes, NSZone *zone);
typedef struct {
}
NSFastEnumerationState;
@protocol NSFastEnumeration  - (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id *)stackbuf count:(NSUInteger)len;
@end           @class NSString, NSDictionary;
typedef double NSTimeInterval;
@interface NSDate : NSObject <NSCopying, NSCoding>  - (NSTimeInterval)timeIntervalSinceReferenceDate;
@end            typedef unsigned short unichar;
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>    - (NSUInteger)length;
- ( const char *)UTF8String;
- (id)initWithUTF8String:(const char *)nullTerminatedCString;
+ (id)stringWithUTF8String:(const char *)nullTerminatedCString;
@end        @class NSDictionary;
@interface NSDictionary : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>  - (NSUInteger)count;
@end    @interface NSMutableDictionary : NSDictionary  - (void)removeObjectForKey:(id)aKey;
- (void)setObject:(id)anObject forKey:(id)aKey;
@end  @interface NSMutableDictionary (NSMutableDictionaryCreation)  + (id)dictionaryWithCapacity:(NSUInteger)numItems;
@end @class NSString, NSDictionary, NSArray;
typedef mach_port_t io_object_t;
typedef io_object_t io_service_t;
typedef struct __DASession * DASessionRef;
extern DASessionRef DASessionCreate( CFAllocatorRef allocator );
typedef struct __DADisk * DADiskRef;
extern DADiskRef DADiskCreateFromBSDName( CFAllocatorRef allocator, DASessionRef session, const char * name );
extern DADiskRef DADiskCreateFromIOMedia( CFAllocatorRef allocator, DASessionRef session, io_service_t media );
extern CFDictionaryRef DADiskCopyDescription( DADiskRef disk );
extern DADiskRef DADiskCopyWholeDisk( DADiskRef disk );
@interface NSTask : NSObject - (id)init;
@end  extern NSString * const NSTaskDidTerminateNotification;
@interface NSResponder : NSObject <NSCoding> {
struct __vaFlags {
}
_vaFlags;
}
@end    @protocol NSAnimatablePropertyContainer      - (id)animator;
@end  extern NSString *NSAnimationTriggerOrderIn ;
@class NSBitmapImageRep, NSCursor, NSGraphicsContext, NSImage, NSPasteboard, NSScrollView, NSTextInputContext, NSWindow, NSAttributedString;
@interface NSView : NSResponder  <NSAnimatablePropertyContainer>  {
struct __VFlags2 {
}
_vFlags2;
}
@end  @class NSColor, NSFont, NSNotification;
@interface NSTextTab : NSObject <NSCopying, NSCoding> {
}
@end @protocol NSValidatedUserInterfaceItem - (SEL)action;
@end   @protocol NSUserInterfaceValidations - (BOOL)validateUserInterfaceItem:(id <NSValidatedUserInterfaceItem>)anItem;
@end @class NSArray, NSError, NSImage, NSView, NSNotificationCenter, NSURL, NSScreen, NSRunningApplication;
@interface NSApplication : NSResponder <NSUserInterfaceValidations> {
}
@end   enum {
NSTerminateCancel = 0,         NSTerminateNow = 1,         NSTerminateLater = 2 };
typedef NSUInteger NSApplicationTerminateReply;
@protocol NSApplicationDelegate <NSObject> @optional        - (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender;
@end    enum {
NSUserInterfaceLayoutDirectionLeftToRight = 0,     NSUserInterfaceLayoutDirectionRightToLeft = 1 };
@interface NSManagedObject : NSObject {
}
@end enum {
kDAReturnSuccess = 0,     kDAReturnError = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x01,     kDAReturnBusy = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x02,     kDAReturnBadArgument = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x03,     kDAReturnExclusiveAccess = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x04,     kDAReturnNoResources = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x05,     kDAReturnNotFound = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x06,     kDAReturnNotMounted = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x07,     kDAReturnNotPermitted = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x08,     kDAReturnNotPrivileged = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x09,     kDAReturnNotReady = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x0A,     kDAReturnNotWritable = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x0B,     kDAReturnUnsupported = (((0x3e)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x0C };
typedef mach_error_t DAReturn;
typedef const struct __DADissenter * DADissenterRef;
extern DADissenterRef DADissenterCreate( CFAllocatorRef allocator, DAReturn status, CFStringRef string );

 @interface NSArray : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>
 - (NSUInteger)count;
 + (id)array;
 @end
 
@interface NSAutoreleasePool : NSObject {}
+ (void)addObject:(id)anObject;
- (void)addObject:(id)anObject;
- (void)drain;
@end

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
  // When allocations fail, CFDateCreate can return null.
  if (!date) *p = 1; // expected-warning{{null}}
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
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks); // no-warning
  [(id) A autorelease]; // no-warning
}

void f13_autorelease_b() {
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
  [(id) A autorelease];
  [(id) A autorelease]; // expected-warning{{Object sent -autorelease too many times}}
}

CFMutableArrayRef f13_autorelease_c() {
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
  [(id) A autorelease];
  [(id) A autorelease]; 
  return A; // expected-warning{{Object sent -autorelease too many times}}
}

CFMutableArrayRef f13_autorelease_d() {
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
  [(id) A autorelease];
  [(id) A autorelease]; 
  CFMutableArrayRef B = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks); // expected-warning{{Object sent -autorelease too many times}}
  CFRelease(B); // no-warning
}


// This case exercises the logic where the leak site is the same as the allocation site.
void f14_leakimmediately() {
  CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks); // expected-warning{{leak}}
}

// Test that we track an allocated object beyond the point where the *name*
// of the variable storing the reference is no longer live.
void f15() {
  // Create the array.
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
  CFMutableArrayRef *B = &A;
  // At this point, the name 'A' is no longer live.
  CFRelease(*B);  // no-warning
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

// Test return of non-owned objects in contexts where an owned object
// is expected.
@interface TestReturnNotOwnedWhenExpectedOwned
- (NSString*)newString;
@end

@implementation TestReturnNotOwnedWhenExpectedOwned
- (NSString*)newString {
  NSString *s = [NSString stringWithUTF8String:"hello"];
  return s; // expected-warning{{Object with +0 retain counts returned to caller where a +1 (owning) retain count is expected}}
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
  // far as returning 0.  In most error conditions it will throw an exception.
  // If allocation fails it could return 0, but again this
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
  if(!isFoo(kindC[0])) // expected-warning{{null}}
    return;
  if(!isFoo(nameC[0])) // no-warning
    return;

  [kind release];
  [name release]; // expected-warning{{Incorrect decrement of the reference count}}
}

// PR 3677 - 'allocWithZone' should be treated as following the Cocoa naming
//  conventions with respect to 'return'ing ownership.
@interface PR3677: NSObject @end
@implementation PR3677
+ (id)allocWithZone:(NSZone *)inZone {
  return [super allocWithZone:inZone];  // no-warning
}
@end

// PR 3820 - Reason about calls to -dealloc
void pr3820_DeallocInsteadOfRelease(void)
{
  id foo = [[NSString alloc] init]; // no-warning
  [foo dealloc];
  // foo is not leaked, since it has been deallocated.
}

void pr3820_ReleaseAfterDealloc(void)
{
  id foo = [[NSString alloc] init];
  [foo dealloc];
  [foo release];  // expected-warning{{used after it is release}}
  // NSInternalInconsistencyException: message sent to deallocated object
}

void pr3820_DeallocAfterRelease(void)
{
  NSLog(@"\n\n[%s]", __FUNCTION__);
  id foo = [[NSString alloc] init];
  [foo release];
  [foo dealloc]; // expected-warning{{used after it is released}}
  // message sent to released object
}

// From <rdar://problem/6704930>.  The problem here is that 'length' binds to
// '($0 - 1)' after '--length', but SimpleConstraintManager doesn't know how to
// reason about '($0 - 1) > constant'.  As a temporary hack, we drop the value
// of '($0 - 1)' and conjure a new symbol.
void rdar6704930(unsigned char *s, unsigned int length) {
  NSString* name = 0;
  if (s != 0) {
    if (length > 0) {
      while (length > 0) {
        if (*s == ':') {
          ++s;
          --length;
          name = [[NSString alloc] init]; // no-warning
          break;
        }
        ++s;
        --length;
      }
      if ((length == 0) && (name != 0)) {
        [name release];
        name = 0;
      }
      if (length == 0) { // no ':' found -> use it all as name
        name = [[NSString alloc] init]; // no-warning
      }
    }
  }

  if (name != 0) {
    [name release];
  }
}

//===----------------------------------------------------------------------===//
// <rdar://problem/6833332>
// One build of the analyzer accidentally stopped tracking the allocated
// object after the 'retain'.
//===----------------------------------------------------------------------===//

@interface rdar_6833332 : NSObject <NSApplicationDelegate> {
    NSWindow *window;
}
@property (nonatomic, retain) NSWindow *window;
@end

@implementation rdar_6833332
@synthesize window;
- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
 NSMutableDictionary *dict = [[NSMutableDictionary dictionaryWithCapacity:4] retain]; // expected-warning{{leak}}

 [dict setObject:@"foo" forKey:@"bar"];

 NSLog(@"%@", dict);
}
- (void)dealloc {
    [window release];
    [super dealloc];
}
@end

//===----------------------------------------------------------------------===//
// <rdar://problem/6257780> clang checker fails to catch use-after-release
//===----------------------------------------------------------------------===//
                                 
int rdar_6257780_Case1() {
  NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
  NSArray *array = [NSArray array];
  [array release]; // expected-warning{{Incorrect decrement of the reference count of an object is not owned at this point by the caller}}
  [pool drain];
  return 0;
}

//===----------------------------------------------------------------------===//
// <rdar://problem/6866843> Checker should understand new/setObject:/release constructs
//===----------------------------------------------------------------------===//

void rdar_6866843() {
 NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
 NSMutableDictionary* dictionary = [[NSMutableDictionary alloc] init];
 NSArray* array = [[NSArray alloc] init];
 [dictionary setObject:array forKey:@"key"];
 [array release];
 // Using 'array' here should be fine
 NSLog(@"array = %@\n", array); // no-warning
 // Now the array is released
 [dictionary release];
 [pool drain];
}


//===----------------------------------------------------------------------===//
// <rdar://problem/6877235> Classes typedef-ed to CF objects should get the same treatment as CF objects
//===----------------------------------------------------------------------===//

typedef CFTypeRef OtherRef;

@interface RDar6877235 : NSObject {}
- (CFTypeRef)_copyCFTypeRef;
- (OtherRef)_copyOtherRef;
@end

@implementation RDar6877235
- (CFTypeRef)_copyCFTypeRef {
  return [[NSString alloc] init]; // no-warning
}
- (OtherRef)_copyOtherRef {
  return [[NSString alloc] init]; // no-warning
}
@end

//===----------------------------------------------------------------------===//
//<rdar://problem/6320065> false positive - init method returns an object owned by caller
//===----------------------------------------------------------------------===//

@interface RDar6320065 : NSObject {
  NSString *_foo;
}
- (id)initReturningNewClass;
- (id)initReturningNewClassBad;
- (id)initReturningNewClassBad2;
@end

@interface RDar6320065Subclass : RDar6320065
@end

@implementation RDar6320065
- (id)initReturningNewClass {
  [self release];
  self = [[RDar6320065Subclass alloc] init]; // no-warning
  return self;
}
- (id)initReturningNewClassBad {
  [self release];
  [[RDar6320065Subclass alloc] init]; // expected-warning {{leak}}
  return self;
}
- (id)initReturningNewClassBad2 {
  [self release];
  self = [[RDar6320065Subclass alloc] init];
  return [self autorelease]; // expected-warning{{Object with +0 retain counts returned to caller where a +1 (owning) retain count is expected}}
}

@end

@implementation RDar6320065Subclass
@end

int RDar6320065_test() {
  RDar6320065 *test = [[RDar6320065 alloc] init]; // no-warning
  [test release];
  return 0;
}

//===----------------------------------------------------------------------===//
// Tests of ownership attributes.
//===----------------------------------------------------------------------===//

typedef NSString* MyStringTy;

@interface TestOwnershipAttr : NSObject
- (NSString*) returnsAnOwnedString  __attribute__((ns_returns_retained)); // no-warning
- (NSString*) returnsAnOwnedCFString  __attribute__((cf_returns_retained)); // no-warning
- (MyStringTy) returnsAnOwnedTypedString __attribute__((ns_returns_retained)); // no-warning
- (int) returnsAnOwnedInt __attribute__((ns_returns_retained)); // expected-warning{{'ns_returns_retained' attribute only applies to functions or methods that return a pointer or Objective-C object}}
@end

static int ownership_attribute_doesnt_go_here __attribute__((ns_returns_retained)); // expected-warning{{'ns_returns_retained' attribute only applies to function or method types}}

void test_attr_1(TestOwnershipAttr *X) {
  NSString *str = [X returnsAnOwnedString]; // expected-warning{{leak}}
}

void test_attr_1b(TestOwnershipAttr *X) {
  NSString *str = [X returnsAnOwnedCFString]; // expected-warning{{leak}}
}

