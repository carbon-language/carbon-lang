// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -analyze -analyzer-checker=core,osx.coreFoundation.CFRetainRelease,osx.cocoa.ClassRelease,osx.cocoa.RetainCount -analyzer-store=region -analyzer-output=text -fblocks -Wno-objc-root-class %s > %t.objc 2>&1
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -analyze -analyzer-checker=core,osx.coreFoundation.CFRetainRelease,osx.cocoa.ClassRelease,osx.cocoa.RetainCount -analyzer-store=region -analyzer-output=text -fblocks -Wno-objc-root-class -x objective-c++ %s > %t.objcpp 2>&1
// RUN: FileCheck -exact-match -input-file=%t.objc %s
// RUN: FileCheck -exact-match -input-file=%t.objcpp %s

#if __has_feature(attribute_ns_returns_retained)
#define NS_RETURNS_RETAINED __attribute__((ns_returns_retained))
#endif
#if __has_feature(attribute_cf_returns_retained)
#define CF_RETURNS_RETAINED __attribute__((cf_returns_retained))
#endif
#if __has_feature(attribute_ns_returns_not_retained)
#define NS_RETURNS_NOT_RETAINED __attribute__((ns_returns_not_retained))
#endif
#if __has_feature(attribute_cf_returns_not_retained)
#define CF_RETURNS_NOT_RETAINED __attribute__((cf_returns_not_retained))
#endif
#if __has_feature(attribute_ns_consumes_self)
#define NS_CONSUMES_SELF __attribute__((ns_consumes_self))
#endif
#if __has_feature(attribute_ns_consumed)
#define NS_CONSUMED __attribute__((ns_consumed))
#endif
#if __has_feature(attribute_cf_consumed)
#define CF_CONSUMED __attribute__((cf_consumed))
#endif

//===----------------------------------------------------------------------===//
// The following code is reduced using delta-debugging from Mac OS X headers:
//
// #include <Cocoa/Cocoa.h>
// #include <CoreFoundation/CoreFoundation.h>
// #include <DiskArbitration/DiskArbitration.h>
// #include <QuartzCore/QuartzCore.h>
// #include <Quartz/Quartz.h>
// #include <IOKit/IOKitLib.h>
//
// It includes the basic definitions for the test cases below.
//===----------------------------------------------------------------------===//

typedef unsigned int __darwin_natural_t;
typedef unsigned long uintptr_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef unsigned int UInt32;
typedef signed long CFIndex;
typedef CFIndex CFByteOrder;
typedef struct {
    CFIndex location;
    CFIndex length;
} CFRange;
static __inline__ __attribute__((always_inline)) CFRange CFRangeMake(CFIndex loc, CFIndex len) {
    CFRange range;
    range.location = loc;
    range.length = len;
    return range;
}
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
typedef struct {
}
CFDictionaryKeyCallBacks;
extern const CFDictionaryKeyCallBacks kCFTypeDictionaryKeyCallBacks;
typedef struct {
}
CFDictionaryValueCallBacks;
extern const CFDictionaryValueCallBacks kCFTypeDictionaryValueCallBacks;
typedef const struct __CFDictionary * CFDictionaryRef;
typedef struct __CFDictionary * CFMutableDictionaryRef;
extern CFMutableDictionaryRef CFDictionaryCreateMutable(CFAllocatorRef allocator, CFIndex capacity, const CFDictionaryKeyCallBacks *keyCallBacks, const CFDictionaryValueCallBacks *valueCallBacks);
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
enum {
kCFNumberSInt8Type = 1,     kCFNumberSInt16Type = 2,     kCFNumberSInt32Type = 3,     kCFNumberSInt64Type = 4,     kCFNumberFloat32Type = 5,     kCFNumberFloat64Type = 6,      kCFNumberCharType = 7,     kCFNumberShortType = 8,     kCFNumberIntType = 9,     kCFNumberLongType = 10,     kCFNumberLongLongType = 11,     kCFNumberFloatType = 12,     kCFNumberDoubleType = 13,      kCFNumberCFIndexType = 14,      kCFNumberNSIntegerType = 15,     kCFNumberCGFloatType = 16,     kCFNumberMaxType = 16    };
typedef CFIndex CFNumberType;
typedef const struct __CFNumber * CFNumberRef;
extern CFNumberRef CFNumberCreate(CFAllocatorRef allocator, CFNumberType theType, const void *valuePtr);
typedef const struct __CFAttributedString *CFAttributedStringRef;
typedef struct __CFAttributedString *CFMutableAttributedStringRef;
extern CFAttributedStringRef CFAttributedStringCreate(CFAllocatorRef alloc, CFStringRef str, CFDictionaryRef attributes) ;
extern CFMutableAttributedStringRef CFAttributedStringCreateMutableCopy(CFAllocatorRef alloc, CFIndex maxLength, CFAttributedStringRef aStr) ;
extern void CFAttributedStringSetAttribute(CFMutableAttributedStringRef aStr, CFRange range, CFStringRef attrName, CFTypeRef value) ;
typedef signed char BOOL;
typedef unsigned long NSUInteger;
@class NSString, Protocol;
extern void NSLog(NSString *format, ...) __attribute__((format(__NSString__, 1, 2)));
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject
- (BOOL)isEqual:(id)object;
- (id)retain;
- (oneway void)release;
- (id)autorelease;
- (NSString *)description;
- (id)init;
@end
@protocol NSCopying 
- (id)copyWithZone:(NSZone *)zone;
@end
@protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone;
@end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@interface NSObject <NSObject> {}
+ (id)allocWithZone:(NSZone *)zone;
+ (id)alloc;
- (void)dealloc;
@end
@interface NSObject (NSCoderMethods)
- (id)awakeAfterUsingCoder:(NSCoder *)aDecoder;
@end
extern id NSAllocateObject(Class aClass, NSUInteger extraBytes, NSZone *zone);
typedef struct {
}
NSFastEnumerationState;
@protocol NSFastEnumeration 
- (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id *)stackbuf count:(NSUInteger)len;
@end
@class NSString, NSDictionary;
@interface NSValue : NSObject <NSCopying, NSCoding>  - (void)getValue:(void *)value;
@end
@interface NSNumber : NSValue
- (char)charValue;
- (id)initWithInt:(int)value;
+ (NSNumber *)numberWithInt:(int)value;
@end
@class NSString;
@interface NSArray : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>
- (NSUInteger)count;
- (id)initWithObjects:(const id [])objects count:(NSUInteger)cnt;
+ (id)arrayWithObject:(id)anObject;
+ (id)arrayWithObjects:(const id [])objects count:(NSUInteger)cnt;
+ (id)arrayWithObjects:(id)firstObj, ... __attribute__((sentinel(0,1)));
- (id)initWithObjects:(id)firstObj, ... __attribute__((sentinel(0,1)));
- (id)initWithArray:(NSArray *)array;
@end  @interface NSArray (NSArrayCreation)  + (id)array;
@end       @interface NSAutoreleasePool : NSObject {
}
- (void)drain;
@end extern NSString * const NSBundleDidLoadNotification;
typedef double NSTimeInterval;
@interface NSDate : NSObject <NSCopying, NSCoding>  - (NSTimeInterval)timeIntervalSinceReferenceDate;
@end            typedef unsigned short unichar;
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>
- (NSUInteger)length;
- (NSString *)stringByAppendingString:(NSString *)aString;
- ( const char *)UTF8String;
- (id)initWithUTF8String:(const char *)nullTerminatedCString;
+ (id)stringWithUTF8String:(const char *)nullTerminatedCString;
@end        @class NSString, NSURL, NSError;
@interface NSData : NSObject <NSCopying, NSMutableCopying, NSCoding>  - (NSUInteger)length;
+ (id)dataWithBytesNoCopy:(void *)bytes length:(NSUInteger)length;
+ (id)dataWithBytesNoCopy:(void *)bytes length:(NSUInteger)length freeWhenDone:(BOOL)b;
@end   @class NSLocale, NSDate, NSCalendar, NSTimeZone, NSError, NSArray, NSMutableDictionary;
@interface NSDictionary : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>
- (NSUInteger)count;
+ (id)dictionaryWithObjects:(NSArray *)objects forKeys:(NSArray *)keys;
+ (id)dictionaryWithObjects:(const id [])objects forKeys:(const id <NSCopying> [])keys count:(NSUInteger)cnt;
@end
@interface NSMutableDictionary : NSDictionary  - (void)removeObjectForKey:(id)aKey;
- (void)setObject:(id)anObject forKey:(id)aKey;
@end  @interface NSMutableDictionary (NSMutableDictionaryCreation)  + (id)dictionaryWithCapacity:(NSUInteger)numItems;
@end  typedef double CGFloat;
struct CGSize {
};
typedef struct CGSize CGSize;
struct CGRect {
};
typedef struct CGRect CGRect;
typedef mach_port_t io_object_t;
typedef char io_name_t[128];
typedef io_object_t io_iterator_t;
typedef io_object_t io_service_t;
typedef struct IONotificationPort * IONotificationPortRef;
typedef void (*IOServiceMatchingCallback)(  void * refcon,  io_iterator_t iterator );
io_service_t IOServiceGetMatchingService(  mach_port_t masterPort,  CFDictionaryRef matching );
kern_return_t IOServiceGetMatchingServices(  mach_port_t masterPort,  CFDictionaryRef matching,  io_iterator_t * existing );
kern_return_t IOServiceAddNotification(  mach_port_t masterPort,  const io_name_t notificationType,  CFDictionaryRef matching,  mach_port_t wakePort,  uintptr_t reference,  io_iterator_t * notification ) __attribute__((deprecated)); // expected-note {{'IOServiceAddNotification' declared here}}
kern_return_t IOServiceAddMatchingNotification(  IONotificationPortRef notifyPort,  const io_name_t notificationType,  CFDictionaryRef matching,         IOServiceMatchingCallback callback,         void * refCon,  io_iterator_t * notification );
CFMutableDictionaryRef IOServiceMatching(  const char * name );
CFMutableDictionaryRef IOServiceNameMatching(  const char * name );
CFMutableDictionaryRef IOBSDNameMatching(  mach_port_t masterPort,  uint32_t options,  const char * bsdName );
CFMutableDictionaryRef IOOpenFirmwarePathMatching(  mach_port_t masterPort,  uint32_t options,  const char * path );
CFMutableDictionaryRef IORegistryEntryIDMatching(  uint64_t entryID );
typedef struct __DASession * DASessionRef;
extern DASessionRef DASessionCreate( CFAllocatorRef allocator );
typedef struct __DADisk * DADiskRef;
extern DADiskRef DADiskCreateFromBSDName( CFAllocatorRef allocator, DASessionRef session, const char * name );
extern DADiskRef DADiskCreateFromIOMedia( CFAllocatorRef allocator, DASessionRef session, io_service_t media );
extern CFDictionaryRef DADiskCopyDescription( DADiskRef disk );
extern DADiskRef DADiskCopyWholeDisk( DADiskRef disk );
@interface NSTask : NSObject - (id)init;
@end                    typedef struct CGColorSpace *CGColorSpaceRef;
typedef struct CGImage *CGImageRef;
typedef struct CGLayer *CGLayerRef;
@interface NSResponder : NSObject <NSCoding> {
}
@end    @protocol NSAnimatablePropertyContainer      - (id)animator;
@end  extern NSString *NSAnimationTriggerOrderIn ;
@interface NSView : NSResponder  <NSAnimatablePropertyContainer>  {
}
@end @protocol NSValidatedUserInterfaceItem - (SEL)action;
@end   @protocol NSUserInterfaceValidations - (BOOL)validateUserInterfaceItem:(id <NSValidatedUserInterfaceItem>)anItem;
@end  @class NSDate, NSDictionary, NSError, NSException, NSNotification;
@class NSTextField, NSPanel, NSArray, NSWindow, NSImage, NSButton, NSError;
@interface NSApplication : NSResponder <NSUserInterfaceValidations> {
}
- (void)beginSheet:(NSWindow *)sheet modalForWindow:(NSWindow *)docWindow modalDelegate:(id)modalDelegate didEndSelector:(SEL)didEndSelector contextInfo:(void *)contextInfo;
@end   enum {
NSTerminateCancel = 0,         NSTerminateNow = 1,         NSTerminateLater = 2 };
typedef NSUInteger NSApplicationTerminateReply;
@protocol NSApplicationDelegate <NSObject> @optional        - (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender;
@end  @class NSAttributedString, NSEvent, NSFont, NSFormatter, NSImage, NSMenu, NSText, NSView, NSTextView;
@interface NSCell : NSObject <NSCopying, NSCoding> {
}
@end 
typedef struct {
}
CVTimeStamp;
@interface CIImage : NSObject <NSCoding, NSCopying> {
}
typedef int CIFormat;
@end  enum {
kDAReturnSuccess = 0,     kDAReturnError = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x01,     kDAReturnBusy = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x02,     kDAReturnBadArgument = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x03,     kDAReturnExclusiveAccess = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x04,     kDAReturnNoResources = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x05,     kDAReturnNotFound = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x06,     kDAReturnNotMounted = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x07,     kDAReturnNotPermitted = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x08,     kDAReturnNotPrivileged = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x09,     kDAReturnNotReady = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x0A,     kDAReturnNotWritable = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x0B,     kDAReturnUnsupported = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x0C };
typedef mach_error_t DAReturn;
typedef const struct __DADissenter * DADissenterRef;
extern DADissenterRef DADissenterCreate( CFAllocatorRef allocator, DAReturn status, CFStringRef string );
@interface CIContext: NSObject {
}
- (CGImageRef)createCGImage:(CIImage *)im fromRect:(CGRect)r;
- (CGImageRef)createCGImage:(CIImage *)im fromRect:(CGRect)r     format:(CIFormat)f colorSpace:(CGColorSpaceRef)cs;
- (CGLayerRef)createCGLayerWithSize:(CGSize)size info:(CFDictionaryRef)d;
@end extern NSString* const QCRendererEventKey;
@protocol QCCompositionRenderer - (NSDictionary*) attributes;
@end   @interface QCRenderer : NSObject <QCCompositionRenderer> {
}
- (id) createSnapshotImageOfType:(NSString*)type;
@end  extern NSString* const QCViewDidStartRenderingNotification;
@interface QCView : NSView <QCCompositionRenderer> {
}
- (id) createSnapshotImageOfType:(NSString*)type;
@end    enum {
ICEXIFOrientation1 = 1,     ICEXIFOrientation2 = 2,     ICEXIFOrientation3 = 3,     ICEXIFOrientation4 = 4,     ICEXIFOrientation5 = 5,     ICEXIFOrientation6 = 6,     ICEXIFOrientation7 = 7,     ICEXIFOrientation8 = 8, };
@class ICDevice;
@protocol ICDeviceDelegate <NSObject>  @required      - (void)didRemoveDevice:(ICDevice*)device;
@end extern NSString *const ICScannerStatusWarmingUp;
@class ICScannerDevice;
@protocol ICScannerDeviceDelegate <ICDeviceDelegate>  @optional       - (void)scannerDeviceDidBecomeAvailable:(ICScannerDevice*)scanner;
@end

typedef long unsigned int __darwin_size_t;
typedef __darwin_size_t size_t;
typedef unsigned long CFTypeID;
struct CGPoint {
  CGFloat x;
  CGFloat y;
};
typedef struct CGPoint CGPoint;
typedef struct CGGradient *CGGradientRef;
typedef uint32_t CGGradientDrawingOptions;
extern CFTypeID CGGradientGetTypeID(void);
extern CGGradientRef CGGradientCreateWithColorComponents(CGColorSpaceRef
  space, const CGFloat components[], const CGFloat locations[], size_t count);
extern CGGradientRef CGGradientCreateWithColors(CGColorSpaceRef space,
  CFArrayRef colors, const CGFloat locations[]);
extern CGGradientRef CGGradientRetain(CGGradientRef gradient);
extern void CGGradientRelease(CGGradientRef gradient);
typedef struct CGContext *CGContextRef;
extern void CGContextDrawLinearGradient(CGContextRef context,
    CGGradientRef gradient, CGPoint startPoint, CGPoint endPoint,
    CGGradientDrawingOptions options);
extern CGColorSpaceRef CGColorSpaceCreateDeviceRGB(void);

@interface NSMutableArray : NSObject
- (void)addObject:(id)object;
+ (id)array;
@end

// This is how NSMakeCollectable is declared in the OS X 10.8 headers.
id NSMakeCollectable(CFTypeRef __attribute__((cf_consumed))) __attribute__((ns_returns_retained));

typedef const struct __CFUUID * CFUUIDRef;

extern
void *CFPlugInInstanceCreate(CFAllocatorRef allocator, CFUUIDRef factoryUUID, CFUUIDRef typeUUID);

//===----------------------------------------------------------------------===//
// Test cases.
//===----------------------------------------------------------------------===//

CFAbsoluteTime f1() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t);
  CFRetain(date);
  CFRelease(date);
  CFDateGetAbsoluteTime(date);
  CFRelease(date);
  t = CFDateGetAbsoluteTime(date);
  return t;
}

CFAbsoluteTime f2() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t);  
  [((NSDate*) date) retain];
  CFRelease(date);
  CFDateGetAbsoluteTime(date);
  [((NSDate*) date) release];
  t = CFDateGetAbsoluteTime(date);
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
  CFDateGetAbsoluteTime(date);
  global_x = (NSDate*) date;  
  [((NSDate*) date) release];
  t = CFDateGetAbsoluteTime(date);
  return t;
}

//---------------------------------------------------------------------------
// Test case 'f4' differs for region store and basic store.  See
// retain-release-region-store.m and retain-release-basic-store.m.
//---------------------------------------------------------------------------

// Test a leak.

CFAbsoluteTime f5(int x) {  
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t);
  
  if (x)
    CFRelease(date);
  
  return t;
}

// Test a leak involving the return.

CFDateRef f6(int x) {  
  CFDateRef date = CFDateCreate(0, CFAbsoluteTimeGetCurrent());
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
  CFDateRef date = MyDateCreate();
  CFRetain(date);  
  return date;
}

__attribute__((cf_returns_retained)) CFDateRef f9() {
  CFDateRef date = CFDateCreate(0, CFAbsoluteTimeGetCurrent());
  int *p = 0;
  // When allocations fail, CFDateCreate can return null.
  if (!date) *p = 1;
  return date;
}

// Handle DiskArbitration API:
//
// http://developer.apple.com/DOCUMENTATION/DARWIN/Reference/DiscArbitrationFramework/
//
void f10(io_service_t media, DADiskRef d, CFStringRef s) {
  DADiskRef disk = DADiskCreateFromBSDName(kCFAllocatorDefault, 0, "hello");
  if (disk) NSLog(@"ok");
  
  disk = DADiskCreateFromIOMedia(kCFAllocatorDefault, 0, media);
  if (disk) NSLog(@"ok");

  CFDictionaryRef dict = DADiskCopyDescription(d);
  if (dict) NSLog(@"ok"); 
  
  disk = DADiskCopyWholeDisk(d);
  if (disk) NSLog(@"ok");
    
  DADissenterRef dissenter = DADissenterCreate(kCFAllocatorDefault,
                                                kDAReturnSuccess, s);
  if (dissenter) NSLog(@"ok");
  
  DASessionRef session = DASessionCreate(kCFAllocatorDefault);
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
  CFRelease(s1);
  
  // Get the string.  We don't own it.
  s1 = (CFStringRef) CFArrayGetValueAtIndex(A, 0);
  
  // Release the array.
  CFRelease(A);
  
  // Release the string.  This is a bug.
  CFRelease(s1);
}

// PR 3337: Handle functions declared using typedefs.
typedef CFTypeRef CREATEFUN();
CREATEFUN MyCreateFun;

void f12() {
  CFTypeRef o = MyCreateFun();
}

void f13_autorelease() {
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
  [(id) A autorelease];
}

void f13_autorelease_b() {
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
  [(id) A autorelease];
  [(id) A autorelease];
}

CFMutableArrayRef f13_autorelease_c() {
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
  [(id) A autorelease];
  [(id) A autorelease]; 
  return A;
}

CFMutableArrayRef f13_autorelease_d() {
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
  [(id) A autorelease];
  [(id) A autorelease]; 
  CFMutableArrayRef B = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
  CFRelease(B);
  while (1) {}
}


// This case exercises the logic where the leak site is the same as the allocation site.
void f14_leakimmediately() {
  CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
}

// Test that we track an allocated object beyond the point where the *name*
// of the variable storing the reference is no longer live.
void f15() {
  // Create the array.
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
  CFMutableArrayRef *B = &A;
  // At this point, the name 'A' is no longer live.
  CFRelease(*B);
}

// Test when we pass NULL to CFRetain/CFRelease.
void f16(int x, CFTypeRef p) {
  if (p)
    return;

  if (x) {
    CFRelease(p);
  }
  else {
    CFRetain(p);
  }
}

// Test that an object is non-null after being CFRetained/CFReleased.
void f17(int x, CFTypeRef p) {
  if (x) {
    CFRelease(p);
    if (!p)
      CFRelease(0);
  }
  else {
    CFRetain(p);
    if (!p)
      CFRetain(0);
  }
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
  myObj = (id) CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
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
  return s;
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
  NSString *kind = [[NSString alloc] initWithUTF8String:inkind];
  
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
  if(!isFoo(kindC[0]))
    return;
  if(!isFoo(nameC[0]))
    return;

  [kind release];
  [name release];
}

// PR 3677 - 'allocWithZone' should be treated as following the Cocoa naming
//  conventions with respect to 'return'ing ownership.
@interface PR3677: NSObject @end
@implementation PR3677
+ (id)allocWithZone:(NSZone *)inZone {
  return [super allocWithZone:inZone];
}
@end

// PR 3820 - Reason about calls to -dealloc
void pr3820_DeallocInsteadOfRelease(void)
{
  id foo = [[NSString alloc] init];
  [foo dealloc];
  // foo is not leaked, since it has been deallocated.
}

void pr3820_ReleaseAfterDealloc(void)
{
  id foo = [[NSString alloc] init];
  [foo dealloc];
  [foo release];
  // NSInternalInconsistencyException: message sent to deallocated object
}

void pr3820_DeallocAfterRelease(void)
{
  NSLog(@"\n\n[%s]", __FUNCTION__);
  id foo = [[NSString alloc] init];
  [foo release];
  [foo dealloc];
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
          name = [[NSString alloc] init];
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
        name = [[NSString alloc] init];
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
 NSMutableDictionary *dict = [[NSMutableDictionary dictionaryWithCapacity:4] retain];

 [dict setObject:@"foo" forKey:@"bar"];

 NSLog(@"%@", dict);
}
- (void)dealloc {
    [window release];
    [super dealloc];
}

- (void)radar10102244 {
 NSMutableDictionary *dict = [[NSMutableDictionary dictionaryWithCapacity:4] retain];
 if (window) 
   NSLog(@"%@", window);    
}
@end

//===----------------------------------------------------------------------===//
// <rdar://problem/6257780> clang checker fails to catch use-after-release
//===----------------------------------------------------------------------===//

int rdar_6257780_Case1() {
  NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
  NSArray *array = [NSArray array];
  [array release];
  [pool drain];
  return 0;
}

//===----------------------------------------------------------------------===//
// <rdar://problem/10640253> Analyzer is confused about NSAutoreleasePool -allocWithZone:.
//===----------------------------------------------------------------------===//

void rdar_10640253_autorelease_allocWithZone() {
    NSAutoreleasePool *pool = [[NSAutoreleasePool allocWithZone:(NSZone*)0] init];
    (void) pool;
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
 NSLog(@"array = %@\n", array);
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
  return [[NSString alloc] init];
}
- (OtherRef)_copyOtherRef {
  return [[NSString alloc] init];
}
@end

//===----------------------------------------------------------------------===//
// <rdar://problem/6320065> false positive - init method returns an object
// owned by caller
//===----------------------------------------------------------------------===//

@interface RDar6320065 : NSObject {
  NSString *_foo;
}
- (id)initReturningNewClass;
- (id)_initReturningNewClassBad;
- (id)initReturningNewClassBad2;
@end

@interface RDar6320065Subclass : RDar6320065
@end

@implementation RDar6320065
- (id)initReturningNewClass {
  [self release];
  self = [[RDar6320065Subclass alloc] init];
  return self;
}
- (id)_initReturningNewClassBad {
  [self release];
  [[RDar6320065Subclass alloc] init];
  return self;
}
- (id)initReturningNewClassBad2 {
  [self release];
  self = [[RDar6320065Subclass alloc] init];
  return [self autorelease];
}

@end

@implementation RDar6320065Subclass
@end

int RDar6320065_test() {
  RDar6320065 *test = [[RDar6320065 alloc] init];
  [test release];
  return 0;
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7129086> -awakeAfterUsingCoder: returns an owned object 
//  and claims the receiver
//===----------------------------------------------------------------------===//

@interface RDar7129086 : NSObject {} @end
@implementation RDar7129086
- (id)awakeAfterUsingCoder:(NSCoder *)aDecoder {
  [self release];
  return [NSString alloc];
}
@end

//===----------------------------------------------------------------------===//
// <rdar://problem/6859457> [NSData dataWithBytesNoCopy] does not return a
//  retained object
//===----------------------------------------------------------------------===//

@interface RDar6859457 : NSObject {}
- (NSString*) NoCopyString;
- (NSString*) noCopyString;
@end

@implementation RDar6859457 
- (NSString*) NoCopyString { return [[NSString alloc] init]; }
- (NSString*) noCopyString { return [[NSString alloc] init]; }
@end

void test_RDar6859457(RDar6859457 *x, void *bytes, NSUInteger dataLength) {
  [x NoCopyString];
  [x noCopyString];
  [NSData dataWithBytesNoCopy:bytes length:dataLength];
  [NSData dataWithBytesNoCopy:bytes length:dataLength freeWhenDone:1];
}

//===----------------------------------------------------------------------===//
// PR 4230 - an autorelease pool is not necessarily leaked during a premature
//  return
//===----------------------------------------------------------------------===//

static void PR4230(void)
{
  NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
  NSString *object = [[[NSString alloc] init] autorelease];
  return;
}

//===----------------------------------------------------------------------===//
// Method name that has a null IdentifierInfo* for its first selector slot.
// This test just makes sure that we handle it.
//===----------------------------------------------------------------------===//

@interface TestNullIdentifier
@end

@implementation TestNullIdentifier
+ (id):(int)x, ... {
  return [[NSString alloc] init];
}
@end

//===----------------------------------------------------------------------===//
// <rdar://problem/6893565> don't flag leaks for return types that cannot be 
//                          determined to be CF types
//===----------------------------------------------------------------------===//

// We don't know if 'struct s6893565' represents a Core Foundation type, so
// we shouldn't emit an error here.
typedef struct s6893565* TD6893565;

@interface RDar6893565 {}
-(TD6893565)newThing;
@end

@implementation RDar6893565
-(TD6893565)newThing {  
  return (TD6893565) [[NSString alloc] init];
}
@end

//===----------------------------------------------------------------------===//
// <rdar://problem/6902710> clang: false positives w/QC and CoreImage methods
//===----------------------------------------------------------------------===//

void rdar6902710(QCView *view, QCRenderer *renderer, CIContext *context,
                 NSString *str, CIImage *img, CGRect rect,
                 CIFormat form, CGColorSpaceRef cs) {
  [view createSnapshotImageOfType:str];
  [renderer createSnapshotImageOfType:str];
  [context createCGImage:img fromRect:rect];
  [context createCGImage:img fromRect:rect format:form colorSpace:cs];
}

//===----------------------------------------------------------------------===//
// <rdar://problem/6945561> -[CIContext createCGLayerWithSize:info:]
//                           misinterpreted by clang scan-build
//===----------------------------------------------------------------------===//

void rdar6945561(CIContext *context, CGSize size, CFDictionaryRef d) {
  [context createCGLayerWithSize:size info:d];
}

//===----------------------------------------------------------------------===//
// <rdar://problem/6961230> add knowledge of IOKit functions to retain/release 
//                          checker
//===----------------------------------------------------------------------===//

void IOBSDNameMatching_wrapper(mach_port_t masterPort, uint32_t options,  const char * bsdName) {  
  IOBSDNameMatching(masterPort, options, bsdName);
}

void IOServiceMatching_wrapper(const char * name) {
  IOServiceMatching(name);
}

void IOServiceNameMatching_wrapper(const char * name) {
  IOServiceNameMatching(name);
}

CF_RETURNS_RETAINED CFDictionaryRef CreateDict();

void IOServiceAddNotification_wrapper(mach_port_t masterPort, const io_name_t notificationType,
  mach_port_t wakePort, uintptr_t reference, io_iterator_t * notification ) {

  CFDictionaryRef matching = CreateDict();
  CFRelease(matching);
  IOServiceAddNotification(masterPort, notificationType, matching,
                           wakePort, reference, notification);
}

void IORegistryEntryIDMatching_wrapper(uint64_t entryID ) {
  IORegistryEntryIDMatching(entryID);
}

void IOOpenFirmwarePathMatching_wrapper(mach_port_t masterPort, uint32_t options,
                                        const char * path) {
  IOOpenFirmwarePathMatching(masterPort, options, path);
}

void IOServiceGetMatchingService_wrapper(mach_port_t masterPort) {
  CFDictionaryRef matching = CreateDict();
  IOServiceGetMatchingService(masterPort, matching);
  CFRelease(matching);
}

void IOServiceGetMatchingServices_wrapper(mach_port_t masterPort, io_iterator_t *existing) {
  CFDictionaryRef matching = CreateDict();
  IOServiceGetMatchingServices(masterPort, matching, existing);
  CFRelease(matching);
}

void IOServiceAddMatchingNotification_wrapper(IONotificationPortRef notifyPort, const io_name_t notificationType, 
  IOServiceMatchingCallback callback, void * refCon, io_iterator_t * notification) {
    
  CFDictionaryRef matching = CreateDict();
  IOServiceAddMatchingNotification(notifyPort, notificationType, matching, callback, refCon, notification);
  CFRelease(matching);
}

//===----------------------------------------------------------------------===//
// Test of handling objects whose references "escape" to containers.
//===----------------------------------------------------------------------===//

void CFDictionaryAddValue(CFMutableDictionaryRef, void *, void *);

// <rdar://problem/6539791>
void rdar_6539791(CFMutableDictionaryRef y, void* key, void* val_key) {
  CFMutableDictionaryRef x = CFDictionaryCreateMutable(kCFAllocatorDefault, 1, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  CFDictionaryAddValue(y, key, x);
  CFRelease(x); // the dictionary keeps a reference, so the object isn't deallocated yet
  signed z = 1;
  CFNumberRef value = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &z);
  if (value) {
    CFDictionaryAddValue(x, val_key, (void*)value);
    CFRelease(value);
    CFDictionaryAddValue(y, val_key, (void*)value);
  }
}

// <rdar://problem/6560661>
// Same issue, except with "AppendValue" functions.
void rdar_6560661(CFMutableArrayRef x) {
  signed z = 1;
  CFNumberRef value = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &z);
  // CFArrayAppendValue keeps a reference to value.
  CFArrayAppendValue(x, value);
  CFRelease(value);
  CFRetain(value);
  CFRelease(value);
}

// <rdar://problem/7152619>
// Same issue, excwept with "CFAttributeStringSetAttribute".
void rdar_7152619(CFStringRef str) {
  CFAttributedStringRef string = CFAttributedStringCreate(kCFAllocatorDefault, str, 0);
  CFMutableAttributedStringRef attrString = CFAttributedStringCreateMutableCopy(kCFAllocatorDefault, 100, string);
  CFRelease(string);
  NSNumber *number = [[NSNumber alloc] initWithInt:5];
  CFAttributedStringSetAttribute(attrString, CFRangeMake(0, 1), str, number);
  [number release];
  [number retain];
  CFRelease(attrString);  
}

//===----------------------------------------------------------------------===//
// Test of handling CGGradientXXX functions.
//===----------------------------------------------------------------------===//

void rdar_7184450(CGContextRef myContext, CGFloat x, CGPoint myStartPoint,
                  CGPoint myEndPoint) {
  size_t num_locations = 6;
  CGFloat locations[6] = { 0.0, 0.265, 0.28, 0.31, 0.36, 1.0 };
  CGFloat components[28] = { 239.0/256.0, 167.0/256.0, 170.0/256.0,
     x,  // Start color
    207.0/255.0, 39.0/255.0, 39.0/255.0, x,
    147.0/255.0, 21.0/255.0, 22.0/255.0, x,
    175.0/255.0, 175.0/255.0, 175.0/255.0, x,
    255.0/255.0,255.0/255.0, 255.0/255.0, x,
    255.0/255.0,255.0/255.0, 255.0/255.0, x
  }; // End color
  
  CGGradientRef myGradient =
    CGGradientCreateWithColorComponents(CGColorSpaceCreateDeviceRGB(),
      components, locations, num_locations);

  CGContextDrawLinearGradient(myContext, myGradient, myStartPoint, myEndPoint,
                              0);
  CGGradientRelease(myGradient);
}

void rdar_7184450_pos(CGContextRef myContext, CGFloat x, CGPoint myStartPoint,
                  CGPoint myEndPoint) {
  size_t num_locations = 6;
  CGFloat locations[6] = { 0.0, 0.265, 0.28, 0.31, 0.36, 1.0 };
  CGFloat components[28] = { 239.0/256.0, 167.0/256.0, 170.0/256.0,
     x,  // Start color
    207.0/255.0, 39.0/255.0, 39.0/255.0, x,
    147.0/255.0, 21.0/255.0, 22.0/255.0, x,
    175.0/255.0, 175.0/255.0, 175.0/255.0, x,
    255.0/255.0,255.0/255.0, 255.0/255.0, x,
    255.0/255.0,255.0/255.0, 255.0/255.0, x
  }; // End color
  
  CGGradientRef myGradient =
   CGGradientCreateWithColorComponents(CGColorSpaceCreateDeviceRGB(), components, locations, num_locations);

  CGContextDrawLinearGradient(myContext, myGradient, myStartPoint, myEndPoint,
                              0);
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7299394> clang false positive: retained instance passed to
//                          thread in pthread_create marked as leak
//
// Until we have full IPA, the analyzer should stop tracking the reference
// count of objects passed to pthread_create.
//
//===----------------------------------------------------------------------===//

struct _opaque_pthread_t {};
struct _opaque_pthread_attr_t {};
typedef struct _opaque_pthread_t *__darwin_pthread_t;
typedef struct _opaque_pthread_attr_t __darwin_pthread_attr_t;
typedef __darwin_pthread_t pthread_t;
typedef __darwin_pthread_attr_t pthread_attr_t;
typedef unsigned long __darwin_pthread_key_t;
typedef __darwin_pthread_key_t pthread_key_t;

int pthread_create(pthread_t *, const pthread_attr_t *,
                   void *(*)(void *), void *);

int pthread_setspecific(pthread_key_t key, const void *value);

void *rdar_7299394_start_routine(void *p) {
  [((id) p) release];
  return 0;
}
void rdar_7299394(pthread_attr_t *attr, pthread_t *thread, void *args) {
  NSNumber *number = [[NSNumber alloc] initWithInt:5];
  pthread_create(thread, attr, rdar_7299394_start_routine, number);
}
void rdar_7299394_positive(pthread_attr_t *attr, pthread_t *thread) {
  NSNumber *number = [[NSNumber alloc] initWithInt:5];
}

//===----------------------------------------------------------------------===//
// <rdar://problem/11282706> false positive with not understanding thread
// local storage
//===----------------------------------------------------------------------===//

void rdar11282706(pthread_key_t key) {
  NSNumber *number = [[NSNumber alloc] initWithInt:5];
  pthread_setspecific(key, (void*) number);
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7283567> False leak associated with call to 
//                          CVPixelBufferCreateWithBytes ()
//
// According to the Core Video Reference (ADC), CVPixelBufferCreateWithBytes and
// CVPixelBufferCreateWithPlanarBytes can release (via a callback) the
// pixel buffer object.  These test cases show how the analyzer stops tracking
// the reference count for the objects passed for this argument.  This
// could be made smarter.
//===----------------------------------------------------------------------===//

typedef int int32_t;
typedef UInt32 FourCharCode;
typedef FourCharCode OSType;
typedef uint64_t CVOptionFlags;
typedef int32_t CVReturn;
typedef struct __CVBuffer *CVBufferRef;
typedef CVBufferRef CVImageBufferRef;
typedef CVImageBufferRef CVPixelBufferRef;
typedef void (*CVPixelBufferReleaseBytesCallback)( void *releaseRefCon, const void *baseAddress );

extern CVReturn CVPixelBufferCreateWithBytes(CFAllocatorRef allocator,
            size_t width,
            size_t height,
            OSType pixelFormatType,
            void *baseAddress,
            size_t bytesPerRow,
            CVPixelBufferReleaseBytesCallback releaseCallback,
            void *releaseRefCon,
            CFDictionaryRef pixelBufferAttributes,
                   CVPixelBufferRef *pixelBufferOut) ;

typedef void (*CVPixelBufferReleasePlanarBytesCallback)( void *releaseRefCon, const void *dataPtr, size_t dataSize, size_t numberOfPlanes, const void *planeAddresses[] );

extern CVReturn CVPixelBufferCreateWithPlanarBytes(CFAllocatorRef allocator,
        size_t width,
        size_t height,
        OSType pixelFormatType,
        void *dataPtr,
        size_t dataSize,
        size_t numberOfPlanes,
        void *planeBaseAddress[],
        size_t planeWidth[],
        size_t planeHeight[],
        size_t planeBytesPerRow[],
        CVPixelBufferReleasePlanarBytesCallback releaseCallback,
        void *releaseRefCon,
        CFDictionaryRef pixelBufferAttributes,
        CVPixelBufferRef *pixelBufferOut) ;

extern CVReturn CVPixelBufferCreateWithBytes(CFAllocatorRef allocator,
            size_t width,
            size_t height,
            OSType pixelFormatType,
            void *baseAddress,
            size_t bytesPerRow,
            CVPixelBufferReleaseBytesCallback releaseCallback,
            void *releaseRefCon,
            CFDictionaryRef pixelBufferAttributes,
                   CVPixelBufferRef *pixelBufferOut) ;

CVReturn rdar_7283567(CFAllocatorRef allocator, size_t width, size_t height,
                      OSType pixelFormatType, void *baseAddress,
                      size_t bytesPerRow,
                      CVPixelBufferReleaseBytesCallback releaseCallback,
                      CFDictionaryRef pixelBufferAttributes,
                      CVPixelBufferRef *pixelBufferOut) {

  // For the allocated object, it doesn't really matter what type it is
  // for the purpose of this test.  All we want to show is that
  // this is freed later by the callback.
  NSNumber *number = [[NSNumber alloc] initWithInt:5];
  
  return CVPixelBufferCreateWithBytes(allocator, width, height, pixelFormatType,
                                baseAddress, bytesPerRow, releaseCallback,
                                number, // potentially released by callback
                                pixelBufferAttributes, pixelBufferOut) ;
}

CVReturn rdar_7283567_2(CFAllocatorRef allocator, size_t width, size_t height,
        OSType pixelFormatType, void *dataPtr, size_t dataSize,
        size_t numberOfPlanes, void *planeBaseAddress[],
        size_t planeWidth[], size_t planeHeight[], size_t planeBytesPerRow[],
        CVPixelBufferReleasePlanarBytesCallback releaseCallback,
        CFDictionaryRef pixelBufferAttributes,
        CVPixelBufferRef *pixelBufferOut) {
    
    // For the allocated object, it doesn't really matter what type it is
    // for the purpose of this test.  All we want to show is that
    // this is freed later by the callback.
    NSNumber *number = [[NSNumber alloc] initWithInt:5];

    return CVPixelBufferCreateWithPlanarBytes(allocator,
              width, height, pixelFormatType, dataPtr, dataSize,
              numberOfPlanes, planeBaseAddress, planeWidth,
              planeHeight, planeBytesPerRow, releaseCallback,
              number, // potentially released by callback
              pixelBufferAttributes, pixelBufferOut) ;
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7358899> False leak associated with 
//  CGBitmapContextCreateWithData
//===----------------------------------------------------------------------===//
typedef uint32_t CGBitmapInfo;
typedef void (*CGBitmapContextReleaseDataCallback)(void *releaseInfo, void *data);
    
CGContextRef CGBitmapContextCreateWithData(void *data,
    size_t width, size_t height, size_t bitsPerComponent,
    size_t bytesPerRow, CGColorSpaceRef space, CGBitmapInfo bitmapInfo,
    CGBitmapContextReleaseDataCallback releaseCallback, void *releaseInfo);

void rdar_7358899(void *data,
      size_t width, size_t height, size_t bitsPerComponent,
      size_t bytesPerRow, CGColorSpaceRef space, CGBitmapInfo bitmapInfo,
      CGBitmapContextReleaseDataCallback releaseCallback) {

    // For the allocated object, it doesn't really matter what type it is
    // for the purpose of this test.  All we want to show is that
    // this is freed later by the callback.
    NSNumber *number = [[NSNumber alloc] initWithInt:5];

  CGBitmapContextCreateWithData(data, width, height, bitsPerComponent,
    bytesPerRow, space, bitmapInfo, releaseCallback, number);
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7265711> allow 'new', 'copy', 'alloc', 'init' prefix to
//  start before '_' when determining Cocoa fundamental rule
//
// Previously the retain/release checker just skipped prefixes before the
// first '_' entirely.  Now the checker honors the prefix if it results in a
// recognizable naming convention (e.g., 'new', 'init').
//===----------------------------------------------------------------------===//

@interface RDar7265711 {}
- (id) new_stuff;
@end

void rdar7265711_a(RDar7265711 *x) {
  id y = [x new_stuff];
}

void rdar7265711_b(RDar7265711 *x) {
  id y = [x new_stuff];
  [y release];
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7306898> clang thinks [NSCursor dragCopyCursor] returns a
//                          retained reference
//===----------------------------------------------------------------------===//

@interface NSCursor : NSObject
+ (NSCursor *)dragCopyCursor;
@end

void rdar7306898(void) {
  // 'dragCopyCursor' does not follow Cocoa's fundamental rule.  It is a noun, not an sentence
  // implying a 'copy' of something.
  NSCursor *c =  [NSCursor dragCopyCursor];
  NSNumber *number = [[NSNumber alloc] initWithInt:5];
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7252064> sending 'release', 'retain', etc. to a Class
// directly is not likely what the user intended
//===----------------------------------------------------------------------===//

@interface RDar7252064 : NSObject @end
void rdar7252064(void) {
  [RDar7252064 release];
  [RDar7252064 retain];
  [RDar7252064 autorelease];
  [NSAutoreleasePool drain];
}

//===----------------------------------------------------------------------===//
// Tests of ownership attributes.
//===----------------------------------------------------------------------===//

typedef NSString* MyStringTy;

@protocol FooP;

@interface TestOwnershipAttr : NSObject
- (NSString*) returnsAnOwnedString  NS_RETURNS_RETAINED;
- (NSString*) returnsAnOwnedCFString  CF_RETURNS_RETAINED;
- (MyStringTy) returnsAnOwnedTypedString NS_RETURNS_RETAINED;
- (NSString*) newString NS_RETURNS_NOT_RETAINED;
- (NSString*) newStringNoAttr;
- (int) returnsAnOwnedInt NS_RETURNS_RETAINED;
- (id) pseudoInit NS_CONSUMES_SELF NS_RETURNS_RETAINED;
+ (void) consume:(id) NS_CONSUMED x;
+ (void) consume2:(id) CF_CONSUMED x;
@end

static int ownership_attribute_doesnt_go_here NS_RETURNS_RETAINED;

void test_attr_1(TestOwnershipAttr *X) {
  NSString *str = [X returnsAnOwnedString];
}

void test_attr_1b(TestOwnershipAttr *X) {
  NSString *str = [X returnsAnOwnedCFString];
}

void test_attr1c(TestOwnershipAttr *X) {
  NSString *str = [X newString];
  NSString *str2 = [X newStringNoAttr];
}

void testattr2_a() {
  TestOwnershipAttr *x = [TestOwnershipAttr alloc];
}

void testattr2_b() {
  TestOwnershipAttr *x = [[TestOwnershipAttr alloc] pseudoInit];
}

void testattr2_b_11358224_self_assign_looses_the_leak() {
  TestOwnershipAttr *x = [[TestOwnershipAttr alloc] pseudoInit];// expected-warning{{leak}}
  x = x;
}

void testattr2_c() {
  TestOwnershipAttr *x = [[TestOwnershipAttr alloc] pseudoInit];
  [x release];
}

void testattr3() {
  TestOwnershipAttr *x = [TestOwnershipAttr alloc];
  [TestOwnershipAttr consume:x];
  TestOwnershipAttr *y = [TestOwnershipAttr alloc];
  [TestOwnershipAttr consume2:y];
}

void consume_ns(id NS_CONSUMED x);
void consume_cf(id CF_CONSUMED x);

void testattr4() {
  TestOwnershipAttr *x = [TestOwnershipAttr alloc];
  consume_ns(x);
  TestOwnershipAttr *y = [TestOwnershipAttr alloc];
  consume_cf(y);
}

@interface TestOwnershipAttr2 : NSObject
- (NSString*) newString NS_RETURNS_NOT_RETAINED;
@end

@implementation TestOwnershipAttr2
- (NSString*) newString {
  return [NSString alloc];
}
@end

@interface MyClassTestCFAttr : NSObject {}
- (NSDate*) returnsCFRetained CF_RETURNS_RETAINED;
- (CFDateRef) returnsCFRetainedAsCF CF_RETURNS_RETAINED;
- (CFDateRef) newCFRetainedAsCF CF_RETURNS_NOT_RETAINED;
- (CFDateRef) newCFRetainedAsCFNoAttr;
- (NSDate*) alsoReturnsRetained;
- (CFDateRef) alsoReturnsRetainedAsCF;
- (NSDate*) returnsNSRetained NS_RETURNS_RETAINED;
@end

CF_RETURNS_RETAINED
CFDateRef returnsRetainedCFDate()  {
  return CFDateCreate(0, CFAbsoluteTimeGetCurrent());
}

@implementation MyClassTestCFAttr
- (NSDate*) returnsCFRetained {
  return (NSDate*) returnsRetainedCFDate(); // No leak.
}

- (CFDateRef) returnsCFRetainedAsCF {
  return returnsRetainedCFDate(); // No leak.
}

- (CFDateRef) newCFRetainedAsCF {
  return (CFDateRef)[(id)[self returnsCFRetainedAsCF] autorelease];
}

- (CFDateRef) newCFRetainedAsCFNoAttr {
  return (CFDateRef)[(id)[self returnsCFRetainedAsCF] autorelease];
}

- (NSDate*) alsoReturnsRetained {
  return (NSDate*) returnsRetainedCFDate();
}

- (CFDateRef) alsoReturnsRetainedAsCF {
  return returnsRetainedCFDate();
}


- (NSDate*) returnsNSRetained {
  return (NSDate*) returnsRetainedCFDate();
}
@end

//===----------------------------------------------------------------------===//
// Test that leaks post-dominated by "panic" functions are not reported.
//
// <rdar://problem/5905851> do not report a leak when post-dominated by a call
// to a noreturn or panic function
//===----------------------------------------------------------------------===//

void panic() __attribute__((noreturn));
void panic_not_in_hardcoded_list() __attribute__((noreturn));

void test_panic_negative() {
  signed z = 1;
  CFNumberRef value = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &z);
}

void test_panic_positive() {
  signed z = 1;
  CFNumberRef value = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &z);
  panic();
}

void test_panic_neg_2(int x) {
  signed z = 1;
  CFNumberRef value = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &z);
  if (x)
    panic();
}

void test_panic_pos_2(int x) {
  signed z = 1;
  CFNumberRef value = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &z);
  if (x)
    panic();
  if (!x) {
    // This showed up in <rdar://problem/7796563>, where we silently missed checking
    // the function type for noreturn.  "panic()" is a hard-coded known panic function
    // that isn't always noreturn.
    panic_not_in_hardcoded_list();
  }
}

//===----------------------------------------------------------------------===//
// Test uses of blocks (closures)
//===----------------------------------------------------------------------===//

void test_blocks_1_pos(void) {
  NSNumber *number = [[NSNumber alloc] initWithInt:5];
  ^{}();
}

void test_blocks_1_indirect_release(void) {
  NSNumber *number = [[NSNumber alloc] initWithInt:5];
  ^{ [number release]; }();
}

void test_blocks_1_indirect_retain(void) {
  // Eventually this should be reported as a leak.
  NSNumber *number = [[NSNumber alloc] initWithInt:5];
  ^{ [number retain]; }();
}

void test_blocks_1_indirect_release_via_call(void) {
  NSNumber *number = [[NSNumber alloc] initWithInt:5];
  ^(NSObject *o){ [o release]; }(number);
}

void test_blocks_1_indirect_retain_via_call(void) {
  NSNumber *number = [[NSNumber alloc] initWithInt:5];
  ^(NSObject *o){ [o retain]; }(number);
}

//===--------------------------------------------------------------------===//
// Test sending message to super that returns an object alias.  Previously
// this caused a crash in the analyzer.
//===--------------------------------------------------------------------===//

@interface Rdar8015556 : NSObject {} @end
@implementation Rdar8015556
- (id)retain {
  return [super retain];
}
@end

// <rdar://problem/8272168> - Correcly handle Class<...> in Cocoa Conventions
// detector.

@protocol Prot_R8272168 @end
Class <Prot_R8272168> GetAClassThatImplementsProt_R8272168();
void r8272168() {
  GetAClassThatImplementsProt_R8272168();
}

// Test case for <rdar://problem/8356342>, which in the past triggered
// a false positive.
@interface RDar8356342
- (NSDate*) rdar8356342:(NSDate *)inValue;
@end

@implementation RDar8356342
- (NSDate*) rdar8356342:(NSDate*)inValue {
  NSDate *outValue = inValue;
  if (outValue == 0)
    outValue = [[NSDate alloc] init];

  if (outValue != inValue)
    [outValue autorelease];

  return outValue;
}
@end

// <rdar://problem/8724287> - This test case previously crashed because
// of a bug in BugReporter.
extern const void *CFDictionaryGetValue(CFDictionaryRef theDict, const void *key);
typedef struct __CFError * CFErrorRef;
extern const CFStringRef kCFErrorUnderlyingErrorKey;
extern CFDictionaryRef CFErrorCopyUserInfo(CFErrorRef err);
static void rdar_8724287(CFErrorRef error)
{
    CFErrorRef error_to_dump;

    error_to_dump = error;
    while (error_to_dump != ((void*)0)) {
        CFDictionaryRef info;

        info = CFErrorCopyUserInfo(error_to_dump);

        if (info != ((void*)0)) {
        }

        error_to_dump = (CFErrorRef) CFDictionaryGetValue(info, kCFErrorUnderlyingErrorKey);
    }
}

// <rdar://problem/9234108> - Make sure the model applies cf_consumed
// correctly in argument positions besides the first.
extern void *CFStringCreate(void);
extern void rdar_9234108_helper(void *key, void * CF_CONSUMED value);
void rdar_9234108() {
  rdar_9234108_helper(0, CFStringCreate());
}

// <rdar://problem/9726279> - Make sure that objc_method_family works
// to override naming conventions.
struct TwoDoubles {
  double one;
  double two;
};
typedef struct TwoDoubles TwoDoubles;

@interface NSValue (Mine)
- (id)_prefix_initWithTwoDoubles:(TwoDoubles)twoDoubles __attribute__((objc_method_family(init)));
@end

@implementation NSValue (Mine)
- (id)_prefix_initWithTwoDoubles:(TwoDoubles)twoDoubles
{
  return [self init];
}
@end

void rdar9726279() {
  TwoDoubles twoDoubles = { 0.0, 0.0 };
  NSValue *value = [[NSValue alloc] _prefix_initWithTwoDoubles:twoDoubles];
  [value release];
}

// <rdar://problem/9732321>
// Test camelcase support for CF conventions.  While Core Foundation APIs
// don't use camel casing, other code is allowed to use it.
CFArrayRef camelcase_create_1() {
  return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
}

CFArrayRef camelcase_createno() {
  return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
}

CFArrayRef camelcase_copy() {
  return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
}

CFArrayRef camelcase_copying() {
  return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
}

CFArrayRef copyCamelCase() {
  return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
}

CFArrayRef __copyCamelCase() {
  return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
}

CFArrayRef __createCamelCase() {
  return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
}

CFArrayRef camel_create() {
  return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
}


CFArrayRef camel_creat() {
  return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
}

CFArrayRef camel_copy() {
  return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
}

CFArrayRef camel_copyMachine() {
  return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
}

CFArrayRef camel_copymachine() {
  return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
}

// rdar://problem/8024350
@protocol F18P
- (id) clone;
@end
@interface F18 : NSObject<F18P> @end
@interface F18(Cat)
- (id) clone NS_RETURNS_RETAINED;
@end

@implementation F18
- (id) clone {
  return [F18 alloc];
}
@end

// Radar 6582778.
void rdar6582778(void) {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFTypeRef vals[] = { CFDateCreate(0, t) };
}

CFTypeRef global;

void rdar6582778_2(void) {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  global = CFDateCreate(0, t);
}

// <rdar://problem/10232019> - Test that objects passed to containers
// are marked "escaped".

void rdar10232019() {
  NSMutableArray *array = [NSMutableArray array];

  NSString *string = [[NSString alloc] initWithUTF8String:"foo"];
  [array addObject:string];
  [string release];

  NSString *otherString = [string stringByAppendingString:@"bar"];
  NSLog(@"%@", otherString);
}

void rdar10232019_positive() {
  NSMutableArray *array = [NSMutableArray array];

  NSString *string = [[NSString alloc] initWithUTF8String:"foo"];
  [string release];

  NSString *otherString = [string stringByAppendingString:@"bar"];
  NSLog(@"%@", otherString);
}

// RetainCountChecker support for XPC.
// <rdar://problem/9658496>
typedef void * xpc_object_t;
xpc_object_t _CFXPCCreateXPCObjectFromCFObject(CFTypeRef cf);
void xpc_release(xpc_object_t object);

void rdar9658496() {
  CFStringRef cf;
  xpc_object_t xpc;
  cf = CFStringCreateWithCString( ((CFAllocatorRef)0), "test", kCFStringEncodingUTF8 );
  xpc = _CFXPCCreateXPCObjectFromCFObject( cf );
  CFRelease(cf);
  xpc_release(xpc);
}

// Support annotations with method families.
@interface RDar10824732 : NSObject
- (id)initWithObj:(id CF_CONSUMED)obj;
@end

@implementation RDar10824732
- (id)initWithObj:(id)obj {
  [obj release];
  return [super init];
}
@end

void rdar_10824732() {
  @autoreleasepool {
    NSString *obj = @"test";
    RDar10824732 *foo = [[RDar10824732 alloc] initWithObj:obj];
    [foo release];
  }
}

// Stop tracking objects passed to functions, which take callbacks as parameters.
// radar://10973977
typedef int (*CloseCallback) (void *);
void ReaderForIO(CloseCallback ioclose, void *ioctx);
int IOClose(void *context);

@protocol SInS <NSObject>
@end

@interface radar10973977 : NSObject
- (id<SInS>)inputS;
- (void)reader;
@end

@implementation radar10973977
- (void)reader
{
    id<SInS> inputS = [[self inputS] retain];
    ReaderForIO(IOClose, inputS);
}
- (id<SInS>)inputS
{
    return 0;
}
@end

// Object escapes through a selector callback: radar://11398514
extern id NSApp;
@interface MySheetController
- (id<SInS>)inputS;
- (void)showDoSomethingSheetAction:(id)action;
- (void)sheetDidEnd:(NSWindow *)sheet returnCode:(int)returnCode contextInfo:(void *)contextInfo;
@end

@implementation MySheetController
- (id<SInS>)inputS {
    return 0;
}
- (void)showDoSomethingSheetAction:(id)action {
  id<SInS> inputS = [[self inputS] retain]; 
  [NSApp beginSheet:0
         modalForWindow:0
         modalDelegate:0
         didEndSelector:@selector(sheetDidEnd:returnCode:contextInfo:)
         contextInfo:(void *)inputS]; // no - warning
}
- (void)sheetDidEnd:(NSWindow *)sheet returnCode:(int)returnCode contextInfo:(void *)contextInfo {
   
      id contextObject = (id)contextInfo;
      [contextObject release];
}
@end
//===----------------------------------------------------------------------===//
// Test returning allocated memory in a struct.
//
// We currently don't have a general way to track pointers that "escape".
// Here we test that RetainCountChecker doesn't get excited about returning
// allocated CF objects in struct fields.
//===----------------------------------------------------------------------===//
void *malloc(size_t);
struct rdar11104566 { CFStringRef myStr; };
struct rdar11104566 test_rdar11104566() {
  CFStringRef cf = CFStringCreateWithCString( ((CFAllocatorRef)0), "test", kCFStringEncodingUTF8 );
  struct rdar11104566 V;
  V.myStr = cf;
  return V;
}

struct rdar11104566 *test_2_rdar11104566() {
  CFStringRef cf = CFStringCreateWithCString( ((CFAllocatorRef)0), "test", kCFStringEncodingUTF8 );
  struct rdar11104566 *V = (struct rdar11104566 *) malloc(sizeof(*V));
  V->myStr = cf;
  return V;
}

//===----------------------------------------------------------------------===//
// ObjC literals support.
//===----------------------------------------------------------------------===//

void test_objc_arrays() {
    { // CASE ONE -- OBJECT IN ARRAY CREATED DIRECTLY
        NSObject *o = [[NSObject alloc] init];
        NSArray *a = [[NSArray alloc] initWithObjects:o, (void*)0];
        [o release];
        [a description];
        [o description];
    }

    { // CASE TWO -- OBJECT IN ARRAY CREATED BY DUPING AUTORELEASED ARRAY
        NSObject *o = [[NSObject alloc] init];
        NSArray *a1 = [NSArray arrayWithObjects:o, (void*)0];
        NSArray *a2 = [[NSArray alloc] initWithArray:a1];
        [o release];        
        [a2 description];
        [o description];
    }

    { // CASE THREE -- OBJECT IN RETAINED @[]
        NSObject *o = [[NSObject alloc] init];
        NSArray *a3 = [@[o] retain];
        [o release];        
        [a3 description];
        [o description];
    }
    
    { // CASE FOUR -- OBJECT IN ARRAY CREATED BY DUPING @[]
        NSObject *o = [[NSObject alloc] init];
        NSArray *a = [[NSArray alloc] initWithArray:@[o]];
        [o release];
        
        [a description];
        [o description];
    }
    
    { // CASE FIVE -- OBJECT IN RETAINED @{}
        NSValue *o = [[NSValue alloc] init];
        NSDictionary *a = [@{o : o} retain];
        [o release];
        
        [a description];
        [o description];
    }
}

void test_objc_integer_literals() {
  id value = [@1 retain];
  [value description];
}

void test_objc_boxed_expressions(int x, const char *y) {
  id value = [@(x) retain];
  [value description];

  value = [@(y) retain];
  [value description];
}

// Test NSLog doesn't escape tracked objects.
void rdar11400885(int y)
{
  @autoreleasepool {
    NSString *printString;
    if(y > 2)
      printString = [[NSString alloc] init];
    else
      printString = [[NSString alloc] init];
    NSLog(@"Once %@", printString);
    [printString release];
    NSLog(@"Again: %@", printString);
  }
}

id makeCollectableNonLeak() {
  extern CFTypeRef CFCreateSomething();

  CFTypeRef object = CFCreateSomething(); // +1
  CFRetain(object); // +2
  id objCObject = NSMakeCollectable(object); // +2
  [objCObject release]; // +1
  return [objCObject autorelease]; // +0
}


void consumeAndStopTracking(id NS_CONSUMED obj, void (^callback)(void));
void CFConsumeAndStopTracking(CFTypeRef CF_CONSUMED obj, void (^callback)(void));

void testConsumeAndStopTracking() {
  id retained = [@[] retain]; // +1
  consumeAndStopTracking(retained, ^{});

  id doubleRetained = [[@[] retain] retain]; // +2
  consumeAndStopTracking(doubleRetained, ^{
    [doubleRetained release];
  });

  id unretained = @[]; // +0
  consumeAndStopTracking(unretained, ^{});
}

void testCFConsumeAndStopTracking() {
  id retained = [@[] retain]; // +1
  CFConsumeAndStopTracking((CFTypeRef)retained, ^{});

  id doubleRetained = [[@[] retain] retain];
  CFConsumeAndStopTracking((CFTypeRef)doubleRetained, ^{
    [doubleRetained release];
  });

  id unretained = @[]; // +0
  CFConsumeAndStopTracking((CFTypeRef)unretained, ^{});
}
//===----------------------------------------------------------------------===//
// Test 'pragma clang arc_cf_code_audited' support.
//===----------------------------------------------------------------------===//

typedef void *MyCFType;
#pragma clang arc_cf_code_audited begin
MyCFType CreateMyCFType();
#pragma clang arc_cf_code_audited end 
    
void test_custom_cf() {
  MyCFType x = CreateMyCFType();
}

//===----------------------------------------------------------------------===//
// Test calling CFPlugInInstanceCreate, which appears in CF but doesn't
// return a CF object.
//===----------------------------------------------------------------------===//

void test_CFPlugInInstanceCreate(CFUUIDRef factoryUUID, CFUUIDRef typeUUID) {
  CFPlugInInstanceCreate(kCFAllocatorDefault, factoryUUID, typeUUID);
}

// CHECK: 934:3: warning: 'IOServiceAddNotification' is deprecated
// CHECK:   IOServiceAddNotification(masterPort, notificationType, matching,
// CHECK:   ^
// CHECK: 204:15: note: 'IOServiceAddNotification' declared here
// CHECK: kern_return_t IOServiceAddNotification(  mach_port_t masterPort,  const io_name_t notificationType,  CFDictionaryRef matching,  mach_port_t wakePort,  uintptr_t reference,  io_iterator_t * notification ) __attribute__((deprecated)); // expected-note {{'IOServiceAddNotification' declared here}}
// CHECK:               ^
// CHECK: 1277:3: warning: class method '+drain' not found (return type defaults to 'id')
// CHECK:   [NSAutoreleasePool drain];
// CHECK:   ^                  ~~~~~
// CHECK: 1294:1: warning: 'ns_returns_retained' attribute only applies to methods that return an Objective-C object
// CHECK: - (int) returnsAnOwnedInt NS_RETURNS_RETAINED;
// CHECK: ^                         ~~~~~~~~~~~~~~~~~~~
// CHECK: 1300:1: warning: 'ns_returns_retained' attribute only applies to functions and methods
// CHECK: static int ownership_attribute_doesnt_go_here NS_RETURNS_RETAINED;
// CHECK: ^                                             ~~~~~~~~~~~~~~~~~~~
// CHECK: 324:7: warning: Reference-counted object is used after it is released
// CHECK:   t = CFDateGetAbsoluteTime(date);
// CHECK:       ^                     ~~~~
// CHECK: 335:7: warning: Reference-counted object is used after it is released
// CHECK:   t = CFDateGetAbsoluteTime(date);
// CHECK:       ^                     ~~~~
// CHECK: 366:20: warning: Potential leak of an object stored into 'date'
// CHECK:   CFDateRef date = CFDateCreate(0, t);
// CHECK:                    ^
// CHECK: 377:20: warning: Potential leak of an object stored into 'date'
// CHECK:   CFDateRef date = CFDateCreate(0, CFAbsoluteTimeGetCurrent());
// CHECK:                    ^
// CHECK: 385:20: warning: Potential leak of an object stored into 'date'
// CHECK:   CFDateRef date = CFDateCreate(0, CFAbsoluteTimeGetCurrent());  //expected-warning{{leak}}
// CHECK:                    ^
// CHECK: 387:10: warning: Potential leak of an object stored into 'date'
// CHECK:   date = CFDateCreate(0, CFAbsoluteTimeGetCurrent());
// CHECK:          ^
// CHECK: 396:20: warning: Potential leak of an object stored into 'date'
// CHECK:   CFDateRef date = MyDateCreate();
// CHECK:                    ^
// CHECK: 405:17: warning: Dereference of null pointer (loaded from variable 'p')
// CHECK:   if (!date) *p = 1;
// CHECK:               ~ ^
// CHECK: 414:20: warning: Potential leak of an object stored into 'disk'
// CHECK:   DADiskRef disk = DADiskCreateFromBSDName(kCFAllocatorDefault, 0, "hello");
// CHECK:                    ^
// CHECK: 417:10: warning: Potential leak of an object stored into 'disk'
// CHECK:   disk = DADiskCreateFromIOMedia(kCFAllocatorDefault, 0, media);
// CHECK:          ^
// CHECK: 420:26: warning: Potential leak of an object stored into 'dict'
// CHECK:   CFDictionaryRef dict = DADiskCopyDescription(d);
// CHECK:                          ^
// CHECK: 423:10: warning: Potential leak of an object stored into 'disk'
// CHECK:   disk = DADiskCopyWholeDisk(d);
// CHECK:          ^
// CHECK: 426:30: warning: Potential leak of an object stored into 'dissenter'
// CHECK:   DADissenterRef dissenter = DADissenterCreate(kCFAllocatorDefault,
// CHECK:                              ^
// CHECK: 430:26: warning: Potential leak of an object stored into 'session'
// CHECK:   DASessionRef session = DASessionCreate(kCFAllocatorDefault);
// CHECK:                          ^
// CHECK: 456:3: warning: Incorrect decrement of the reference count of an object that is not owned at this point by the caller
// CHECK:   CFRelease(s1);
// CHECK:   ^         ~~
// CHECK: 464:17: warning: Potential leak of an object stored into 'o'
// CHECK:   CFTypeRef o = MyCreateFun();
// CHECK:                 ^
// CHECK: 475:3: warning: Object sent -autorelease too many times
// CHECK:   [(id) A autorelease];
// CHECK:   ^~~~~~~~~~~~~~~~~~~~
// CHECK: 482:3: warning: Object sent -autorelease too many times
// CHECK:   return A;
// CHECK:   ^
// CHECK: 489:25: warning: Object sent -autorelease too many times
// CHECK:   CFMutableArrayRef B = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
// CHECK:                         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK: 497:3: warning: Potential leak of an object
// CHECK:   CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
// CHECK:   ^
// CHECK: 516:5: warning: Null pointer argument in call to CFRelease
// CHECK:     CFRelease(p);
// CHECK:     ^         ~
// CHECK: 519:5: warning: Null pointer argument in call to CFRetain
// CHECK:     CFRetain(p);
// CHECK:     ^        ~
// CHECK: 561:3: warning: Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected
// CHECK:   return s;
// CHECK:   ^
// CHECK: 574:20: warning: Potential leak of an object stored into 'kind'
// CHECK:   NSString *kind = [[NSString alloc] initWithUTF8String:inkind];
// CHECK:                    ^
// CHECK: 596:13: warning: Array access (from variable 'kindC') results in a null pointer dereference
// CHECK:   if(!isFoo(kindC[0]))
// CHECK:             ^~~~~
// CHECK: 602:3: warning: Incorrect decrement of the reference count of an object that is not owned at this point by the caller
// CHECK:   [name release];
// CHECK:   ^~~~~
// CHECK: 626:3: warning: Reference-counted object is used after it is released
// CHECK:   [foo release];
// CHECK:   ^~~~
// CHECK: 635:3: warning: Reference-counted object is used after it is released
// CHECK:   [foo dealloc];
// CHECK:   ^~~~
// CHECK: 687:31: warning: Potential leak of an object stored into 'dict'
// CHECK:  NSMutableDictionary *dict = [[NSMutableDictionary dictionaryWithCapacity:4] retain];
// CHECK:                               ^
// CHECK: 699:31: warning: Potential leak of an object stored into 'dict'
// CHECK:  NSMutableDictionary *dict = [[NSMutableDictionary dictionaryWithCapacity:4] retain];
// CHECK:                               ^
// CHECK: 712:3: warning: Incorrect decrement of the reference count of an object that is not owned at this point by the caller
// CHECK:   [array release];
// CHECK:   ^~~~~~
// CHECK: 788:3: warning: Potential leak of an object
// CHECK:   [[RDar6320065Subclass alloc] init];
// CHECK:   ^
// CHECK: 794:3: warning: Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected
// CHECK:   return [self autorelease];
// CHECK:   ^
// CHECK: 832:37: warning: Potential leak of an object
// CHECK: - (NSString*) NoCopyString { return [[NSString alloc] init]; }
// CHECK:                                     ^
// CHECK: 833:37: warning: Potential leak of an object
// CHECK: - (NSString*) noCopyString { return [[NSString alloc] init]; }
// CHECK:                                     ^
// CHECK: 837:3: warning: Potential leak of an object
// CHECK:   [x NoCopyString];
// CHECK:   ^
// CHECK: 838:3: warning: Potential leak of an object
// CHECK:   [x noCopyString];
// CHECK:   ^
// CHECK: 865:10: warning: Potential leak of an object
// CHECK:   return [[NSString alloc] init];
// CHECK:          ^
// CHECK: 895:3: warning: Potential leak of an object
// CHECK:   [view createSnapshotImageOfType:str];
// CHECK:   ^
// CHECK: 896:3: warning: Potential leak of an object
// CHECK:   [renderer createSnapshotImageOfType:str];
// CHECK:   ^
// CHECK: 897:3: warning: Potential leak of an object
// CHECK:   [context createCGImage:img fromRect:rect];
// CHECK:   ^
// CHECK: 898:3: warning: Potential leak of an object
// CHECK:   [context createCGImage:img fromRect:rect format:form colorSpace:cs];
// CHECK:   ^
// CHECK: 907:3: warning: Potential leak of an object
// CHECK:   [context createCGLayerWithSize:size info:d];
// CHECK:   ^
// CHECK: 916:3: warning: Potential leak of an object
// CHECK:   IOBSDNameMatching(masterPort, options, bsdName);
// CHECK:   ^
// CHECK: 920:3: warning: Potential leak of an object
// CHECK:   IOServiceMatching(name);
// CHECK:   ^
// CHECK: 924:3: warning: Potential leak of an object
// CHECK:   IOServiceNameMatching(name);
// CHECK:   ^
// CHECK: 934:3: warning: Reference-counted object is used after it is released
// CHECK:   IOServiceAddNotification(masterPort, notificationType, matching,
// CHECK:   ^                                                      ~~~~~~~~
// CHECK: 939:3: warning: Potential leak of an object
// CHECK:   IORegistryEntryIDMatching(entryID);
// CHECK:   ^
// CHECK: 944:3: warning: Potential leak of an object
// CHECK:   IOOpenFirmwarePathMatching(masterPort, options, path);
// CHECK:   ^
// CHECK: 950:3: warning: Reference-counted object is used after it is released
// CHECK:   CFRelease(matching);
// CHECK:   ^         ~~~~~~~~
// CHECK: 956:3: warning: Reference-counted object is used after it is released
// CHECK:   CFRelease(matching);
// CHECK:   ^         ~~~~~~~~
// CHECK: 964:3: warning: Reference-counted object is used after it is released
// CHECK:   CFRelease(matching);
// CHECK:   ^         ~~~~~~~~
// CHECK: 1005:22: warning: Potential leak of an object stored into 'number'
// CHECK:   NSNumber *number = [[NSNumber alloc] initWithInt:5];
// CHECK:                      ^
// CHECK: 1030:41: warning: Potential leak of an object
// CHECK:     CGGradientCreateWithColorComponents(CGColorSpaceCreateDeviceRGB(),
// CHECK:                                         ^
// CHECK: 1052:4: warning: Potential leak of an object stored into 'myGradient'
// CHECK:    CGGradientCreateWithColorComponents(CGColorSpaceCreateDeviceRGB(), components, locations, num_locations);
// CHECK:    ^
// CHECK: 1052:40: warning: Potential leak of an object
// CHECK:    CGGradientCreateWithColorComponents(CGColorSpaceCreateDeviceRGB(), components, locations, num_locations);
// CHECK:                                        ^
// CHECK: 1090:22: warning: Potential leak of an object stored into 'number'
// CHECK:   NSNumber *number = [[NSNumber alloc] initWithInt:5];
// CHECK:                      ^
// CHECK: 1225:3: warning: Potential leak of an object
// CHECK:   CGBitmapContextCreateWithData(data, width, height, bitsPerComponent,
// CHECK:   ^
// CHECK: 1243:10: warning: Potential leak of an object stored into 'y'
// CHECK:   id y = [x new_stuff];
// CHECK:          ^
// CHECK: 1264:22: warning: Potential leak of an object stored into 'number'
// CHECK:   NSNumber *number = [[NSNumber alloc] initWithInt:5];
// CHECK:                      ^
// CHECK: 1274:3: warning: The 'release' message should be sent to instances of class 'RDar7252064' and not the class directly
// CHECK:   [RDar7252064 release];
// CHECK:   ^~~~~~~~~~~~~~~~~~~~~
// CHECK: 1275:3: warning: The 'retain' message should be sent to instances of class 'RDar7252064' and not the class directly
// CHECK:   [RDar7252064 retain];
// CHECK:   ^~~~~~~~~~~~~~~~~~~~
// CHECK: 1276:3: warning: The 'autorelease' message should be sent to instances of class 'RDar7252064' and not the class directly
// CHECK:   [RDar7252064 autorelease];
// CHECK:   ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK: 1277:3: warning: The 'drain' message should be sent to instances of class 'NSAutoreleasePool' and not the class directly
// CHECK:   [NSAutoreleasePool drain];
// CHECK:   ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK: 1303:19: warning: Potential leak of an object stored into 'str'
// CHECK:   NSString *str = [X returnsAnOwnedString];
// CHECK:                   ^
// CHECK: 1307:19: warning: Potential leak of an object stored into 'str'
// CHECK:   NSString *str = [X returnsAnOwnedCFString];
// CHECK:                   ^
// CHECK: 1312:20: warning: Potential leak of an object stored into 'str2'
// CHECK:   NSString *str2 = [X newStringNoAttr];
// CHECK:                    ^
// CHECK: 1316:26: warning: Potential leak of an object stored into 'x'
// CHECK:   TestOwnershipAttr *x = [TestOwnershipAttr alloc];
// CHECK:                          ^
// CHECK: 1320:26: warning: Potential leak of an object stored into 'x'
// CHECK:   TestOwnershipAttr *x = [[TestOwnershipAttr alloc] pseudoInit];
// CHECK:                          ^
// CHECK: 1324:26: warning: Potential leak of an object stored into 'x'
// CHECK:   TestOwnershipAttr *x = [[TestOwnershipAttr alloc] pseudoInit];// expected-warning{{leak}}
// CHECK:                          ^
// CHECK: 1356:10: warning: Potential leak of an object
// CHECK:   return [NSString alloc];
// CHECK:          ^
// CHECK: 1389:3: warning: Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected
// CHECK:   return (CFDateRef)[(id)[self returnsCFRetainedAsCF] autorelease];
// CHECK:   ^
// CHECK: 1393:20: warning: Potential leak of an object
// CHECK:   return (NSDate*) returnsRetainedCFDate();
// CHECK:                    ^
// CHECK: 1397:10: warning: Potential leak of an object
// CHECK:   return returnsRetainedCFDate();
// CHECK:          ^
// CHECK: 1418:23: warning: Potential leak of an object stored into 'value'
// CHECK:   CFNumberRef value = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &z);
// CHECK:                       ^
// CHECK: 1429:23: warning: Potential leak of an object stored into 'value'
// CHECK:   CFNumberRef value = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &z);
// CHECK:                       ^
// CHECK: 1452:22: warning: Potential leak of an object stored into 'number'
// CHECK:   NSNumber *number = [[NSNumber alloc] initWithInt:5];
// CHECK:                      ^
// CHECK: 1473:22: warning: Potential leak of an object stored into 'number'
// CHECK:   NSNumber *number = [[NSNumber alloc] initWithInt:5];
// CHECK:                      ^
// CHECK: 1531:16: warning: Potential leak of an object stored into 'info'
// CHECK:         info = CFErrorCopyUserInfo(error_to_dump);
// CHECK:                ^
// CHECK: 1581:10: warning: Potential leak of an object
// CHECK:   return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
// CHECK:          ^
// CHECK: 1589:10: warning: Potential leak of an object
// CHECK:   return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
// CHECK:          ^
// CHECK: 1610:10: warning: Potential leak of an object
// CHECK:   return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
// CHECK:          ^
// CHECK: 1622:10: warning: Potential leak of an object
// CHECK:   return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
// CHECK:          ^
// CHECK: 1643:24: warning: Potential leak of an object stored into 'vals'
// CHECK:   CFTypeRef vals[] = { CFDateCreate(0, t) };
// CHECK:                        ^
// CHECK: 1673:27: warning: Reference-counted object is used after it is released
// CHECK:   NSString *otherString = [string stringByAppendingString:@"bar"];
// CHECK:                           ^~~~~~~
// CHECK: 1794:22: warning: Potential leak of an object stored into 'a'
// CHECK:         NSArray *a = [[NSArray alloc] initWithObjects:o, (void*)0];
// CHECK:                      ^
// CHECK: 1803:23: warning: Potential leak of an object stored into 'a2'
// CHECK:         NSArray *a2 = [[NSArray alloc] initWithArray:a1];
// CHECK:                       ^
// CHECK: 1811:24: warning: Potential leak of an object stored into 'a3'
// CHECK:         NSArray *a3 = [@[o] retain];
// CHECK:                        ^
// CHECK: 1819:22: warning: Potential leak of an object stored into 'a'
// CHECK:         NSArray *a = [[NSArray alloc] initWithArray:@[o]];
// CHECK:                      ^
// CHECK: 1828:28: warning: Potential leak of an object stored into 'a'
// CHECK:         NSDictionary *a = [@{o : o} retain];
// CHECK:                            ^
// CHECK: 1837:15: warning: Potential leak of an object stored into 'value'
// CHECK:   id value = [@1 retain];
// CHECK:               ^
// CHECK: 1842:15: warning: Potential leak of an object stored into 'value'
// CHECK:   id value = [@(x) retain];
// CHECK:               ^
// CHECK: 1845:12: warning: Potential leak of an object stored into 'value'
// CHECK:   value = [@(y) retain];
// CHECK:            ^
// CHECK: 1860:5: warning: Reference-counted object is used after it is released
// CHECK:     NSLog(@"Again: %@", printString);
// CHECK:     ^                   ~~~~~~~~~~~
// CHECK: 1888:3: warning: Incorrect decrement of the reference count of an object that is not owned at this point by the caller
// CHECK:   consumeAndStopTracking(unretained, ^{});
// CHECK:   ^                      ~~~~~~~~~~
// CHECK: 1901:3: warning: Incorrect decrement of the reference count of an object that is not owned at this point by the caller
// CHECK:   CFConsumeAndStopTracking((CFTypeRef)unretained, ^{});
// CHECK:   ^                        ~~~~~~~~~~~~~~~~~~~~~
// CHECK: 1913:16: warning: Potential leak of an object stored into 'x'
// CHECK:   MyCFType x = CreateMyCFType();
// CHECK:                ^
// CHECK: 319:20: note: Call to function 'CFDateCreate' returns a Core Foundation object with a +1 retain count
// CHECK:   CFDateRef date = CFDateCreate(0, t);
// CHECK:                    ^
// CHECK: 320:3: note: Reference count incremented. The object now has a +2 retain count
// CHECK:   CFRetain(date);
// CHECK:   ^
// CHECK: 321:3: note: Reference count decremented. The object now has a +1 retain count
// CHECK:   CFRelease(date);
// CHECK:   ^
// CHECK: 323:3: note: Object released
// CHECK:   CFRelease(date);
// CHECK:   ^
// CHECK: 324:7: note: Reference-counted object is used after it is released
// CHECK:   t = CFDateGetAbsoluteTime(date);
// CHECK:       ^
// CHECK: 330:20: note: Call to function 'CFDateCreate' returns a Core Foundation object with a +1 retain count
// CHECK:   CFDateRef date = CFDateCreate(0, t);  
// CHECK:                    ^
// CHECK: 331:3: note: Reference count incremented. The object now has a +2 retain count
// CHECK:   [((NSDate*) date) retain];
// CHECK:   ^
// CHECK: 332:3: note: Reference count decremented. The object now has a +1 retain count
// CHECK:   CFRelease(date);
// CHECK:   ^
// CHECK: 334:3: note: Object released
// CHECK:   [((NSDate*) date) release];
// CHECK:   ^
// CHECK: 335:7: note: Reference-counted object is used after it is released
// CHECK:   t = CFDateGetAbsoluteTime(date);
// CHECK:       ^
// CHECK: 366:20: note: Call to function 'CFDateCreate' returns a Core Foundation object with a +1 retain count
// CHECK:   CFDateRef date = CFDateCreate(0, t);
// CHECK:                    ^
// CHECK: 368:3: note: Taking false branch
// CHECK:   if (x)
// CHECK:   ^
// CHECK: 371:10: note: Object leaked: object allocated and stored into 'date' is not referenced later in this execution path and has a retain count of +1
// CHECK:   return t;
// CHECK:          ^
// CHECK: 377:20: note: Call to function 'CFDateCreate' returns a Core Foundation object with a +1 retain count
// CHECK:   CFDateRef date = CFDateCreate(0, CFAbsoluteTimeGetCurrent());
// CHECK:                    ^
// CHECK: 378:3: note: Reference count incremented. The object now has a +2 retain count
// CHECK:   CFRetain(date);
// CHECK:   ^
// CHECK: 379:3: note: Object returned to caller as an owning reference (single retain count transferred to caller)
// CHECK:   return date;
// CHECK:   ^
// CHECK: 380:1: note: Object leaked: object allocated and stored into 'date' is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 385:20: note: Call to function 'CFDateCreate' returns a Core Foundation object with a +1 retain count
// CHECK:   CFDateRef date = CFDateCreate(0, CFAbsoluteTimeGetCurrent());  //expected-warning{{leak}}
// CHECK:                    ^
// CHECK: 386:3: note: Reference count incremented. The object now has a +2 retain count
// CHECK:   CFRetain(date);
// CHECK:   ^
// CHECK: 388:3: note: Object leaked: object allocated and stored into 'date' is not referenced later in this execution path and has a retain count of +2
// CHECK:   return date;
// CHECK:   ^
// CHECK: 387:10: note: Call to function 'CFDateCreate' returns a Core Foundation object with a +1 retain count
// CHECK:   date = CFDateCreate(0, CFAbsoluteTimeGetCurrent());
// CHECK:          ^
// CHECK: 388:3: note: Object returned to caller as an owning reference (single retain count transferred to caller)
// CHECK:   return date;
// CHECK:   ^
// CHECK: 388:3: note: Object leaked: object allocated and stored into 'date' is returned from a function whose name ('f7') does not contain 'Copy' or 'Create'.  This violates the naming convention rules given in the Memory Management Guide for Core Foundation
// CHECK: 396:20: note: Call to function 'MyDateCreate' returns a Core Foundation object with a +1 retain count
// CHECK:   CFDateRef date = MyDateCreate();
// CHECK:                    ^
// CHECK: 397:3: note: Reference count incremented. The object now has a +2 retain count
// CHECK:   CFRetain(date);  
// CHECK:   ^
// CHECK: 398:3: note: Object returned to caller as an owning reference (single retain count transferred to caller)
// CHECK:   return date;
// CHECK:   ^
// CHECK: 399:1: note: Object leaked: object allocated and stored into 'date' is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 403:3: note: Variable 'p' initialized to a null pointer value
// CHECK:   int *p = 0;
// CHECK:   ^
// CHECK: 405:3: note: Taking true branch
// CHECK:   if (!date) *p = 1;
// CHECK:   ^
// CHECK: 405:14: note: Dereference of null pointer (loaded from variable 'p')
// CHECK:   if (!date) *p = 1;
// CHECK:              ^
// CHECK: 415:3: note: Taking false branch
// CHECK:   if (disk) NSLog(@"ok");
// CHECK:   ^
// CHECK: 418:3: note: Taking false branch
// CHECK:   if (disk) NSLog(@"ok");
// CHECK:   ^
// CHECK: 420:26: note: Call to function 'DADiskCopyDescription' returns a Core Foundation object with a +1 retain count
// CHECK:   CFDictionaryRef dict = DADiskCopyDescription(d);
// CHECK:                          ^
// CHECK: 421:7: note: Assuming 'dict' is non-null
// CHECK:   if (dict) NSLog(@"ok"); 
// CHECK:       ^
// CHECK: 421:3: note: Taking true branch
// CHECK:   if (dict) NSLog(@"ok"); 
// CHECK:   ^
// CHECK: 421:20: note: Object leaked: object allocated and stored into 'dict' is not referenced later in this execution path and has a retain count of +1
// CHECK:   if (dict) NSLog(@"ok"); 
// CHECK:                    ^
// CHECK: 415:3: note: Taking false branch
// CHECK:   if (disk) NSLog(@"ok");
// CHECK:   ^
// CHECK: 418:3: note: Taking false branch
// CHECK:   if (disk) NSLog(@"ok");
// CHECK:   ^
// CHECK: 421:3: note: Taking false branch
// CHECK:   if (dict) NSLog(@"ok"); 
// CHECK:   ^
// CHECK: 423:10: note: Call to function 'DADiskCopyWholeDisk' returns a Core Foundation object with a +1 retain count
// CHECK:   disk = DADiskCopyWholeDisk(d);
// CHECK:          ^
// CHECK: 424:7: note: Assuming 'disk' is non-null
// CHECK:   if (disk) NSLog(@"ok");
// CHECK:       ^
// CHECK: 424:3: note: Taking true branch
// CHECK:   if (disk) NSLog(@"ok");
// CHECK:   ^
// CHECK: 424:20: note: Object leaked: object allocated and stored into 'disk' is not referenced later in this execution path and has a retain count of +1
// CHECK:   if (disk) NSLog(@"ok");
// CHECK:                    ^
// CHECK: 415:3: note: Taking false branch
// CHECK:   if (disk) NSLog(@"ok");
// CHECK:   ^
// CHECK: 418:3: note: Taking false branch
// CHECK:   if (disk) NSLog(@"ok");
// CHECK:   ^
// CHECK: 421:3: note: Taking false branch
// CHECK:   if (dict) NSLog(@"ok"); 
// CHECK:   ^
// CHECK: 424:3: note: Taking false branch
// CHECK:   if (disk) NSLog(@"ok");
// CHECK:   ^
// CHECK: 426:30: note: Call to function 'DADissenterCreate' returns a Core Foundation object with a +1 retain count
// CHECK:   DADissenterRef dissenter = DADissenterCreate(kCFAllocatorDefault,
// CHECK:                              ^
// CHECK: 428:7: note: Assuming 'dissenter' is non-null
// CHECK:   if (dissenter) NSLog(@"ok");
// CHECK:       ^
// CHECK: 428:3: note: Taking true branch
// CHECK:   if (dissenter) NSLog(@"ok");
// CHECK:   ^
// CHECK: 428:25: note: Object leaked: object allocated and stored into 'dissenter' is not referenced later in this execution path and has a retain count of +1
// CHECK:   if (dissenter) NSLog(@"ok");
// CHECK:                         ^
// CHECK: 415:3: note: Taking false branch
// CHECK:   if (disk) NSLog(@"ok");
// CHECK:   ^
// CHECK: 418:3: note: Taking false branch
// CHECK:   if (disk) NSLog(@"ok");
// CHECK:   ^
// CHECK: 421:3: note: Taking false branch
// CHECK:   if (dict) NSLog(@"ok"); 
// CHECK:   ^
// CHECK: 424:3: note: Taking false branch
// CHECK:   if (disk) NSLog(@"ok");
// CHECK:   ^
// CHECK: 428:3: note: Taking false branch
// CHECK:   if (dissenter) NSLog(@"ok");
// CHECK:   ^
// CHECK: 430:26: note: Call to function 'DASessionCreate' returns a Core Foundation object with a +1 retain count
// CHECK:   DASessionRef session = DASessionCreate(kCFAllocatorDefault);
// CHECK:                          ^
// CHECK: 431:7: note: Assuming 'session' is non-null
// CHECK:   if (session) NSLog(@"ok");
// CHECK:       ^
// CHECK: 431:3: note: Taking true branch
// CHECK:   if (session) NSLog(@"ok");
// CHECK:   ^
// CHECK: 431:23: note: Object leaked: object allocated and stored into 'session' is not referenced later in this execution path and has a retain count of +1
// CHECK:   if (session) NSLog(@"ok");
// CHECK:                       ^
// CHECK: 415:3: note: Taking false branch
// CHECK:   if (disk) NSLog(@"ok");
// CHECK:   ^
// CHECK: 417:10: note: Call to function 'DADiskCreateFromIOMedia' returns a Core Foundation object with a +1 retain count
// CHECK:   disk = DADiskCreateFromIOMedia(kCFAllocatorDefault, 0, media);
// CHECK:          ^
// CHECK: 418:7: note: Assuming 'disk' is non-null
// CHECK:   if (disk) NSLog(@"ok");
// CHECK:       ^
// CHECK: 418:3: note: Taking true branch
// CHECK:   if (disk) NSLog(@"ok");
// CHECK:   ^
// CHECK: 421:3: note: Taking false branch
// CHECK:   if (dict) NSLog(@"ok"); 
// CHECK:   ^
// CHECK: 424:3: note: Taking false branch
// CHECK:   if (disk) NSLog(@"ok");
// CHECK:   ^
// CHECK: 428:3: note: Taking false branch
// CHECK:   if (dissenter) NSLog(@"ok");
// CHECK:   ^
// CHECK: 431:3: note: Taking false branch
// CHECK:   if (session) NSLog(@"ok");
// CHECK:   ^
// CHECK: 432:1: note: Object leaked: object allocated and stored into 'disk' is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 450:22: note: Call to function 'CFArrayGetValueAtIndex' returns a Core Foundation object with a +0 retain count
// CHECK:   s1 = (CFStringRef) CFArrayGetValueAtIndex(A, 0);
// CHECK:                      ^
// CHECK: 456:3: note: Incorrect decrement of the reference count of an object that is not owned at this point by the caller
// CHECK:   CFRelease(s1);
// CHECK:   ^
// CHECK: 464:17: note: Call to function 'MyCreateFun' returns a Core Foundation object with a +1 retain count
// CHECK:   CFTypeRef o = MyCreateFun();
// CHECK:                 ^
// CHECK: 465:1: note: Object leaked: object allocated and stored into 'o' is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 473:25: note: Call to function 'CFArrayCreateMutable' returns a Core Foundation object with a +1 retain count
// CHECK:   CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
// CHECK:                         ^
// CHECK: 474:3: note: Object sent -autorelease message
// CHECK:   [(id) A autorelease];
// CHECK:   ^
// CHECK: 475:3: note: Object sent -autorelease message
// CHECK:   [(id) A autorelease];
// CHECK:   ^
// CHECK: 476:1: note: Object over-autoreleased: object was sent -autorelease 2 times but the object has a +1 retain count
// CHECK: }
// CHECK: ^
// CHECK: 479:25: note: Call to function 'CFArrayCreateMutable' returns a Core Foundation object with a +1 retain count
// CHECK:   CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
// CHECK:                         ^
// CHECK: 480:3: note: Object sent -autorelease message
// CHECK:   [(id) A autorelease];
// CHECK:   ^
// CHECK: 481:3: note: Object sent -autorelease message
// CHECK:   [(id) A autorelease]; 
// CHECK:   ^
// CHECK: 482:3: note: Object over-autoreleased: object was sent -autorelease 2 times but the object has a +0 retain count
// CHECK:   return A;
// CHECK:   ^
// CHECK: 486:25: note: Call to function 'CFArrayCreateMutable' returns a Core Foundation object with a +1 retain count
// CHECK:   CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
// CHECK:                         ^
// CHECK: 487:3: note: Object sent -autorelease message
// CHECK:   [(id) A autorelease];
// CHECK:   ^
// CHECK: 488:3: note: Object sent -autorelease message
// CHECK:   [(id) A autorelease]; 
// CHECK:   ^
// CHECK: 489:25: note: Object over-autoreleased: object was sent -autorelease 2 times but the object has a +1 retain count
// CHECK:   CFMutableArrayRef B = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
// CHECK:                         ^
// CHECK: 497:3: note: Call to function 'CFArrayCreateMutable' returns a Core Foundation object with a +1 retain count
// CHECK:   CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
// CHECK:   ^
// CHECK: 498:1: note: Object leaked: allocated object is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 512:7: note: Assuming 'p' is null
// CHECK:   if (p)
// CHECK:       ^
// CHECK: 512:7: note: Assuming pointer value is null
// CHECK: 512:3: note: Taking false branch
// CHECK:   if (p)
// CHECK:   ^
// CHECK: 515:3: note: Taking true branch
// CHECK:   if (x) {
// CHECK:   ^
// CHECK: 516:5: note: Null pointer argument in call to CFRelease
// CHECK:     CFRelease(p);
// CHECK:     ^
// CHECK: 512:7: note: Assuming 'p' is null
// CHECK:   if (p)
// CHECK:       ^
// CHECK: 512:7: note: Assuming pointer value is null
// CHECK: 512:3: note: Taking false branch
// CHECK:   if (p)
// CHECK:   ^
// CHECK: 515:3: note: Taking false branch
// CHECK:   if (x) {
// CHECK:   ^
// CHECK: 519:5: note: Null pointer argument in call to CFRetain
// CHECK:     CFRetain(p);
// CHECK:     ^
// CHECK: 560:17: note: Method returns an Objective-C object with a +0 retain count
// CHECK:   NSString *s = [NSString stringWithUTF8String:"hello"];
// CHECK:                 ^
// CHECK: 561:3: note: Object returned to caller with a +0 retain count
// CHECK:   return s;
// CHECK:   ^
// CHECK: 561:3: note: Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected
// CHECK: 574:20: note: Method returns an Objective-C object with a +1 retain count
// CHECK:   NSString *kind = [[NSString alloc] initWithUTF8String:inkind];
// CHECK:                    ^
// CHECK: 581:3: note: Taking true branch
// CHECK:   if(!name)
// CHECK:   ^
// CHECK: 582:5: note: Object leaked: object allocated and stored into 'kind' is not referenced later in this execution path and has a retain count of +1
// CHECK:     return;
// CHECK:     ^
// CHECK: 581:3: note: Taking false branch
// CHECK:   if(!name)
// CHECK:   ^
// CHECK: 584:3: note: Variable 'kindC' initialized to a null pointer value
// CHECK:   const char *kindC = 0;
// CHECK:   ^
// CHECK: 592:3: note: Taking false branch
// CHECK:   if(kind)
// CHECK:   ^
// CHECK: 594:3: note: Taking true branch
// CHECK:   if(name)
// CHECK:   ^
// CHECK: 596:13: note: Array access (from variable 'kindC') results in a null pointer dereference
// CHECK:   if(!isFoo(kindC[0]))
// CHECK:             ^
// CHECK: 580:20: note: Method returns an Objective-C object with a +0 retain count
// CHECK:   NSString *name = [NSString stringWithUTF8String:inname];
// CHECK:                    ^
// CHECK: 581:6: note: Assuming 'name' is non-nil
// CHECK:   if(!name)
// CHECK:      ^
// CHECK: 581:3: note: Taking false branch
// CHECK:   if(!name)
// CHECK:   ^
// CHECK: 592:3: note: Taking true branch
// CHECK:   if(kind)
// CHECK:   ^
// CHECK: 594:3: note: Taking true branch
// CHECK:   if(name)
// CHECK:   ^
// CHECK: 596:3: note: Taking false branch
// CHECK:   if(!isFoo(kindC[0]))
// CHECK:   ^
// CHECK: 598:3: note: Taking false branch
// CHECK:   if(!isFoo(nameC[0]))
// CHECK:   ^
// CHECK: 602:3: note: Incorrect decrement of the reference count of an object that is not owned at this point by the caller
// CHECK:   [name release];
// CHECK:   ^
// CHECK: 624:12: note: Method returns an Objective-C object with a +1 retain count
// CHECK:   id foo = [[NSString alloc] init];
// CHECK:            ^
// CHECK: 625:3: note: Object released by directly sending the '-dealloc' message
// CHECK:   [foo dealloc];
// CHECK:   ^
// CHECK: 626:3: note: Reference-counted object is used after it is released
// CHECK:   [foo release];
// CHECK:   ^
// CHECK: 633:12: note: Method returns an Objective-C object with a +1 retain count
// CHECK:   id foo = [[NSString alloc] init];
// CHECK:            ^
// CHECK: 634:3: note: Object released
// CHECK:   [foo release];
// CHECK:   ^
// CHECK: 635:3: note: Reference-counted object is used after it is released
// CHECK:   [foo dealloc];
// CHECK:   ^
// CHECK: 687:31: note: Method returns an Objective-C object with a +0 retain count
// CHECK:  NSMutableDictionary *dict = [[NSMutableDictionary dictionaryWithCapacity:4] retain];
// CHECK:                               ^
// CHECK: 687:30: note: Reference count incremented. The object now has a +1 retain count
// CHECK:  NSMutableDictionary *dict = [[NSMutableDictionary dictionaryWithCapacity:4] retain];
// CHECK:                              ^
// CHECK: 692:1: note: Object leaked: object allocated and stored into 'dict' is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 699:31: note: Method returns an Objective-C object with a +0 retain count
// CHECK:  NSMutableDictionary *dict = [[NSMutableDictionary dictionaryWithCapacity:4] retain];
// CHECK:                               ^
// CHECK: 699:30: note: Reference count incremented. The object now has a +1 retain count
// CHECK:  NSMutableDictionary *dict = [[NSMutableDictionary dictionaryWithCapacity:4] retain];
// CHECK:                              ^
// CHECK: 700:2: note: Taking false branch
// CHECK:  if (window) 
// CHECK:  ^
// CHECK: 702:1: note: Object leaked: object allocated and stored into 'dict' is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 711:20: note: Method returns an Objective-C object with a +0 retain count
// CHECK:   NSArray *array = [NSArray array];
// CHECK:                    ^
// CHECK: 712:3: note: Incorrect decrement of the reference count of an object that is not owned at this point by the caller
// CHECK:   [array release];
// CHECK:   ^
// CHECK: 788:3: note: Method returns an Objective-C object with a +1 retain count
// CHECK:   [[RDar6320065Subclass alloc] init];
// CHECK:   ^
// CHECK: 790:1: note: Object leaked: allocated object is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 793:10: note: Method returns an Objective-C object with a +1 retain count
// CHECK:   self = [[RDar6320065Subclass alloc] init];
// CHECK:          ^
// CHECK: 794:10: note: Object sent -autorelease message
// CHECK:   return [self autorelease];
// CHECK:          ^
// CHECK: 794:3: note: Object returned to caller with a +0 retain count
// CHECK:   return [self autorelease];
// CHECK:   ^
// CHECK: 794:3: note: Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected
// CHECK: 832:37: note: Method returns an Objective-C object with a +1 retain count
// CHECK: - (NSString*) NoCopyString { return [[NSString alloc] init]; }
// CHECK:                                     ^
// CHECK: 832:30: note: Object returned to caller as an owning reference (single retain count transferred to caller)
// CHECK: - (NSString*) NoCopyString { return [[NSString alloc] init]; }
// CHECK:                              ^
// CHECK: 832:30: note: Object leaked: allocated object is returned from a method whose name ('NoCopyString') does not start with 'copy', 'mutableCopy', 'alloc' or 'new'.  This violates the naming convention rules given in the Memory Management Guide for Cocoa
// CHECK: 833:37: note: Method returns an Objective-C object with a +1 retain count
// CHECK: - (NSString*) noCopyString { return [[NSString alloc] init]; }
// CHECK:                                     ^
// CHECK: 833:30: note: Object returned to caller as an owning reference (single retain count transferred to caller)
// CHECK: - (NSString*) noCopyString { return [[NSString alloc] init]; }
// CHECK:                              ^
// CHECK: 833:30: note: Object leaked: allocated object is returned from a method whose name ('noCopyString') does not start with 'copy', 'mutableCopy', 'alloc' or 'new'.  This violates the naming convention rules given in the Memory Management Guide for Cocoa
// CHECK: 837:3: note: Calling 'NoCopyString'
// CHECK:   [x NoCopyString];
// CHECK:   ^
// CHECK: 832:37: note: Method returns an Objective-C object with a +1 retain count
// CHECK: - (NSString*) NoCopyString { return [[NSString alloc] init]; }
// CHECK:                                     ^
// CHECK: 837:3: note: Returning from 'NoCopyString'
// CHECK:   [x NoCopyString];
// CHECK:   ^
// CHECK: 841:1: note: Object leaked: allocated object is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 865:10: note: Method returns an Objective-C object with a +1 retain count
// CHECK:   return [[NSString alloc] init];
// CHECK:          ^
// CHECK: 865:3: note: Object returned to caller as an owning reference (single retain count transferred to caller)
// CHECK:   return [[NSString alloc] init];
// CHECK:   ^
// CHECK: 865:3: note: Object leaked: allocated object is returned from a method whose name (':') does not start with 'copy', 'mutableCopy', 'alloc' or 'new'.  This violates the naming convention rules given in the Memory Management Guide for Cocoa
// CHECK: 895:3: note: Method returns an Objective-C object with a +1 retain count
// CHECK:   [view createSnapshotImageOfType:str];
// CHECK:   ^
// CHECK: 899:1: note: Object leaked: allocated object is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 907:3: note: Method returns a Core Foundation object with a +1 retain count
// CHECK:   [context createCGLayerWithSize:size info:d];
// CHECK:   ^
// CHECK: 908:1: note: Object leaked: allocated object is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 916:3: note: Call to function 'IOBSDNameMatching' returns a Core Foundation object with a +1 retain count
// CHECK:   IOBSDNameMatching(masterPort, options, bsdName);
// CHECK:   ^
// CHECK: 917:1: note: Object leaked: allocated object is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 920:3: note: Call to function 'IOServiceMatching' returns a Core Foundation object with a +1 retain count
// CHECK:   IOServiceMatching(name);
// CHECK:   ^
// CHECK: 921:1: note: Object leaked: allocated object is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 924:3: note: Call to function 'IOServiceNameMatching' returns a Core Foundation object with a +1 retain count
// CHECK:   IOServiceNameMatching(name);
// CHECK:   ^
// CHECK: 925:1: note: Object leaked: allocated object is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 932:30: note: Call to function 'CreateDict' returns a Core Foundation object with a +1 retain count
// CHECK:   CFDictionaryRef matching = CreateDict();
// CHECK:                              ^
// CHECK: 933:3: note: Object released
// CHECK:   CFRelease(matching);
// CHECK:   ^
// CHECK: 934:3: note: Reference-counted object is used after it is released
// CHECK:   IOServiceAddNotification(masterPort, notificationType, matching,
// CHECK:   ^
// CHECK: 939:3: note: Call to function 'IORegistryEntryIDMatching' returns a Core Foundation object with a +1 retain count
// CHECK:   IORegistryEntryIDMatching(entryID);
// CHECK:   ^
// CHECK: 940:1: note: Object leaked: allocated object is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 944:3: note: Call to function 'IOOpenFirmwarePathMatching' returns a Core Foundation object with a +1 retain count
// CHECK:   IOOpenFirmwarePathMatching(masterPort, options, path);
// CHECK:   ^
// CHECK: 945:1: note: Object leaked: allocated object is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 948:30: note: Call to function 'CreateDict' returns a Core Foundation object with a +1 retain count
// CHECK:   CFDictionaryRef matching = CreateDict();
// CHECK:                              ^
// CHECK: 949:3: note: Object released
// CHECK:   IOServiceGetMatchingService(masterPort, matching);
// CHECK:   ^
// CHECK: 950:3: note: Reference-counted object is used after it is released
// CHECK:   CFRelease(matching);
// CHECK:   ^
// CHECK: 954:30: note: Call to function 'CreateDict' returns a Core Foundation object with a +1 retain count
// CHECK:   CFDictionaryRef matching = CreateDict();
// CHECK:                              ^
// CHECK: 955:3: note: Object released
// CHECK:   IOServiceGetMatchingServices(masterPort, matching, existing);
// CHECK:   ^
// CHECK: 956:3: note: Reference-counted object is used after it is released
// CHECK:   CFRelease(matching);
// CHECK:   ^
// CHECK: 962:30: note: Call to function 'CreateDict' returns a Core Foundation object with a +1 retain count
// CHECK:   CFDictionaryRef matching = CreateDict();
// CHECK:                              ^
// CHECK: 963:3: note: Object released
// CHECK:   IOServiceAddMatchingNotification(notifyPort, notificationType, matching, callback, refCon, notification);
// CHECK:   ^
// CHECK: 964:3: note: Reference-counted object is used after it is released
// CHECK:   CFRelease(matching);
// CHECK:   ^
// CHECK: 1005:22: note: Method returns an Objective-C object with a +1 retain count
// CHECK:   NSNumber *number = [[NSNumber alloc] initWithInt:5];
// CHECK:                      ^
// CHECK: 1007:3: note: Reference count decremented
// CHECK:   [number release];
// CHECK:   ^
// CHECK: 1008:3: note: Reference count incremented. The object now has a +1 retain count
// CHECK:   [number retain];
// CHECK:   ^
// CHECK: 1009:3: note: Object leaked: object allocated and stored into 'number' is not referenced later in this execution path and has a retain count of +1
// CHECK:   CFRelease(attrString);  
// CHECK:   ^
// CHECK: 1030:41: note: Call to function 'CGColorSpaceCreateDeviceRGB' returns a Core Foundation object with a +1 retain count
// CHECK:     CGGradientCreateWithColorComponents(CGColorSpaceCreateDeviceRGB(),
// CHECK:                                         ^
// CHECK: 1029:3: note: Object leaked: allocated object is not referenced later in this execution path and has a retain count of +1
// CHECK:   CGGradientRef myGradient =
// CHECK:   ^
// CHECK: 1052:40: note: Call to function 'CGColorSpaceCreateDeviceRGB' returns a Core Foundation object with a +1 retain count
// CHECK:    CGGradientCreateWithColorComponents(CGColorSpaceCreateDeviceRGB(), components, locations, num_locations);
// CHECK:                                        ^
// CHECK: 1051:3: note: Object leaked: allocated object is not referenced later in this execution path and has a retain count of +1
// CHECK:   CGGradientRef myGradient =
// CHECK:   ^
// CHECK: 1052:4: note: Call to function 'CGGradientCreateWithColorComponents' returns a Core Foundation object with a +1 retain count
// CHECK:    CGGradientCreateWithColorComponents(CGColorSpaceCreateDeviceRGB(), components, locations, num_locations);
// CHECK:    ^
// CHECK: 1056:1: note: Object leaked: object allocated and stored into 'myGradient' is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 1090:22: note: Method returns an Objective-C object with a +1 retain count
// CHECK:   NSNumber *number = [[NSNumber alloc] initWithInt:5];
// CHECK:                      ^
// CHECK: 1091:1: note: Object leaked: object allocated and stored into 'number' is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 1225:3: note: Call to function 'CGBitmapContextCreateWithData' returns a Core Foundation object with a +1 retain count
// CHECK:   CGBitmapContextCreateWithData(data, width, height, bitsPerComponent,
// CHECK:   ^
// CHECK: 1227:1: note: Object leaked: allocated object is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 1243:10: note: Method returns an Objective-C object with a +1 retain count
// CHECK:   id y = [x new_stuff];
// CHECK:          ^
// CHECK: 1244:1: note: Object leaked: object allocated and stored into 'y' is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 1264:22: note: Method returns an Objective-C object with a +1 retain count
// CHECK:   NSNumber *number = [[NSNumber alloc] initWithInt:5];
// CHECK:                      ^
// CHECK: 1265:1: note: Object leaked: object allocated and stored into 'number' is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 1274:3: note: The 'release' message should be sent to instances of class 'RDar7252064' and not the class directly
// CHECK:   [RDar7252064 release];
// CHECK:   ^
// CHECK: 1275:3: note: The 'retain' message should be sent to instances of class 'RDar7252064' and not the class directly
// CHECK:   [RDar7252064 retain];
// CHECK:   ^
// CHECK: 1276:3: note: The 'autorelease' message should be sent to instances of class 'RDar7252064' and not the class directly
// CHECK:   [RDar7252064 autorelease];
// CHECK:   ^
// CHECK: 1277:3: note: The 'drain' message should be sent to instances of class 'NSAutoreleasePool' and not the class directly
// CHECK:   [NSAutoreleasePool drain];
// CHECK:   ^
// CHECK: 1303:19: note: Method returns an Objective-C object with a +1 retain count
// CHECK:   NSString *str = [X returnsAnOwnedString];
// CHECK:                   ^
// CHECK: 1304:1: note: Object leaked: object allocated and stored into 'str' is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 1307:19: note: Method returns a Core Foundation object with a +1 retain count
// CHECK:   NSString *str = [X returnsAnOwnedCFString];
// CHECK:                   ^
// CHECK: 1308:1: note: Object leaked: object allocated and stored into 'str' is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 1312:20: note: Method returns an Objective-C object with a +1 retain count
// CHECK:   NSString *str2 = [X newStringNoAttr];
// CHECK:                    ^
// CHECK: 1313:1: note: Object leaked: object allocated and stored into 'str2' is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 1316:26: note: Method returns an Objective-C object with a +1 retain count
// CHECK:   TestOwnershipAttr *x = [TestOwnershipAttr alloc];
// CHECK:                          ^
// CHECK: 1317:1: note: Object leaked: object allocated and stored into 'x' is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 1320:26: note: Method returns an Objective-C object with a +1 retain count
// CHECK:   TestOwnershipAttr *x = [[TestOwnershipAttr alloc] pseudoInit];
// CHECK:                          ^
// CHECK: 1321:1: note: Object leaked: object allocated and stored into 'x' is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 1324:26: note: Method returns an Objective-C object with a +1 retain count
// CHECK:   TestOwnershipAttr *x = [[TestOwnershipAttr alloc] pseudoInit];// expected-warning{{leak}}
// CHECK:                          ^
// CHECK: 1326:1: note: Object leaked: object allocated and stored into 'x' is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 1356:10: note: Method returns an Objective-C object with a +1 retain count
// CHECK:   return [NSString alloc];
// CHECK:          ^
// CHECK: 1356:3: note: Object returned to caller as an owning reference (single retain count transferred to caller)
// CHECK:   return [NSString alloc];
// CHECK:   ^
// CHECK: 1356:3: note: Object leaked: allocated object is returned from a method that is annotated as NS_RETURNS_NOT_RETAINED
// CHECK: 1389:26: note: Calling 'returnsCFRetainedAsCF'
// CHECK:   return (CFDateRef)[(id)[self returnsCFRetainedAsCF] autorelease];
// CHECK:                          ^
// CHECK: 1381:10: note: Calling 'returnsRetainedCFDate'
// CHECK:   return returnsRetainedCFDate(); // No leak.
// CHECK:          ^
// CHECK: 1372:10: note: Call to function 'CFDateCreate' returns a Core Foundation object with a +1 retain count
// CHECK:   return CFDateCreate(0, CFAbsoluteTimeGetCurrent());
// CHECK:          ^
// CHECK: 1381:10: note: Returning from 'returnsRetainedCFDate'
// CHECK:   return returnsRetainedCFDate(); // No leak.
// CHECK:          ^
// CHECK: 1389:26: note: Returning from 'returnsCFRetainedAsCF'
// CHECK:   return (CFDateRef)[(id)[self returnsCFRetainedAsCF] autorelease];
// CHECK:                          ^
// CHECK: 1389:21: note: Object sent -autorelease message
// CHECK:   return (CFDateRef)[(id)[self returnsCFRetainedAsCF] autorelease];
// CHECK:                     ^
// CHECK: 1389:3: note: Object returned to caller with a +0 retain count
// CHECK:   return (CFDateRef)[(id)[self returnsCFRetainedAsCF] autorelease];
// CHECK:   ^
// CHECK: 1389:3: note: Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected
// CHECK: 1393:20: note: Calling 'returnsRetainedCFDate'
// CHECK:   return (NSDate*) returnsRetainedCFDate();
// CHECK:                    ^
// CHECK: 1372:10: note: Call to function 'CFDateCreate' returns a Core Foundation object with a +1 retain count
// CHECK:   return CFDateCreate(0, CFAbsoluteTimeGetCurrent());
// CHECK:          ^
// CHECK: 1393:20: note: Returning from 'returnsRetainedCFDate'
// CHECK:   return (NSDate*) returnsRetainedCFDate();
// CHECK:                    ^
// CHECK: 1393:3: note: Object returned to caller as an owning reference (single retain count transferred to caller)
// CHECK:   return (NSDate*) returnsRetainedCFDate();
// CHECK:   ^
// CHECK: 1393:3: note: Object leaked: allocated object is returned from a method whose name ('alsoReturnsRetained') does not start with 'copy', 'mutableCopy', 'alloc' or 'new'.  This violates the naming convention rules given in the Memory Management Guide for Cocoa
// CHECK: 1397:10: note: Calling 'returnsRetainedCFDate'
// CHECK:   return returnsRetainedCFDate();
// CHECK:          ^
// CHECK: 1372:10: note: Call to function 'CFDateCreate' returns a Core Foundation object with a +1 retain count
// CHECK:   return CFDateCreate(0, CFAbsoluteTimeGetCurrent());
// CHECK:          ^
// CHECK: 1397:10: note: Returning from 'returnsRetainedCFDate'
// CHECK:   return returnsRetainedCFDate();
// CHECK:          ^
// CHECK: 1397:3: note: Object returned to caller as an owning reference (single retain count transferred to caller)
// CHECK:   return returnsRetainedCFDate();
// CHECK:   ^
// CHECK: 1397:3: note: Object leaked: allocated object is returned from a method whose name ('alsoReturnsRetainedAsCF') does not start with 'copy', 'mutableCopy', 'alloc' or 'new'.  This violates the naming convention rules given in the Memory Management Guide for Cocoa
// CHECK: 1418:23: note: Call to function 'CFNumberCreate' returns a Core Foundation object with a +1 retain count
// CHECK:   CFNumberRef value = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &z);
// CHECK:                       ^
// CHECK: 1419:1: note: Object leaked: object allocated and stored into 'value' is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 1429:23: note: Call to function 'CFNumberCreate' returns a Core Foundation object with a +1 retain count
// CHECK:   CFNumberRef value = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &z);
// CHECK:                       ^
// CHECK: 1430:3: note: Taking false branch
// CHECK:   if (x)
// CHECK:   ^
// CHECK: 1432:1: note: Object leaked: object allocated and stored into 'value' is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 1452:22: note: Method returns an Objective-C object with a +1 retain count
// CHECK:   NSNumber *number = [[NSNumber alloc] initWithInt:5];
// CHECK:                      ^
// CHECK: 1453:3: note: Object leaked: object allocated and stored into 'number' is not referenced later in this execution path and has a retain count of +1
// CHECK:   ^{}();
// CHECK:   ^
// CHECK: 1473:22: note: Method returns an Objective-C object with a +1 retain count
// CHECK:   NSNumber *number = [[NSNumber alloc] initWithInt:5];
// CHECK:                      ^
// CHECK: 1474:3: note: Calling anonymous block
// CHECK:   ^(NSObject *o){ [o retain]; }(number);
// CHECK:   ^
// CHECK: 1474:19: note: Reference count incremented. The object now has a +2 retain count
// CHECK:   ^(NSObject *o){ [o retain]; }(number);
// CHECK:                   ^
// CHECK: 1474:3: note: Returning to caller
// CHECK:   ^(NSObject *o){ [o retain]; }(number);
// CHECK:   ^
// CHECK: 1475:1: note: Object leaked: object allocated and stored into 'number' is not referenced later in this execution path and has a retain count of +2
// CHECK: }
// CHECK: ^
// CHECK: 1528:5: note: Loop condition is true.  Entering loop body
// CHECK:     while (error_to_dump != ((void*)0)) {
// CHECK:     ^
// CHECK: 1531:16: note: Call to function 'CFErrorCopyUserInfo' returns a Core Foundation object with a +1 retain count
// CHECK:         info = CFErrorCopyUserInfo(error_to_dump);
// CHECK:                ^
// CHECK: 1533:13: note: Assuming 'info' is not equal to null
// CHECK:         if (info != ((void*)0)) {
// CHECK:             ^
// CHECK: 1533:9: note: Taking true branch
// CHECK:         if (info != ((void*)0)) {
// CHECK:         ^
// CHECK: 1528:5: note: Loop condition is false. Execution jumps to the end of the function
// CHECK:     while (error_to_dump != ((void*)0)) {
// CHECK:     ^
// CHECK: 1538:1: note: Object leaked: object allocated and stored into 'info' is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 1581:10: note: Call to function 'CFArrayCreateMutable' returns a Core Foundation object with a +1 retain count
// CHECK:   return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
// CHECK:          ^
// CHECK: 1581:3: note: Object returned to caller as an owning reference (single retain count transferred to caller)
// CHECK:   return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
// CHECK:   ^
// CHECK: 1581:3: note: Object leaked: allocated object is returned from a function whose name ('camelcase_createno') does not contain 'Copy' or 'Create'.  This violates the naming convention rules given in the Memory Management Guide for Core Foundation
// CHECK: 1589:10: note: Call to function 'CFArrayCreateMutable' returns a Core Foundation object with a +1 retain count
// CHECK:   return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
// CHECK:          ^
// CHECK: 1589:3: note: Object returned to caller as an owning reference (single retain count transferred to caller)
// CHECK:   return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
// CHECK:   ^
// CHECK: 1589:3: note: Object leaked: allocated object is returned from a function whose name ('camelcase_copying') does not contain 'Copy' or 'Create'.  This violates the naming convention rules given in the Memory Management Guide for Core Foundation
// CHECK: 1610:10: note: Call to function 'CFArrayCreateMutable' returns a Core Foundation object with a +1 retain count
// CHECK:   return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
// CHECK:          ^
// CHECK: 1610:3: note: Object returned to caller as an owning reference (single retain count transferred to caller)
// CHECK:   return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
// CHECK:   ^
// CHECK: 1610:3: note: Object leaked: allocated object is returned from a function whose name ('camel_creat') does not contain 'Copy' or 'Create'.  This violates the naming convention rules given in the Memory Management Guide for Core Foundation
// CHECK: 1622:10: note: Call to function 'CFArrayCreateMutable' returns a Core Foundation object with a +1 retain count
// CHECK:   return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
// CHECK:          ^
// CHECK: 1622:3: note: Object returned to caller as an owning reference (single retain count transferred to caller)
// CHECK:   return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks);
// CHECK:   ^
// CHECK: 1622:3: note: Object leaked: allocated object is returned from a function whose name ('camel_copymachine') does not contain 'Copy' or 'Create'.  This violates the naming convention rules given in the Memory Management Guide for Core Foundation
// CHECK: 1643:24: note: Call to function 'CFDateCreate' returns a Core Foundation object with a +1 retain count
// CHECK:   CFTypeRef vals[] = { CFDateCreate(0, t) };
// CHECK:                        ^
// CHECK: 1644:1: note: Object leaked: object allocated and stored into 'vals' is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 1670:22: note: Method returns an Objective-C object with a +1 retain count
// CHECK:   NSString *string = [[NSString alloc] initWithUTF8String:"foo"];
// CHECK:                      ^
// CHECK: 1671:3: note: Object released
// CHECK:   [string release];
// CHECK:   ^
// CHECK: 1673:27: note: Reference-counted object is used after it is released
// CHECK:   NSString *otherString = [string stringByAppendingString:@"bar"];
// CHECK:                           ^
// CHECK: 1794:22: note: Method returns an Objective-C object with a +1 retain count
// CHECK:         NSArray *a = [[NSArray alloc] initWithObjects:o, (void*)0];
// CHECK:                      ^
// CHECK: 1797:9: note: Object leaked: object allocated and stored into 'a' is not referenced later in this execution path and has a retain count of +1
// CHECK:         [o description];
// CHECK:         ^
// CHECK: 1803:23: note: Method returns an Objective-C object with a +1 retain count
// CHECK:         NSArray *a2 = [[NSArray alloc] initWithArray:a1];
// CHECK:                       ^
// CHECK: 1806:9: note: Object leaked: object allocated and stored into 'a2' is not referenced later in this execution path and has a retain count of +1
// CHECK:         [o description];
// CHECK:         ^
// CHECK: 1811:24: note: NSArray literal is an object with a +0 retain count
// CHECK:         NSArray *a3 = [@[o] retain];
// CHECK:                        ^
// CHECK: 1811:23: note: Reference count incremented. The object now has a +1 retain count
// CHECK:         NSArray *a3 = [@[o] retain];
// CHECK:                       ^
// CHECK: 1814:9: note: Object leaked: object allocated and stored into 'a3' is not referenced later in this execution path and has a retain count of +1
// CHECK:         [o description];
// CHECK:         ^
// CHECK: 1819:22: note: Method returns an Objective-C object with a +1 retain count
// CHECK:         NSArray *a = [[NSArray alloc] initWithArray:@[o]];
// CHECK:                      ^
// CHECK: 1823:9: note: Object leaked: object allocated and stored into 'a' is not referenced later in this execution path and has a retain count of +1
// CHECK:         [o description];
// CHECK:         ^
// CHECK: 1828:28: note: NSDictionary literal is an object with a +0 retain count
// CHECK:         NSDictionary *a = [@{o : o} retain];
// CHECK:                            ^
// CHECK: 1828:27: note: Reference count incremented. The object now has a +1 retain count
// CHECK:         NSDictionary *a = [@{o : o} retain];
// CHECK:                           ^
// CHECK: 1832:9: note: Object leaked: object allocated and stored into 'a' is not referenced later in this execution path and has a retain count of +1
// CHECK:         [o description];
// CHECK:         ^
// CHECK: 1837:15: note: NSNumber literal is an object with a +0 retain count
// CHECK:   id value = [@1 retain];
// CHECK:               ^
// CHECK: 1837:14: note: Reference count incremented. The object now has a +1 retain count
// CHECK:   id value = [@1 retain];
// CHECK:              ^
// CHECK: 1839:1: note: Object leaked: object allocated and stored into 'value' is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 1842:15: note: NSNumber boxed expression produces an object with a +0 retain count
// CHECK:   id value = [@(x) retain];
// CHECK:               ^
// CHECK: 1842:14: note: Reference count incremented. The object now has a +1 retain count
// CHECK:   id value = [@(x) retain];
// CHECK:              ^
// CHECK: 1846:3: note: Object leaked: object allocated and stored into 'value' is not referenced later in this execution path and has a retain count of +1
// CHECK:   [value description];
// CHECK:   ^
// CHECK: 1845:12: note: NSString boxed expression produces an object with a +0 retain count
// CHECK:   value = [@(y) retain];
// CHECK:            ^
// CHECK: 1845:11: note: Reference count incremented. The object now has a +1 retain count
// CHECK:   value = [@(y) retain];
// CHECK:           ^
// CHECK: 1847:1: note: Object leaked: object allocated and stored into 'value' is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
// CHECK: 1854:5: note: Taking false branch
// CHECK:     if(y > 2)
// CHECK:     ^
// CHECK: 1857:21: note: Method returns an Objective-C object with a +1 retain count
// CHECK:       printString = [[NSString alloc] init];
// CHECK:                     ^
// CHECK: 1859:5: note: Object released
// CHECK:     [printString release];
// CHECK:     ^
// CHECK: 1860:5: note: Reference-counted object is used after it is released
// CHECK:     NSLog(@"Again: %@", printString);
// CHECK:     ^
// CHECK: 1887:19: note: NSArray literal is an object with a +0 retain count
// CHECK:   id unretained = @[]; // +0
// CHECK:                   ^
// CHECK: 1888:3: note: Incorrect decrement of the reference count of an object that is not owned at this point by the caller
// CHECK:   consumeAndStopTracking(unretained, ^{});
// CHECK:   ^
// CHECK: 1900:19: note: NSArray literal is an object with a +0 retain count
// CHECK:   id unretained = @[]; // +0
// CHECK:                   ^
// CHECK: 1901:3: note: Incorrect decrement of the reference count of an object that is not owned at this point by the caller
// CHECK:   CFConsumeAndStopTracking((CFTypeRef)unretained, ^{});
// CHECK:   ^
// CHECK: 1913:16: note: Call to function 'CreateMyCFType' returns a Core Foundation object with a +1 retain count
// CHECK:   MyCFType x = CreateMyCFType();
// CHECK:                ^
// CHECK: 1914:1: note: Object leaked: object allocated and stored into 'x' is not referenced later in this execution path and has a retain count of +1
// CHECK: }
// CHECK: ^
