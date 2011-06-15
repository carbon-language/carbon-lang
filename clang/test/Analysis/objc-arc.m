// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -analyze -analyzer-checker=core -analyzer-checker=deadcode -analyzer-store=region -verify -fblocks  -analyzer-opt-analyze-nested-blocks -fobjc-nonfragile-abi -fobjc-arc %s

typedef signed char BOOL;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;

@protocol NSObject
- (BOOL)isEqual:(id)object;
@end
@protocol NSCopying
- (id)copyWithZone:(NSZone *)zone;
@end
@protocol NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@interface NSObject <NSObject> {}
+ (id)alloc;
@end
typedef const struct __CFAllocator * CFAllocatorRef;
extern const CFAllocatorRef kCFAllocatorDefault;
typedef double CFTimeInterval;
typedef CFTimeInterval CFAbsoluteTime;
extern CFAbsoluteTime CFAbsoluteTimeGetCurrent(void);
typedef const struct __CFDate * CFDateRef;
extern CFDateRef CFDateCreate(CFAllocatorRef allocator, CFAbsoluteTime at);

typedef const void* objc_objectptr_t;
__attribute__((ns_returns_retained)) id objc_retainedObject(objc_objectptr_t __attribute__((cf_consumed)) pointer);
__attribute__((ns_returns_not_retained)) id objc_unretainedObject(objc_objectptr_t pointer);

// Test the analyzer is working at all.
void test_working() {
  int *p = 0;
  *p = 0xDEADBEEF; // expected-warning {{null}}
}

// Test that in ARC mode that blocks are correctly automatically copied
// and not flagged as warnings by the analyzer.
typedef void (^Block)(void);
void testblock_bar(int x);

Block testblock_foo(int x) {
  Block b = ^{ testblock_bar(x); };
  return b; // no-warning
}

Block testblock_baz(int x) {
  return ^{ testblock_bar(x); }; // no-warning
}

Block global_block;

void testblock_qux(int x) {
  global_block = ^{ testblock_bar(x); }; // no-warning
}

// Test that Objective-C pointers are null initialized.
void test_nil_initialized() {
  id x;
  if (x == 0)
    return;
  int *p = 0;
  *p = 0xDEADBEEF; // no-warning
}

// Test that we don't flag leaks of Objective-C objects.
void test_alloc() {
  [NSObject alloc]; // no-warning
}

// Test that CF allocations are still caught as leaks.
void test_cf_leak() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t); // expected-warning {{Potential leak}}
  (void) date;
}

// Test that 'init' methods do not try to claim ownerhip of an *unowned* allocated object
// in ARC mode.
@interface RDar9424890_A :  NSObject
- (id)initWithCleaner:(int)pop mop:(NSString *)mop ;
- (RDar9424890_A *)rdar9424890:(NSString *)identifier;
@end
@interface RDar9424890_B : NSObject
@end
@implementation RDar9424890_B
- (RDar9424890_A *)obj:(RDar9424890_A *)obj {
  static NSString *WhizFiz = @"WhizFiz";
  RDar9424890_A *cell = [obj rdar9424890:WhizFiz];
  if (cell == ((void*)0)) {
    cell = [[RDar9424890_A alloc] initWithCleaner:0 mop:WhizFiz]; // no-warning
  }
  return cell;
}
@end

// Test that dead store checking works in the prescence of "cleanups" in the AST.
void rdar9424882() {
  id x = [NSObject alloc]; // expected-warning {{Value stored to 'x' during its initialization is never read}}
}

// Test 
typedef const void *CFTypeRef;
typedef const struct __CFString *CFStringRef;

@interface NSString
- (id) self;
@end

CFTypeRef CFCreateSomething();
CFStringRef CFCreateString();
CFTypeRef CFGetSomething();
CFStringRef CFGetString();

id CreateSomething();
NSString *CreateNSString();

void from_cf() {
  id obj1 = (__bridge_transfer id)CFCreateSomething(); // expected-warning{{never read}}
  id obj2 = (__bridge_transfer NSString*)CFCreateString();
  [obj2 self]; // Add a use, to show we can use the object after it has been transfered.
  id obj3 = (__bridge id)CFGetSomething();
  [obj3 self]; // Add a use, to show we can use the object after it has been bridged.
  id obj4 = (__bridge NSString*)CFGetString(); // expected-warning{{never read}}
  id obj5 = (__bridge id)CFCreateSomething(); // expected-warning{{never read}} expected-warning{{leak}}
  id obj6 = (__bridge NSString*)CFCreateString(); // expected-warning{{never read}} expected-warning{{leak}}
}

void to_cf(id obj) {
  CFTypeRef cf1 = (__bridge_retained CFTypeRef)CreateSomething(); // expected-warning{{never read}}
  CFStringRef cf2 = (__bridge_retained CFStringRef)CreateNSString(); // expected-warning{{never read}}
  CFTypeRef cf3 = (__bridge CFTypeRef)CreateSomething(); // expected-warning{{never read}}
  CFStringRef cf4 = (__bridge CFStringRef)CreateNSString();  // expected-warning{{never read}}
}

void test_objc_retainedObject() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t);
  id x = objc_retainedObject(date);
  (void) x;
}

void test_objc_unretainedObject() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t);  // expected-warning {{Potential leak}}
  id x = objc_unretainedObject(date);
  (void) x;
}

