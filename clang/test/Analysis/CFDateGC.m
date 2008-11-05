// RUN: clang -checker-cfref -verify -fobjc-gc %s

//===----------------------------------------------------------------------===//
// The following code is reduced using delta-debugging from
// Foundation.h and CoreFoundation.h (Mac OS X).
//
// It includes the basic definitions for the test cases below.
// Not directly including [Core]Foundation.h directly makes this test case 
// both svelte and portable to non-Mac platforms.
//===----------------------------------------------------------------------===//

typedef const void * CFTypeRef;
typedef const struct __CFAllocator * CFAllocatorRef;
typedef double CFTimeInterval;
typedef CFTimeInterval CFAbsoluteTime;
typedef const struct __CFDate * CFDateRef;
extern CFDateRef CFDateCreate(CFAllocatorRef allocator, CFAbsoluteTime at);
typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
static __inline__ __attribute__((always_inline)) id NSMakeCollectable(CFTypeRef cf) {}
@protocol NSObject  - (BOOL)isEqual:(id)object; - (oneway void)release; @end
extern id NSAllocateObject(Class aClass, NSUInteger extraBytes, NSZone *zone);
CFTypeRef CFMakeCollectable(CFTypeRef cf);

//===----------------------------------------------------------------------===//
// Test cases.
//===----------------------------------------------------------------------===//

CFAbsoluteTime f1() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t);
  CFRetain(date);
  [NSMakeCollectable(date) release];
  CFDateGetAbsoluteTime(date); // no-warning
  CFRelease(date);
  t = CFDateGetAbsoluteTime(date);   // expected-warning{{Reference-counted object is used after it is released.}}
  return t;
}

CFAbsoluteTime f1b() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t);
  CFRetain(date);
  [(id) CFMakeCollectable(date) release];
  CFDateGetAbsoluteTime(date); // no-warning
  t = CFDateGetAbsoluteTime(date);  // no-warning
  CFRelease(date); // no-warning
  return t;
}


