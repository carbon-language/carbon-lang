// RUN: %clang_cc1 -fsyntax-only -verify %s

//===----------------------------------------------------------------------===//
// The following code is reduced using delta-debugging from
// Foundation.h (Mac OS X).
//
// It includes the basic definitions for the test cases below.
// Not including Foundation.h directly makes this test case both svelt and
// portable to non-Mac platforms.
//===----------------------------------------------------------------------===//

typedef signed char BOOL;
typedef unsigned int NSUInteger;
@class NSString, Protocol;
extern void NSLog(NSString *format, ...) __attribute__((format(__NSString__, 1, 2)));
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@protocol NSCopying  - (id)copyWithZone:(NSZone *)zone; @end
@protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone; @end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder; @end
@interface NSObject <NSObject> {} @end
typedef float CGFloat;
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>    - (NSUInteger)length; @end
@interface NSSimpleCString : NSString {} @end
@interface NSConstantString : NSSimpleCString @end
extern void *_NSConstantStringClassReference;

typedef const struct __CFString * CFStringRef;
extern void CFStringCreateWithFormat(CFStringRef format, ...) __attribute__((format(CFString, 1, 2)));

int printf(const char * restrict, ...) ;

//===----------------------------------------------------------------------===//
// Test cases.
//===----------------------------------------------------------------------===//

void check_nslog(unsigned k) {
  NSLog(@"%d%%", k); // no-warning
  NSLog(@"%s%lb%d", "unix", 10,20); // expected-warning {{invalid conversion specifier 'b'}}
  NSLog(@"%@", "a"); // expected-warning {{conversion specifies type 'id' but the argument has type 'char *'}}
}

// Check type validation
extern void NSLog2(int format, ...) __attribute__((format(__NSString__, 1, 2))); // expected-error {{format argument not an NSString}}
extern void CFStringCreateWithFormat2(int *format, ...) __attribute__((format(CFString, 1, 2))); // expected-error {{format argument not a CFString}}

// <rdar://problem/7068334> - Catch use of long long with int arguments.
void rdar_7068334() {
  long long test = 500;  
  printf("%i ",test); // expected-warning{{conversion specifies type 'int' but the argument has type 'long long'}}
  NSLog(@"%i ",test); // expected-warning{{conversion specifies type 'int' but the argument has type 'long long'}}
}

// <rdar://problem/7697748>
void rdar_7697748() {
  NSLog(@"%@!"); // expected-warning{{more '%' conversions than data arguments}}
}

@protocol Foo;

void test_p_conversion_with_objc_pointer(id x, id<Foo> y) {
  printf("%p", x); // no-warning
  printf("%p", y); // no-warning
}

