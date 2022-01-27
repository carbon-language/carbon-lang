// RUN: %clang_analyze_cc1 -fblocks -fobjc-arc -verify %s -Wno-objc-root-class \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=alpha.core.StackAddressAsyncEscape \
// RUN:   -analyzer-checker=nullability \
// RUN:   -analyzer-checker=osx

#include "Inputs/system-header-simulator-for-nullability.h"
#include "os_object_base.h"

struct OSIterator : public OSObject {
  static const OSMetaClass * const metaClass;
};

@interface TestObject : NSObject
- (int *_Nonnull)returnsNonnull;
- (int *_Nullable)returnsNullable;
- (int *)returnsUnspecified;
- (void)takesNonnull:(int *_Nonnull)p;
- (void)takesNullable:(int *_Nullable)p;
- (void)takesUnspecified:(int *)p;
@property(readonly, strong) NSString *stuff;
@end

TestObject * getUnspecifiedTestObject();
TestObject *_Nonnull getNonnullTestObject();
TestObject *_Nullable getNullableTestObject();

int getRandom();

typedef struct Dummy { int val; } Dummy;

void takesNullable(Dummy *_Nullable);
void takesNonnull(Dummy *_Nonnull);
void takesUnspecified(Dummy *);

Dummy *_Nullable returnsNullable();
Dummy *_Nonnull returnsNonnull();
Dummy *returnsUnspecified();
int *_Nullable returnsNullableInt();

template <typename T> T *eraseNullab(T *p) { return p; }

void takesAttrNonnull(Dummy *p) __attribute((nonnull(1)));

void testBasicRules() {
  // FIXME: None of these should be tied to a modeling checker.
  Dummy *p = returnsNullable();
  int *ptr = returnsNullableInt();
  // Make every dereference a different path to avoid sinks after errors.
  switch (getRandom()) {
  case 0: {
    Dummy &r = *p; // expected-warning {{Nullable pointer is dereferenced [nullability.NullableDereferenced]}}
  } break;
  case 1: {
    int b = p->val; // expected-warning {{Nullable pointer is dereferenced [nullability.NullableDereferenced]}}
  } break;
  case 2: {
    int stuff = *ptr; // expected-warning {{Nullable pointer is dereferenced [nullability.NullableDereferenced]}}
  } break;
  case 3:
    takesNonnull(p); // expected-warning {{Nullable pointer is passed to a callee that requires a non-null 1st parameter [nullability.NullablePassedToNonnull]}}
    break;
  case 4: {
    Dummy d;
    takesNullable(&d);
    Dummy dd(d);
    break;
  }
  case 5:
    takesAttrNonnull(p); // expected-warning {{Nullable pointer is passed to a callee that requires a non-null [nullability.NullableDereferenced]}}
    break;
  default: { Dummy d = *p; } break; // expected-warning {{Nullable pointer is dereferenced [nullability.NullableDereferenced]}}
  }
  if (p) {
    takesNonnull(p);
    if (getRandom()) {
      Dummy &r = *p;
    } else {
      int b = p->val;
    }
  }
  Dummy *q = 0;
  if (getRandom()) {
    takesNullable(q);
  // FIXME: This shouldn't be tied to a modeling checker.
    takesNonnull(q); // expected-warning {{Null passed to a callee that requires a non-null 1st parameter [nullability.NullPassedToNonnull]}}
  }
  Dummy a;
  Dummy *_Nonnull nonnull = &a;
  // FIXME: This shouldn't be tied to a modeling checker.
  nonnull = q; // expected-warning {{Null assigned to a pointer which is expected to have non-null value [nullability.NullPassedToNonnull]}}
  q = &a;
  takesNullable(q);
  takesNonnull(q);
}

typedef int NSInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@class NSDictionary;
@interface NSError : NSObject <NSCopying, NSCoding> {}
+ (id)errorWithDomain:(NSString *)domain code:(NSInteger)code userInfo:(NSDictionary *)dict;
@end

struct __CFError {};
typedef struct __CFError* CFErrorRef;

void foo(CFErrorRef* error) { // expected-warning{{Function accepting CFErrorRef* should have a non-void return value to indicate whether or not an error occurred [osx.coreFoundation.CFError]}}
  // FIXME: This shouldn't be tied to a modeling checker.
  *error = 0; // expected-warning {{Potential null dereference.  According to coding standards documented in CoreFoundation/CFError.h the parameter may be null [osx.coreFoundation.CFError]}}
}

@interface A
- (void)myMethodWhichMayFail:(NSError **)error;
@end

@implementation A
- (void)myMethodWhichMayFail:(NSError **)error {                  // expected-warning {{Method accepting NSError** should have a non-void return value to indicate whether or not an error occurred [osx.cocoa.NSError]}}
  *error = [NSError errorWithDomain:@"domain" code:1 userInfo:0]; // expected-warning {{Potential null dereference.  According to coding standards in 'Creating and Returning NSError Objects' the parameter may be null [osx.cocoa.NSError]}}
}
@end

bool write_into_out_param_on_success(OS_RETURNS_RETAINED OSObject **obj);

void use_out_param_leak() {
  OSObject *obj;
  // FIXME: This shouldn't be tied to a modeling checker.
  write_into_out_param_on_success(&obj); // expected-warning{{Potential leak of an object stored into 'obj' [osx.cocoa.RetainCount]}}
}

typedef struct dispatch_queue_s *dispatch_queue_t;
typedef void (^dispatch_block_t)(void);
void dispatch_async(dispatch_queue_t queue, dispatch_block_t block);
typedef long dispatch_once_t;
void dispatch_once(dispatch_once_t *predicate, dispatch_block_t block);
typedef long dispatch_time_t;
void dispatch_after(dispatch_time_t when, dispatch_queue_t queue, dispatch_block_t block);
void dispatch_barrier_sync(dispatch_queue_t queue, dispatch_block_t block);

extern dispatch_queue_t queue;
extern dispatch_once_t *predicate;
extern dispatch_time_t when;

dispatch_block_t get_leaking_block() {
  int leaked_x = 791;
  int *p = &leaked_x;
  return ^void(void) {
    *p = 1;
  };
  // expected-warning@-3 {{Address of stack memory associated with local variable 'leaked_x' \
is captured by a returned block [core.StackAddressEscape]}}
}

void test_returned_from_func_block_async() {
  dispatch_async(queue, get_leaking_block());
  // expected-warning@-1 {{Address of stack memory associated with local variable 'leaked_x' \
is captured by an asynchronously-executed block [alpha.core.StackAddressAsyncEscape]}}
}
