// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -verify %s
// RUN: not %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// rdar://11913153

typedef const struct __CFString * CFStringRef;
typedef struct __CFString * CFMutableStringRef;
typedef signed long CFIndex;
typedef const struct __CFAllocator * CFAllocatorRef;

extern const CFStringRef kCFBundleNameKey;

@protocol NSCopying @end

@interface NSDictionary
- (id)objectForKeyedSubscript:(id<NSCopying>)key;
@end

#pragma clang arc_cf_code_audited begin
extern
CFMutableStringRef CFStringCreateMutable(CFAllocatorRef alloc, CFIndex maxLength);
#pragma clang arc_cf_code_audited end

typedef const void * CFTypeRef;

id CFBridgingRelease(CFTypeRef __attribute__((cf_consumed)) X);

@interface NSMutableString @end

NSMutableString *test() {
  NSDictionary *infoDictionary;
  infoDictionary[kCFBundleNameKey] = 0; // expected-error {{indexing expression is invalid because subscript type 'CFStringRef' (aka 'const struct __CFString *') is not an integral or Objective-C pointer type}}
  return infoDictionary[CFStringCreateMutable(((void*)0), 100)]; // expected-error {{indexing expression is invalid because subscript type 'CFMutableStringRef' (aka 'struct __CFString *') is not an integral or Objective-C pointer type}} \
                                       // expected-error {{implicit conversion of C pointer type 'CFMutableStringRef' (aka 'struct __CFString *') to Objective-C pointer type '__strong id<NSCopying>' requires a bridged cast}} \
                                        // expected-note {{use CFBridgingRelease call to transfer ownership of a +1 'CFMutableStringRef' (aka 'struct __CFString *') into ARC}}
					
}

// CHECK: fix-it:"{{.*}}":{32:25-32:25}:"CFBridgingRelease("
// CHECK: fix-it:"{{.*}}":{32:63-32:63}:")"
