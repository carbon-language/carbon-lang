// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
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

extern
CFMutableStringRef CFStringCreateMutable(CFAllocatorRef alloc, CFIndex maxLength);

typedef const void * CFTypeRef;

id CFBridgingRelease(CFTypeRef __attribute__((cf_consumed)) X);

@interface NSMutableString @end

NSMutableString *test() {
  NSDictionary *infoDictionary;
  infoDictionary[kCFBundleNameKey] = 0; // expected-error {{indexing expression is invalid because subscript type 'CFStringRef' (aka 'const struct __CFString *') is not an integral or Objective-C pointer type}} \
                                        // expected-error {{implicit conversion of C pointer type 'CFStringRef' (aka 'const struct __CFString *') to Objective-C pointer type '__strong id<NSCopying>' requires a bridged cast}} \
                                        // expected-note {{use __bridge to convert directly (no change in ownership)}} \
                                        // expected-note {{use CFBridgingRelease call to transfer ownership of a +1 'CFStringRef' (aka 'const struct __CFString *') into ARC}}
  return infoDictionary[CFStringCreateMutable(((void*)0), 100)]; // expected-error {{indexing expression is invalid because subscript type 'CFMutableStringRef' (aka 'struct __CFString *') is not an integral or Objective-C pointer type}} \
                                       // expected-error {{implicit conversion of C pointer type 'CFMutableStringRef' (aka 'struct __CFString *') to Objective-C pointer type '__strong id<NSCopying>' requires a bridged cast}} \
                                        // expected-note {{use __bridge to convert directly (no change in ownership)}} \
                                        // expected-note {{use CFBridgingRelease call to transfer ownership of a +1 'CFMutableStringRef' (aka 'struct __CFString *') into ARC}}
					
}

// CHECK: fix-it:"{{.*}}":{29:18-29:18}:"(__bridge __strong id<NSCopying>)("
// CHECK: fix-it:"{{.*}}":{29:34-29:34}:")"
// CHECK: fix-it:"{{.*}}":{29:18-29:18}:"CFBridgingRelease("
// CHECK: fix-it:"{{.*}}":{29:34-29:34}:")"
// CHECK: fix-it:"{{.*}}":{33:25-33:25}:"(__bridge __strong id<NSCopying>)("
// CHECK: fix-it:"{{.*}}":{33:63-33:63}:")"
// CHECK: fix-it:"{{.*}}":{33:25-33:25}:"CFBridgingRelease("
// CHECK: fix-it:"{{.*}}":{33:63-33:63}:")"
