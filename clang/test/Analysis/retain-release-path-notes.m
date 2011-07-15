// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -analyze -analyzer-checker=core,osx.coreFoundation.CFRetainRelease,osx.cocoa.ClassRelease -analyzer-store=basic -analyzer-output=text -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -analyze -analyzer-checker=core,osx.coreFoundation.CFRetainRelease,osx.cocoa.ClassRelease -analyzer-store=region -analyzer-output=text -verify %s

/***
This file is for testing the path-sensitive notes for retain/release errors.
Its goal is to have simple branch coverage of any path-based diagnostics,
not to actually check all possible retain/release errors.

This file includes notes that only appear in a ref-counted analysis. 
GC-specific notes should go in retain-release-path-notes-gc.m.
***/

@interface NSObject
+ (id)alloc;
- (id)init;
- (void)dealloc;

- (Class)class;

- (id)retain;
- (void)release;
- (void)autorelease;
@end

@interface Foo : NSObject
- (id)methodWithValue;
@property(retain) id propertyValue;
@end

typedef struct CFType *CFTypeRef;
CFTypeRef CFRetain(CFTypeRef);
void CFRelease(CFTypeRef);

id NSMakeCollectable(CFTypeRef);
CFTypeRef CFMakeCollectable(CFTypeRef);

CFTypeRef CFCreateSomething();
CFTypeRef CFGetSomething();


void creationViaAlloc () {
  id leaked = [[NSObject alloc] init]; // expected-warning{{leak}} expected-note{{Method returns an Objective-C object with a +1 retain count}}
  return; // expected-note{{Object leaked: object allocated and stored into 'leaked' is not referenced later in this execution path and has a retain count of +1}}
}

void creationViaCFCreate () {
  CFTypeRef leaked = CFCreateSomething(); // expected-warning{{leak}} expected-note{{Call to function 'CFCreateSomething' returns a Core Foundation object with a +1 retain count}}
  return; // expected-note{{Object leaked: object allocated and stored into 'leaked' is not referenced later in this execution path and has a retain count of +1}}
}

void acquisitionViaMethod (Foo *foo) {
  id leaked = [foo methodWithValue]; // expected-warning{{leak}} expected-note{{Method returns an Objective-C object with a +0 retain count}}
  [leaked retain]; // expected-note{{Reference count incremented. The object now has a +1 retain count}}
  [leaked retain]; // expected-note{{Reference count incremented. The object now has a +2 retain count}}
  [leaked release]; // expected-note{{Reference count decremented. The object now has a +1 retain count}}
  return; // expected-note{{Object leaked: object allocated and stored into 'leaked' is not referenced later in this execution path and has a retain count of +1}}
}

void acquisitionViaProperty (Foo *foo) {
  id leaked = foo.propertyValue; // expected-warning{{leak}} expected-note{{Property returns an Objective-C object with a +0 retain count}}
  [leaked retain]; // expected-note{{Reference count incremented. The object now has a +1 retain count}}
  return; // expected-note{{Object leaked: object allocated and stored into 'leaked' is not referenced later in this execution path and has a retain count of +1}}
}

void acquisitionViaCFFunction () {
  CFTypeRef leaked = CFGetSomething(); // expected-warning{{leak}} expected-note{{Call to function 'CFGetSomething' returns a Core Foundation object with a +0 retain count}}
  CFRetain(leaked); // expected-note{{Reference count incremented. The object now has a +1 retain count}}
  return; // expected-note{{Object leaked: object allocated and stored into 'leaked' is not referenced later in this execution path and has a retain count of +1}}
}

void explicitDealloc () {
  id object = [[NSObject alloc] init]; // expected-note{{Method returns an Objective-C object with a +1 retain count}}
  [object dealloc]; // expected-note{{Object released by directly sending the '-dealloc' message}}
  [object class]; // expected-warning{{Reference-counted object is used after it is released}} // expected-note{{Reference-counted object is used after it is released}}
}

void implicitDealloc () {
  id object = [[NSObject alloc] init]; // expected-note{{Method returns an Objective-C object with a +1 retain count}}
  [object release]; // expected-note{{Object released}}
  [object class]; // expected-warning{{Reference-counted object is used after it is released}} // expected-note{{Reference-counted object is used after it is released}}
}

void overAutorelease () {
  id object = [[NSObject alloc] init]; // expected-note{{Method returns an Objective-C object with a +1 retain count}}
  [object autorelease]; // expected-note{{Object sent -autorelease message}}
  [object autorelease]; // expected-note{{Object sent -autorelease message}} 
  return; // expected-warning{{Object sent -autorelease too many times}} expected-note{{Object over-autoreleased: object was sent -autorelease 2 times but the object has a +1 retain count}} 
}

void autoreleaseUnowned (Foo *foo) {
  id object = foo.propertyValue; // expected-note{{Property returns an Objective-C object with a +0 retain count}}
  [object autorelease]; // expected-note{{Object sent -autorelease message}} 
  return; // expected-warning{{Object sent -autorelease too many times}} expected-note{{Object over-autoreleased: object was sent -autorelease but the object has a +0 retain count}}
}

void makeCollectableIgnored () {
  CFTypeRef leaked = CFCreateSomething(); // expected-warning{{leak}} expected-note{{Call to function 'CFCreateSomething' returns a Core Foundation object with a +1 retain count}}
  CFMakeCollectable(leaked); // expected-note{{When GC is not enabled a call to 'CFMakeCollectable' has no effect on its argument}}
  NSMakeCollectable(leaked); // expected-note{{When GC is not enabled a call to 'NSMakeCollectable' has no effect on its argument}}
  return; // expected-note{{Object leaked: object allocated and stored into 'leaked' is not referenced later in this execution path and has a retain count of +1}}
}

CFTypeRef CFCopyRuleViolation () {
  CFTypeRef object = CFGetSomething(); // expected-note{{Call to function 'CFGetSomething' returns a Core Foundation object with a +0 retain counte}}
  return object; // expected-warning{{Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected}} expected-note{{Object returned to caller with a +0 retain count}} expected-note{{Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected}}
}

CFTypeRef CFGetRuleViolation () {
  CFTypeRef object = CFCreateSomething(); // expected-warning{{leak}} expected-note{{Call to function 'CFCreateSomething' returns a Core Foundation object with a +1 retain counte}}
  return object; // expected-note{{Object returned to caller as an owning reference (single retain count transferred to caller)}} expected-note{{Object leaked: object allocated and stored into 'object' is return from a function whose name ('CFGetRuleViolation') does not contain 'Copy' or 'Create'.  This violates the naming convention rules given the Memory Management Guide for Core Foundation}}
}

@implementation Foo (FundamentalMemoryManagementRules)
- (id)copyViolation {
  id result = self.propertyValue; // expected-note{{Property returns an Objective-C object with a +0 retain count}}
  return result; // expected-warning{{Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected}} expected-note{{Object returned to caller with a +0 retain count}} expected-note{{Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected}}
}

- (id)getViolation {
  id result = [[Foo alloc] init]; // expected-warning{{leak}} expected-note{{Method returns an Objective-C object with a +1 retain count}}
  return result; // expected-note{{Object returned to caller as an owning reference (single retain count transferred to caller)}} expected-note{{Object leaked: object allocated and stored into 'result' is returned from a method whose name ('getViolation') does not start with 'copy', 'mutableCopy', 'alloc' or 'new'.  This violates the naming convention rules given in the Memory Management Guide for Cocoa}}
}
@end
