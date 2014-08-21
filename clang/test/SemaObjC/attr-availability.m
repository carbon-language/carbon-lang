// RUN: %clang_cc1 -triple x86_64-apple-darwin9.0.0 -fsyntax-only -verify %s

@protocol P
- (void)proto_method __attribute__((availability(macosx,introduced=10.1,deprecated=10.2))); // expected-note 2 {{'proto_method' has been explicitly marked deprecated here}}
@end

@interface A <P>
- (void)method __attribute__((availability(macosx,introduced=10.1,deprecated=10.2))); // expected-note {{'method' has been explicitly marked deprecated here}}

- (void)overridden __attribute__((availability(macosx,introduced=10.3))); // expected-note{{overridden method is here}}
- (void)overridden2 __attribute__((availability(macosx,introduced=10.3)));
- (void)overridden3 __attribute__((availability(macosx,deprecated=10.3)));
- (void)overridden4 __attribute__((availability(macosx,deprecated=10.3))); // expected-note{{overridden method is here}}
- (void)overridden5 __attribute__((availability(macosx,unavailable)));
- (void)overridden6 __attribute__((availability(macosx,introduced=10.3))); // expected-note{{overridden method is here}}
@end

// rdar://11475360
@interface B : A
- (void)method; // NOTE: we expect 'method' to *not* inherit availability.
- (void)overridden __attribute__((availability(macosx,introduced=10.4))); // expected-warning{{overriding method introduced after overridden method on OS X (10.4 vs. 10.3)}}
- (void)overridden2 __attribute__((availability(macosx,introduced=10.2)));
- (void)overridden3 __attribute__((availability(macosx,deprecated=10.4)));
- (void)overridden4 __attribute__((availability(macosx,deprecated=10.2))); // expected-warning{{overriding method deprecated before overridden method on OS X (10.3 vs. 10.2)}}
- (void)overridden5 __attribute__((availability(macosx,introduced=10.3)));
- (void)overridden6 __attribute__((availability(macosx,unavailable))); // expected-warning{{overriding method cannot be unavailable on OS X when its overridden method is available}}
@end

void f(A *a, B *b) {
  [a method]; // expected-warning{{'method' is deprecated: first deprecated in OS X 10.2}}
  [b method]; // no-warning
  [a proto_method]; // expected-warning{{'proto_method' is deprecated: first deprecated in OS X 10.2}}
  [b proto_method]; // expected-warning{{'proto_method' is deprecated: first deprecated in OS X 10.2}}
}

// Test case for <rdar://problem/11627873>.  Warn about
// using a deprecated method when that method is re-implemented in a
// subclass where the redeclared method is not deprecated.
@interface C
- (void) method __attribute__((availability(macosx,introduced=10.1,deprecated=10.2))); // expected-note {{'method' has been explicitly marked deprecated here}}
@end

@interface D : C
- (void) method;
@end

@interface E : D
- (void) method;
@end

@implementation D
- (void) method {
  [super method]; // expected-warning {{'method' is deprecated: first deprecated in OS X 10.2}}
}
@end

@implementation E
- (void) method {
  [super method]; // no-warning
}
@end

// rdar://18059669
@class NSMutableArray;

@interface NSDictionary
+ (instancetype)dictionaryWithObjectsAndKeys:(id)firstObject, ... __attribute__((sentinel(0,1)));
@end

@class NSString;

extern NSString *NSNibTopLevelObjects __attribute__((availability(macosx,introduced=10.0 ,deprecated=10.8,message="" )));
id NSNibOwner, topNibObjects;

@interface AppDelegate (SIEImport) // expected-error {{cannot find interface declaration for 'AppDelegate'}}

-(void)__attribute__((ibaction))importFromSIE:(id)sender;

@end

@implementation AppDelegate (SIEImport) // expected-error {{cannot find interface declaration for 'AppDelegate'}}

-(void)__attribute__((ibaction))importFromSIE:(id)sender {

 NSMutableArray *topNibObjects;
 NSDictionary *nibLoadDict = [NSDictionary dictionaryWithObjectsAndKeys:self, NSNibOwner, topNibObjects, NSNibTopLevelObjects, ((void *)0)];
}

@end
