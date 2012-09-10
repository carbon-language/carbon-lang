// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1  -fsyntax-only -fobjc-arc -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fobjc-arc -fsyntax-only -verify -Wno-objc-root-class %s

// rdar://6386358
@protocol NSObject // expected-note {{protocol is declared here}}
- MyDealloc __attribute((objc_requires_super)); // expected-warning {{'objc_requires_super' attribute cannot be applied to methods in protocols}}
@end

@interface Root
- MyDealloc __attribute((objc_requires_super));
- (void)XXX __attribute((objc_requires_super));
- (void) dealloc __attribute((objc_requires_super)); // expected-warning {{'objc_requires_super' attribute cannot be applied to dealloc}}
@end

@interface Baz : Root<NSObject>
- MyDealloc;
@end

@implementation Baz
-  MyDealloc {
   [super MyDealloc];
        return 0;
}

- (void)XXX {
  [super MyDealloc];
} // expected-warning {{method possibly missing a [super XXX] call}}
@end

