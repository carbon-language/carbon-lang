// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s
// rdar://16462586

__attribute__((objc_runtime_name)) // expected-error {{'objc_runtime_name' attribute takes one argument}}
@interface BInterface
@end

__attribute__((objc_runtime_name(123))) // expected-error {{'objc_runtime_name' attribute requires a string}}
@protocol BProtocol1
@end

__attribute__((objc_runtime_name("MySecretNamespace.Protocol")))
@protocol Protocol
@end

__attribute__((objc_runtime_name("MySecretNamespace.Message")))
@interface Message <Protocol> { 
__attribute__((objc_runtime_name("MySecretNamespace.Message"))) // expected-error {{'objc_runtime_name' attribute only applies to Objective-C interfaces and Objective-C protocols}}
  id MyIVAR;
}
__attribute__((objc_runtime_name("MySecretNamespace.Message")))
@property int MyProperty; // expected-error {{prefix attribute must be followed by an interface, protocol, or implementation}}}}

- (int) getMyProperty __attribute__((objc_runtime_name("MySecretNamespace.Message"))); // expected-error {{'objc_runtime_name' attribute only applies to}}

- (void) setMyProperty : (int) arg __attribute__((objc_runtime_name("MySecretNamespace.Message"))); // expected-error {{'objc_runtime_name' attribute only applies to}}

@end

__attribute__((objc_runtime_name("MySecretNamespace.ForwardClass")))
@class ForwardClass; // expected-error {{prefix attribute must be followed by an interface, protocol, or implementation}}

__attribute__((objc_runtime_name("MySecretNamespace.ForwardProtocol")))
@protocol ForwardProtocol;

@implementation Message
// expected-error@+1 {{'objc_runtime_name' attribute only applies to Objective-C interfaces and Objective-C protocols}}
- (id) MyMethod __attribute__((objc_runtime_name("MySecretNamespace.Message"))) {
  return MyIVAR;
}

-(int)getMyProperty { return 0; }
-(void)setMyProperty:(int)arg {}
@end

@interface NoImpl @end

// expected-error@+1 {{'objc_runtime_name' attribute only applies to Objective-C interfaces and Objective-C protocols}}
__attribute__((objc_runtime_name("MySecretNamespace.Message")))
@implementation NoImpl @end
