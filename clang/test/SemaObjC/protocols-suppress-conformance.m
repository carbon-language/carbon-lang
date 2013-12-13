// RUN: %clang_cc1  -triple x86_64-apple-darwin11 -fsyntax-only -verify %s -Wno-objc-root-class

// Mark this protocol as requiring all of its methods and properties
// to be explicitly implemented in the adopting class.
__attribute__((objc_protocol_requires_explicit_implementation))
@protocol Protocol
- (void) theBestOfTimes; // expected-note {{method 'theBestOfTimes' declared here}}
@property (readonly) id theWorstOfTimes;
@end

// In this example, ClassA adopts the protocol.  We won't
// provide the implementation here, but this protocol will
// be adopted later by a subclass.
@interface ClassA <Protocol>
- (void) theBestOfTimes;
@property (readonly) id theWorstOfTimes;
@end

// This class subclasses ClassA (which adopts 'Protocol'),
// but does not provide the needed implementation.
@interface ClassB : ClassA <Protocol>
@end

@implementation ClassB // expected-warning {{method 'theBestOfTimes' in protocol 'Protocol' not implemented}}
@end

// Test that inherited protocols do not get the explicit conformance requirement.
@protocol Inherited
- (void) fairIsFoul;
@end

__attribute__((objc_protocol_requires_explicit_implementation))
@protocol Derived <Inherited>
- (void) foulIsFair; // expected-note {{method 'foulIsFair' declared here}}
@end

@interface ClassC <Inherited>
@end

@interface ClassD : ClassC <Derived>
@end

@implementation ClassD // expected-warning {{method 'foulIsFair' in protocol 'Derived' not implemented}}
@end

// Test that the attribute is used correctly.
__attribute__((objc_protocol_requires_explicit_implementation(1+2))) // expected-error {{attribute takes no arguments}}
@protocol AnotherProtocol @end

// Cannot put the attribute on classes or other non-protocol declarations.
__attribute__((objc_protocol_requires_explicit_implementation)) // expected-error {{attribute only applies to Objective-C protocols}}
@interface AnotherClass @end

__attribute__((objc_protocol_requires_explicit_implementation)) // expected-error {{attribute only applies to Objective-C protocols}}
int x;

// Test that inherited protocols with the attribute
// are treated properly.
__attribute__((objc_protocol_requires_explicit_implementation))
@protocol ProtocolA
@required
- (void)rlyeh;
- (void)innsmouth;
@end

@protocol ProtocolB <ProtocolA>
@required
- (void)dunwich;
- (id)innsmouth;
@end

@protocol ProtocolC
@required
- (void)rlyeh;
- (void)innsmouth;
- (void)dunwich;
@end

@interface MyObject <ProtocolC>
@end

@interface MyLovecraft <ProtocolA>
@end

@interface MyShoggoth : MyLovecraft <ProtocolB>
@end

@implementation MyObject
- (void)innsmouth {}
- (void)rlyeh {}
- (void)dunwich {}
@end

@implementation MyLovecraft
- (void)innsmouth {}
- (void)rlyeh {}
@end

@implementation MyShoggoth
- (void)dunwich {}
@end

