// RUN: %clang_cc1  -triple x86_64-apple-darwin11 -fsyntax-only -verify %s -Wno-objc-root-class

@protocol Protocol
- (void) theBestOfTimes; // expected-note {{method 'theBestOfTimes' declared here}}
@property (readonly) id theWorstOfTimes; // expected-note {{property declared here}}
@end

// In this example, the root class provides all the methods for
// a protocol, and the immediate subclass adopts the attribute.
//
// The further subclasses should not have access to the root class's
// methods for checking protocol conformance.
//
// ClassC states protocol conformance, but does not redeclare the method.
// For this case we get a warning.
//
// ClassD states protocol conformance, but does redeclare the method.
// For this case we do not get a warning.
//

@interface ClassA <Protocol>
- (void) theBestOfTimes;
//@property (readonly) id theWorstOfTimes;
@end

__attribute__((objc_suppress_protocol_methods(Protocol))) @interface ClassB : ClassA @end

@interface ClassC : ClassB <Protocol> @end // expected-note {{required for direct or indirect protocol 'Protocol'}}

@interface ClassD : ClassB <Protocol>
- (void) theBestOfTimes;
@property (readonly) id theWorstOfTimes;
@end

@implementation ClassA // expected-warning {{auto property synthesis will not synthesize property declared in a protocol}}
- (void) theBestOfTimes {}
@end

@implementation ClassC @end // expected-warning {{method 'theBestOfTimes' in protocol not implemented}}

@implementation ClassD // no-warning
- (void) theBestOfTimes {}
@end

// In this example, the class both conforms to the protocl and adopts
// the attribute.  This illustrates that the attribute does not
// interfere with the protocol conformance checking for the class
// itself.
__attribute__((objc_suppress_protocol_methods(Protocol)))
@interface AdoptsAndConforms <Protocol>
- (void) theBestOfTimes;
@property (readonly) id theWorstOfTimes;
@end

@implementation AdoptsAndConforms // no-warning
- (void) theBestOfTimes {}
@end

// This attribute cannot be added to a class extension or category.
@interface ClassE
-(void) theBestOfTimes;
@end

__attribute__((objc_supress_protocol(Protocol)))
@interface ClassE () @end // expected-error {{attributes may not be specified on a category}}

__attribute__((objc_supress_protocol(Protocol)))
@interface ClassE (MyCat) @end // expected-error {{attributes may not be specified on a category}}

// The attribute requires one or more identifiers.
__attribute__((objc_suppress_protocol_methods())) // expected-error {{'objc_suppress_protocol_methods' attribute takes one argument}}
@interface ClassF @end

// The attribute requires one or more identifiers.
__attribute__((objc_suppress_protocol_methods(ProtoA, ProtoB))) // expected-error {{use of undeclared identifier 'ProtoB'}}
@interface ClassG @end
__attribute__((objc_suppress_protocol_methods(1+2)))
@interface ClassH @end // expected-error {{parameter of 'objc_suppress_protocol_methods' attribute must be a single name of an Objective-C protocol}}
  