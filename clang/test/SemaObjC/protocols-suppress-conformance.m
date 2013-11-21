// RUN: %clang_cc1  -fsyntax-only -verify %s -Wno-objc-root-class

@protocol FooProto
- (void) theBestOfTimes; // expected-note {{method 'theBestOfTimes' declared here}}
@end

__attribute__((objc_suppress_protocol(FooProto)))
@interface Bar
- (void) theBestOfTimes;
@end

@interface Bar2 : Bar
@end

@interface Baz : Bar2 <FooProto> // expected-note {{required for direct or indirect protocol 'FooProto'}}
@end

@interface Baz2 : Bar2 <FooProto>
- (void) theBestOfTimes;
@end

@implementation Baz // expected-warning {{method 'theBestOfTimes' in protocol not implemented}}
@end

@implementation Baz2 // no-warning
- (void) theBestOfTimes {}
@end