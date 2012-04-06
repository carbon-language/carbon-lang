// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// Test that arithmatic types on property and its ivar have exact match.

@interface Test4 
{
   char ivar; // expected-note{{ivar is declared here}}
}
@property int prop;
@end

@implementation Test4
@synthesize prop = ivar;  // expected-error {{type of property 'prop' ('int') does not match type of ivar 'ivar' ('char')}}
@end

