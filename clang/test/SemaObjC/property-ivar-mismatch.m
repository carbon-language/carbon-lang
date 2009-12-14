// RUN: clang -cc1 -fsyntax-only -verify %s
// Test that arithmatic types on property and its ivar have exact match.

@interface Test4 
{
   char ivar;
}
@property int prop;
@end

@implementation Test4
@synthesize prop = ivar;  // expected-error {{type of property 'prop' does not match type of ivar 'ivar'}}
@end

