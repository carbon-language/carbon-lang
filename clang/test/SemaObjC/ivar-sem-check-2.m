// RUN: clang-cc  -fsyntax-only -fobjc-nonfragile-abi -verify %s

@interface Super  {
  id value2; // expected-note {{previously declared 'value2' here}}
} 
@property(retain) id value;
@property(retain) id value1;
@property(retain) id prop;
@end

@interface Sub : Super 
{
  id value; 
}
@end

@implementation Sub
@synthesize value; // expected-note {{previous use is here}}
@synthesize value1=value; // expected-error {{synthesized properties 'value1' and 'value' both claim ivar 'value'}} 
@synthesize prop=value2;  // expected-error {{property 'prop' attempting to use ivar 'value2' declared in super class 'Super'}}
@end


