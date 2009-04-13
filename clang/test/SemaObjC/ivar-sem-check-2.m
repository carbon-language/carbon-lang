// RUN: clang-cc  -fsyntax-only -triple x86_64-apple-darwin10 -verify %s

@interface Super  {
  id value; // expected-note {{previously declared 'value' here}}
} 
@property(retain) id value;
@property(retain) id value1;
@end

@interface Sub : Super @end

@implementation Sub
@synthesize value; // expected-error {{property 'value' attempting to use ivar 'value' declared in in super class 'Super'}} // expected-note {{previous use is here}}
@synthesize value1=value; // expected-error {{synthesized properties 'value1' and 'value' both claim ivar 'value'}}
@end


