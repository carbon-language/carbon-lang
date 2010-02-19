// RUN: %clang_cc1 -fsyntax-only -fobjc-nonfragile-abi2 -verify %s

@interface Super @end

@interface INTFSTANDALONE : Super
{
  id IVAR;	// expected-note {{previous definition is here}}
}

@end

@implementation INTFSTANDALONE : Super // expected-warning {{class implementation may not have super class}}
{
@private
  id IVAR1;
@protected
  id IVAR2;	// expected-error {{only private ivars may be declared in implementation}}
@private
  id IVAR3;
  int IVAR;	// expected-error {{instance variable is already declared}}
}
@end
