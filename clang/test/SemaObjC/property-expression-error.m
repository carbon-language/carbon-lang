// RUN: %clang_cc1  -fsyntax-only -verify %s

@interface AddressMyProperties 
{
  unsigned index;
}
@property unsigned index;
@end

@implementation AddressMyProperties
@synthesize index;
@end

int main() {
	AddressMyProperties *object;
	&object.index; // expected-error {{address of property expression requested}}
	return 0;
}
