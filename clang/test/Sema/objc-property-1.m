// RUN: clang -fsyntax-only -verify %s

@interface I 
{
	int IVAR;
}
@property int d1;
@property id  prop_id;
@end

@interface I(CAT)
@property int d1;
@end

@implementation I
@synthesize d1;		// expected-error {{property synthesize requires specification of an ivar}}
@dynamic    bad;	// expected-error {{property implementation must have its declaration in the class 'I'}}
@synthesize prop_id;	// expected-error {{property synthesize requires specification of an ivar}}
@synthesize prop_id = IVAR;	// expected-error {{type of property 'prop_id'  does not match type of ivar 'IVAR'}}
@end

@implementation I(CAT)
@synthesize d1;		// expected-error {{@synthesize not allowed in a category's implementation}}
@dynamic bad;		// expected-error {{property implementation must have its declaration in the category 'CAT'}}
@end

@implementation E	// expected-warning {{cannot find interface declaration for 'E'}}
@dynamic d;		// expected-error {{property implementation must have its declaration in the class 'E'}}
@end

@implementation Q(MYCAT)  // expected-error {{cannot find interface declaration for 'Q'}}
@dynamic d;		// expected-error {{property implementation in a category with no category declaration}}
@end



