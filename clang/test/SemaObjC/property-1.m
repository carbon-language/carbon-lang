// RUN: clang -fsyntax-only -verify %s

@interface I 
{
	int IVAR;
	int name;
}
@property int d1;
@property id  prop_id; // expected-warning {{no 'assign', 'retain', or 'copy' attribute is specified - 'assign' is assumed}}, expected-warning {{default property attribute 'assign' not appropriate for non-gc object}}
@property int name;
@end

@interface I(CAT)
@property int d1;
@end

@implementation I
@synthesize d1;		// expected-error {{synthesized property 'd1' must either be named the same as}}
@dynamic    bad;	// expected-error {{property implementation must have its declaration in interface 'I'}}
@synthesize prop_id;	// expected-error {{synthesized property 'prop_id' must either be named the same}}
@synthesize prop_id = IVAR;	// expected-error {{type of property 'prop_id'  does not match type of ivar 'IVAR'}}
@synthesize name;	// OK! property with same name as an accessible ivar of same name
@end

@implementation I(CAT)  // expected-warning {{incomplete implementation}}, expected-warning {{method definition for 'd1' not found}}, // expected-warning {{method definition for 'setD1:' not found}} 
@synthesize d1;		// expected-error {{@synthesize not allowed in a category's implementation}}
@dynamic bad;		// expected-error {{property implementation must have its declaration in the category 'CAT'}}
@end

@implementation E	// expected-warning {{cannot find interface declaration for 'E'}}
@dynamic d;		// expected-error {{property implementation must have its declaration in interface 'E'}}
@end

@implementation Q(MYCAT)  // expected-error {{cannot find interface declaration for 'Q'}}
@dynamic d;		// expected-error {{property implementation in a category with no category declaration}}
@end



