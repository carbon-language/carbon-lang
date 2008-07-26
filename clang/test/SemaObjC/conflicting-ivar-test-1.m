// RUN: clang -fsyntax-only -verify %s

@interface INTF 
{
@public
	int IVAR; // expected-error {{previous definition is here}}
}
@end

@implementation INTF
{
@private

        int XIVAR; // expected-error {{conflicting instance variable name 'XIVAR'}}
}
@end



@interface INTF1 
{
@public
	int IVAR;
	int IVAR1; // expected-error {{inconsistent number of instance variables specified}}
}
@end

@implementation INTF1
{
@private

        int IVAR;
}
@end


@interface INTF2 
{
@public
	int IVAR;
}
@end

@implementation INTF2
{
@private

        int IVAR;
	int IVAR1; // expected-error {{inconsistent number of instance variables specified}}
}
@end


@interface INTF3
{
@public
	int IVAR; // expected-error {{previous definition is here}}
}
@end

@implementation INTF3
{
@private

        short IVAR; // expected-error {{conflicting instance variable type}}
}
@end

@implementation  INTF4 // expected-warning {{cannot find interface declaration for 'INTF4'}}
{
@private

        short IVAR;
}
@end

@interface INTF5
{
  char * ch;
}
@end

@implementation  INTF5
{
}
@end
