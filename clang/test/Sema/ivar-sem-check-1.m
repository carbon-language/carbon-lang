struct S;
typedef int FOO();

@interface INTF
{
	struct F {} JJ;
	int arr[];  // expected-error {{field 'arr' has incomplete type}}
	struct S IC;  // expected-error {{field 'IC' has incomplete type}}
	struct T { struct T {} X; }YYY; // expected-error {{nested redefinition of 'struct'}}
	FOO    BADFUNC;  // expected-error {{field 'BADFUNC' declared as a function}}
	int kaka;
	int kaka;	// expected-error {{duplicate member 'kaka'}}
	char ch[];	// expected-error {{field 'ch' has incomplete type}}
}
@end
