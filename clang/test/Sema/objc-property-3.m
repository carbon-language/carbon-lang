// RUN: clang -verify %s

@interface I 
{
	id d1;
}
@property (readwrite, copy) id d1;
@property (readwrite, copy) id d2;
@end

@interface NOW : I
@property (readonly, retain) id d1; // expected-warning {{attribute 'readonly' of property 'd1' restricts attribute 'readwrite' of 'I' property in super class}} expected-warning {{property 'd1' 'copy' attribute does not match super class 'I' property}}
@property (readwrite, copy) I* d2; // expected-warning {{property type 'I *' does not match super class 'I' property type}}
@end

