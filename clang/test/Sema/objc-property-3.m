// RUN: clang -verify %s

@interface I 
{
	id d1;
}
@property (readwrite, copy) id d1;
@property (readwrite, copy) id d2;
@end

@interface NOW : I
@property (readonly, retain) id d1; // expected-warning {{attribute 'readonly' of property 'd1' restricts attribute 'readwrite' of property inherited from 'I'}} expected-warning {{property 'd1' 'copy' attribute does not match the property inherited from'I'}}
@property (readwrite, copy) I* d2; // expected-warning {{property type 'I *' does not match property type inherited from 'I'}}
@end
