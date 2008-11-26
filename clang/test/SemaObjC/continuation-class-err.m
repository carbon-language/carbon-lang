// RUN: clang -fsyntax-only -verify %s

@interface ReadOnly 
{
  id _object;
  id _object1;
}
@property(readonly, assign) id object;
@property(readwrite, assign) id object1;
@end

@interface ReadOnly ()
@property(readwrite, copy) id object;	// expected-warning {{property attribute in continuation class does not match the primary class}}
@property(readonly) id object1; // expected-error {{use contination class to override 'readonly' property with 'readwrite'}}
@end
