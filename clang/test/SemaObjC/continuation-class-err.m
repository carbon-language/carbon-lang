// RUN: clang-cc -fsyntax-only -verify %s

@interface ReadOnly 
{
  id _object;
  id _object1;
}
@property(readonly) id object;
@property(readwrite, assign) id object1;
@end

@interface ReadOnly ()
@property(readwrite, copy) id object;	
@property(readonly) id object1; // expected-error {{attribute of property in continuation class of 'ReadOnly' can only  be 'readwrite'}}
@end
