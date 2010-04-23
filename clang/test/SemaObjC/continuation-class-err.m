// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface ReadOnly 
{
  id _object;
  id _object1;
}
@property(readonly) id object;	// expected-note {{property declared here}}
@property(readwrite, assign) id object1; // expected-note {{property declared here}}
@property (readonly) int indentLevel;
@end

@interface ReadOnly ()
@property(readwrite, copy) id object;	// expected-warning {{property attribute in continuation class does not match the primary class}}
@property(readonly) id object1; // expected-error {{illegal declaration of property in continuation class 'ReadOnly': attribute must be}}
@property (readwrite, assign) int indentLevel; // OK. assign the the default in any case.
@end

@protocol Proto
  @property (copy) id fee; // expected-note {{property declared here}}
@end

@protocol Foo<Proto>
  @property (copy) id foo; // expected-note {{property declared here}}
@end

@interface Bar  <Foo> {
        id _foo;
        id _fee;
}
@end

@interface Bar ()
@property (copy) id foo;	// expected-error {{illegal declaration of property in continuation class 'Bar': attribute must be}}
@property (copy) id fee;	// expected-error {{illegal declaration of property in continuation class 'Bar': attribute must be}}
@end

@implementation Bar
@synthesize foo = _foo;
@synthesize fee = _fee;
@end

