// RUN: clang -cc1 -fsyntax-only -verify %s

// Note: GCC doesn't produce any of the following errors.
@interface Super @end // expected-note {{previous definition is here}}

@interface MyWpModule @end  // expected-note {{previous definition is here}}

@compatibility_alias  MyAlias MyWpModule;

@compatibility_alias  AliasForSuper Super;

@interface MyAlias : AliasForSuper // expected-error {{duplicate interface definition for class 'MyWpModule'}}
@end

@implementation MyAlias : AliasForSuper // expected-error {{conflicting super class name 'Super'}}
@end

