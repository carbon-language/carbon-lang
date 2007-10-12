// RUN: clang -fsyntax-only -verify %s

@interface Super @end

@interface MyWpModule @end

@compatibility_alias  MyAlias MyWpModule;

@compatibility_alias  AliasForSuper Super;

@interface MyAlias : AliasForSuper // expected-error {{duplicate interface declaration for class 'MyWpModule'}}
@end

@implementation MyAlias : AliasForSuper
@end

