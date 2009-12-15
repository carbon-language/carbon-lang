// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface _Child
+ (int) flashCache;
@end

@interface Child (Categ) // expected-error {{cannot find interface declaration for 'Child'}}
+ (int) flushCache2;
@end

@implementation Child (Categ) // expected-error {{cannot find interface declaration for 'Child'}}
+ (int) flushCache2 { [super flashCache]; } // expected-error {{no @interface declaration found in class messaging of 'flushCache2'}}
@end
