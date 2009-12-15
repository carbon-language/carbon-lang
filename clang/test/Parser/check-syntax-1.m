// RUN: %clang_cc1 -fsyntax-only -verify %s

int @interface bla  ; // expected-error {{cannot combine with previous 'int' declaration specifier}}
@end

typedef float CGFloat;
@interface XNSNumber 
+ (XNSNumber *) numberWithCGFloat  : (CGFloat) float; // expected-error {{expected identifier}}  \
                                                      // expected-error {{ expected ';' after method prototype}}
@end

