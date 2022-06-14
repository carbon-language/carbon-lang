// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -I%S/Inputs/malformed-overload -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Wno-strict-prototypes -verify %s
NSLog(@"%@", path); // expected-error {{expected parameter declarator}} expected-error {{expected ')'}} expected-error {{type specifier missing}} expected-warning {{incompatible redeclaration}} expected-note {{to match this '('}} expected-note {{'NSLog' is a builtin with type}}
#import "X.h"

@class NSString;
void f(NSString *a) {
 NSLog(@"***** failed to get URL for %@", a);
}
