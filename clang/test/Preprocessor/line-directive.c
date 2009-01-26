// RUN: clang -fsyntax-only -verify -pedantic %s

#line 'a'            // expected-error {{#line directive requires a positive integer argument}}
#line 0              // expected-error {{#line directive requires a positive integer argument}}
#line 2147483648     // expected-warning {{C requires #line number to be less than 2147483648, allowed as extension}}
#line 42             // ok
#line 42 'a'         // expected-error {{nvalid filename for #line directive}}
#line 42 "foo/bar/baz.h"  // ok


