// RUN: clang -fsyntax-only -verify -pedantic %s

#line 'a'            // expected-error {{#line directive requires a positive integer argument}}
#line 0              // expected-error {{#line directive requires a positive integer argument}}
#line 2147483648     // expected-warning {{C requires #line number to be less than 2147483648, allowed as extension}}
#line 42             // ok
#line 42 'a'         // expected-error {{nvalid filename for #line directive}}
#line 42 "foo/bar/baz.h"  // ok


// #line directives expand macros.
#define A 42 "foo"
#line A

# 42
# 42 "foo"
# 42 "foo" 1 3
# 42 "foo" 2 3
# 42 "foo" 2 3 4
# 42 "foo" 3 4

# 'a'            // expected-error {{invalid preprocessing directive}}
# 42 'f'         // expected-error {{invalid filename for line marker directive}}
# 42 1 3         // expected-error {{invalid filename for line marker directive}}
# 42 "foo" 3 1   // expected-error {{invalid flag line marker directive}}
# 42 "foo" 42    // expected-error {{invalid flag line marker directive}}
# 42 "foo" 1 2   // expected-error {{invalid flag line marker directive}}
