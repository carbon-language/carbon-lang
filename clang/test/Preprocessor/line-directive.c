// RUN: clang -fsyntax-only -verify -pedantic %s &&
// RUN: clang -E %s 2>&1 | grep 'blonk.c:92:2: error: #error ABC' &&
// RUN: clang -E %s 2>&1 | grep 'blonk.c:93:2: error: #error DEF'

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


// These are checked by the RUN line.
#line 92 "blonk.c"
#error ABC  // expected-error {{#error ABC}}
#error DEF  // expected-error {{#error DEF}}


// Verify that linemarker diddling of the system header flag works.

# 192 "glomp.h" // not a system header.
typedef int x;  // expected-note {{previous definition is here}}
typedef int x;  // expected-error {{redefinition of 'x'}}

# 192 "glomp.h" 3 // System header.
typedef int y;  // ok
typedef int y;  // ok

#line 42 "blonk.h"  // doesn't change system headerness.

typedef int z;  // ok
typedef int z;  // ok

# 97     // doesn't change system headerness.

typedef int z1;  // ok
typedef int z1;  // ok

# 42 "blonk.h"  // DOES change system headerness.

typedef int w;  // expected-note {{previous definition is here}}
typedef int w;  // expected-error {{redefinition of 'w'}}
