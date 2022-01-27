// RUN: %clang_cc1 %s -fsyntax-only -std=c99 -pedantic -verify -Wundef
// RUN: %clang_cc1 %s -fsyntax-only -x c++ -pedantic -verify -Wundef
// RUN: not %clang_cc1 %s -fsyntax-only -std=c99 -pedantic -Wundef 2>&1 | FileCheck -strict-whitespace %s

#define \u00FC
#define a\u00FD() 0
#ifndef \u00FC
#error "This should never happen"
#endif

#if a\u00FD()
#error "This should never happen"
#endif

#if a\U000000FD()
#error "This should never happen"
#endif

#if a\u{FD}() //expected-warning {{Clang extension}}
#error "This should never happen"
#endif

#if \uarecool // expected-warning{{incomplete universal character name; treating as '\' followed by identifier}} expected-error {{invalid token at start of a preprocessor expression}}
#endif
#if \uwerecool // expected-warning{{\u used with no following hex digits; treating as '\' followed by identifier}} expected-error {{invalid token at start of a preprocessor expression}}
#endif
#if \U0001000  // expected-warning{{incomplete universal character name; treating as '\' followed by identifier}} expected-error {{invalid token at start of a preprocessor expression}}
#endif

// Make sure we reject disallowed UCNs
#define \ufffe // expected-error {{macro name must be an identifier}}
#define \U10000000  // expected-error {{macro name must be an identifier}}
#define \u0061  // expected-error {{character 'a' cannot be specified by a universal character name}} expected-error {{macro name must be an identifier}}
#define \u{fffe} // expected-error {{macro name must be an identifier}} expected-warning {{Clang extension}}

#define a\u0024

#if \u0110 // expected-warning {{is not defined, evaluates to 0}}
#endif


#define \u0110 1 / 0
#if \u0110 // expected-error {{division by zero in preprocessor expression}}
#endif

#define STRINGIZE(X) # X

extern int check_size[sizeof(STRINGIZE(\u0112)) == 3 ? 1 : -1];

// Check that we still diagnose disallowed UCNs in #if 0 blocks.
// C99 5.1.1.2p1 and C++11 [lex.phases]p1 dictate that preprocessor tokens are
// formed before directives are parsed.
// expected-error@+4 {{character 'a' cannot be specified by a universal character name}}
#if 0
#define \ufffe // okay
#define \U10000000 // okay
#define \u0061 // error, but -verify only looks at comments outside #if 0
#endif


// A UCN formed by token pasting is undefined in both C99 and C++.
// Right now we don't do anything special, which causes us to coincidentally
// accept the first case below but reject the second two.
#define PASTE(A, B) A ## B
extern int PASTE(\, u00FD);
extern int PASTE(\u, 00FD); // expected-warning{{\u used with no following hex digits}}
extern int PASTE(\u0, 0FD); // expected-warning{{incomplete universal character name}}
#ifdef __cplusplus
// expected-error@-3 {{expected unqualified-id}}
// expected-error@-3 {{expected unqualified-id}}
#else
// expected-error@-6 {{expected identifier}}
// expected-error@-6 {{expected identifier}}
#endif


// A UCN produced by line splicing is valid in C99 but undefined in C++.
// Since undefined behavior can do anything including working as intended,
// we just accept it in C++ as well.;
#define newline_1_\u00F\
C 1
#define newline_2_\u00\
F\
C 1
#define newline_3_\u\
00\
FC 1
#define newline_4_\\
u00FC 1
#define newline_5_\\
u\
\
0\
0\
F\
C 1

#if (newline_1_\u00FC && newline_2_\u00FC && newline_3_\u00FC && \
     newline_4_\u00FC && newline_5_\u00FC)
#else
#error "Line splicing failed to produce UCNs"
#endif


#define capital_u_\U00FC
// expected-warning@-1 {{incomplete universal character name}} expected-note@-1 {{did you mean to use '\u'?}} expected-warning@-1 {{whitespace}}
// CHECK: note: did you mean to use '\u'?
// CHECK-NEXT:   #define capital_u_\U00FC
// CHECK-NEXT: {{^                   \^}}
// CHECK-NEXT: {{^                   u}}

#define \u{}           // expected-warning {{empty delimited universal character name; treating as '\' 'u' '{' '}'}} expected-error {{macro name must be an identifier}}
#define \u{123456789}  // expected-error {{hex escape sequence out of range}} expected-error {{macro name must be an identifier}}
#define \u{            // expected-warning {{incomplete delimited universal character name; treating as '\' 'u' '{' identifier}} expected-error {{macro name must be an identifier}}
#define \u{fgh}        // expected-warning {{incomplete delimited universal character name; treating as '\' 'u' '{' identifier}} expected-error {{macro name must be an identifier}}
