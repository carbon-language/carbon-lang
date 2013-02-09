// RUN: %clang_cc1 %s -fsyntax-only -std=c99 -verify
// RUN: %clang_cc1 %s -fsyntax-only -std=c11 -Wc99-compat -verify
// RUN: %clang_cc1 %s -fsyntax-only -x c++ -std=c++03 -Wc++11-compat -verify
// RUN: %clang_cc1 %s -fsyntax-only -x c++ -std=c++11 -Wc++98-compat -verify

// Note: This file contains Unicode characters; please do not remove them!

// Identifier characters
extern char aǶ; // C11, C++11
extern char aª; // C99, C11, C++11
extern char a΄; // C++03, C11, C++11
extern char a๐; // C99, C++03, C11, C++11
extern char a﹅; // none
extern char x̀; // C11, C++11. Note that this does not have a composed form.




// Identifier initial characters
extern char ๐; // C++03, C11, C++11
extern char ̀; // disallowed initially in C11/C++11, always in C99/C++03








#if __cplusplus
# if __cplusplus >= 201103L
// C++11
// expected-warning@9 {{using this character in an identifier is incompatible with C++98}}
// expected-warning@10 {{using this character in an identifier is incompatible with C++98}}
// expected-error@13 {{non-ASCII characters are not allowed outside of literals and identifiers}}
// expected-warning@14 {{using this character in an identifier is incompatible with C++98}}
// expected-error@21 {{expected unqualified-id}}

# else
// C++03
// expected-error@9 {{non-ASCII characters are not allowed outside of literals and identifiers}}
// expected-error@10 {{non-ASCII characters are not allowed outside of literals and identifiers}}
// expected-error@13 {{non-ASCII characters are not allowed outside of literals and identifiers}}
// expected-error@14 {{non-ASCII characters are not allowed outside of literals and identifiers}}
// expected-error@21 {{non-ASCII characters are not allowed outside of literals and identifiers}} expected-warning@21 {{declaration does not declare anything}}

# endif
#else
# if __STDC_VERSION__ >= 201112L
// C11
// expected-warning@9 {{using this character in an identifier is incompatible with C99}}
// expected-warning@11 {{using this character in an identifier is incompatible with C99}}
// expected-error@13 {{non-ASCII characters are not allowed outside of literals and identifiers}}
// expected-warning@14 {{using this character in an identifier is incompatible with C99}}
// expected-warning@20 {{starting an identifier with this character is incompatible with C99}}
// expected-error@21 {{expected identifier}}

# else
// C99
// expected-error@9 {{non-ASCII characters are not allowed outside of literals and identifiers}}
// expected-error@11 {{non-ASCII characters are not allowed outside of literals and identifiers}}
// expected-error@13 {{non-ASCII characters are not allowed outside of literals and identifiers}}
// expected-error@14 {{non-ASCII characters are not allowed outside of literals and identifiers}}
// expected-error@20 {{expected identifier}}
// expected-error@21 {{non-ASCII characters are not allowed outside of literals and identifiers}} expected-warning@21 {{declaration does not declare anything}}

# endif
#endif
