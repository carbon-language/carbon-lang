// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

#include <stddef.h>

struct tag {
  void operator "" _tag_bad (const char *); // expected-error {{literal operator 'operator "" _tag_bad' must be in a namespace or global scope}}
  friend void operator "" _tag_good (const char *);
};

namespace ns { void operator "" _ns_good (const char *); }

// Check extern "C++" declarations
extern "C++" void operator "" _extern_good (const char *);
extern "C++" { void operator "" _extern_good (const char *); }

void fn () { void operator "" _fn_bad (const char *); } // expected-error {{literal operator 'operator "" _fn_bad' must be in a namespace or global scope}}

// One-param declarations (const char * was already checked)
void operator "" _good (char);
void operator "" _good (wchar_t);
void operator "" _good (char16_t);
void operator "" _good (char32_t);
void operator "" _good (unsigned long long);
void operator "" _good (long double);

// Two-param declarations
void operator "" _good (const char *, size_t);
void operator "" _good (const wchar_t *, size_t);
void operator "" _good (const char16_t *, size_t);
void operator "" _good (const char32_t *, size_t);

// Check typedef and array equivalences
void operator "" _good (const char[]);
typedef const char c;
void operator "" _good (c*);

// Check extra cv-qualifiers
void operator "" _cv_good (volatile const char *, const size_t);

// Template delcaration (not implemented yet)
// template <char...> void operator "" good ();

// FIXME: Test some invalid decls that might crop up.
