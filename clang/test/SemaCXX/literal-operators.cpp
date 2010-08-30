// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s

#include <stddef.h>

struct tag {
  void operator "" tag_bad (const char *); // expected-error {{literal operator 'operator "" tag_bad' must be in a namespace or global scope}}
  friend void operator "" tag_good (const char *);
};

namespace ns { void operator "" ns_good (const char *); }

// Check extern "C++" declarations
extern "C++" void operator "" extern_good (const char *);
extern "C++" { void operator "" extern_good (const char *); }

void fn () { void operator "" fn_bad (const char *); } // expected-error {{literal operator 'operator "" fn_bad' must be in a namespace or global scope}}

// One-param declarations (const char * was already checked)
void operator "" good (char);
void operator "" good (wchar_t);
void operator "" good (char16_t);
void operator "" good (char32_t);
void operator "" good (unsigned long long);
void operator "" good (long double);

// Two-param declarations
void operator "" good (const char *, size_t);
void operator "" good (const wchar_t *, size_t);
void operator "" good (const char16_t *, size_t);
void operator "" good (const char32_t *, size_t);

// Check typedef and array equivalences
void operator "" good (const char[]);
typedef const char c;
void operator "" good (c*);

// Check extra cv-qualifiers
void operator "" cv_good (volatile const char *, const size_t);

// Template delcaration (not implemented yet)
// template <char...> void operator "" good ();

// FIXME: Test some invalid decls that might crop up.
