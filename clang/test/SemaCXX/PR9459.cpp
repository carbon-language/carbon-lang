// RUN: %clang_cc1 -fsyntax-only -verify %s

// Don't crash.

template<typename>struct ae_same; // expected-note {{declared here}}
template<typename>struct ts{}ap()
{ts<a>::ap<ae_same<int>::&ae_same<>>::p(a); }; // expected-error 2 {{undeclared identifier}} \
    // expected-error 2 {{expected}} expected-error {{a space is required}} \
    // expected-error 2 {{global}} expected-error {{too few}}
