// RUN: clang-cc -fsyntax-only -verify %s

namespace N { };

namespace A = N;

int B; // expected-note {{previous definition is here}}
namespace B = N; // expected-error {{redefinition of 'B' as different kind of symbol}}

namespace C { } // expected-note {{previous definition is here}}
namespace C = N; // expected-error {{redefinition of 'C'}}

int i;
namespace D = i; // expected-error {{expected namespace name}}

namespace E = N::Foo; // expected-error {{expected namespace name}}
