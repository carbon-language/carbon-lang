// RUN: %clang_cc1 -fsyntax-only -verify %s 

// Don't crash.

template<typename aT>
struct basic_string{
  a; // expected-error {{requires a type specifier}}
  basic_string(aT*);
};

struct runtime_error{ // expected-note {{candidate constructor}}
  runtime_error( // expected-note {{candidate constructor}}
basic_string<char> struct{ // expected-error {{cannot combine with previous 'type-name' declaration specifier}}
a(){ // expected-error {{requires a type specifier}}
  runtime_error(0); // expected-error {{no matching conversion}}
}
}
);
};
