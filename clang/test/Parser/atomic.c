// RUN: %clang_cc1 -std=c11 %s -fsyntax-only -verify -pedantic

typedef _Atomic(int) atomic_int;
typedef _Atomic int atomic_int;
typedef _Atomic _Atomic _Atomic(int) atomic_int; // expected-warning {{duplicate '_Atomic' declaration specifier}}

typedef const int const_int;

typedef const atomic_int const_atomic_int;
typedef _Atomic const int const_atomic_int;
typedef const _Atomic int const_atomic_int;
typedef const _Atomic(int) const_atomic_int;
typedef const _Atomic(_Atomic int) const_atomic_int; // expected-error {{_Atomic cannot be applied to atomic type '_Atomic(int)'}}
typedef _Atomic const_int const_atomic_int;
typedef _Atomic(const_int) const_atomic_int; // expected-error {{_Atomic cannot be applied to qualified type 'const_int' (aka 'const int')}}

typedef int *_Atomic atomic_int_ptr;
typedef _Atomic(int *) atomic_int_ptr;
typedef int (*_Atomic atomic_int_ptr);

typedef int _Atomic *int_atomic_ptr;
typedef _Atomic(int) *int_atomic_ptr;

typedef int int_fn();
typedef _Atomic int_fn atomic_int_fn; // expected-error {{_Atomic cannot be applied to function type 'int_fn' (aka 'int ()')}}
typedef _Atomic int atomic_int_array[3];
typedef _Atomic atomic_int_array atomic_int_atomic_array; // expected-error {{_Atomic cannot be applied to array type 'atomic_int_array' (aka '_Atomic(int) [3]')}}

_Atomic struct S { int n; }; // expected-warning {{'_Atomic' ignored on this declaration}}

typedef _Atomic int __attribute__((address_space(1))) atomic_addr_space_int;
typedef _Atomic(int) __attribute__((address_space(1))) atomic_addr_space_int;

typedef _Atomic int __attribute__((vector_size(16))) atomic_vector_int;
typedef _Atomic(int __attribute__((vector_size(16)))) atomic_vector_int;

struct S
_Atomic atomic_s_no_missing_semicolon;

int *const _Atomic atomic_return_type();
