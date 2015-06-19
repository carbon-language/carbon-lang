// RUN: %clang_cc1 -std=c++11 -fsyntax-only -Wno-nullability-declspec %s -verify

typedef decltype(nullptr) nullptr_t;

class X {
};

// Nullability applies to all pointer types.
typedef int (X::* __nonnull member_function_type_1)(int);
typedef int X::* __nonnull member_data_type_1;
typedef nullptr_t __nonnull nonnull_nullptr_t; // expected-error{{nullability specifier '__nonnull' cannot be applied to non-pointer type 'nullptr_t'}}

// Nullability can move into member pointers (this is suppressing a warning).
typedef __nonnull int (X::* member_function_type_2)(int);
typedef int (X::* __nonnull member_function_type_3)(int);
typedef __nonnull int X::* member_data_type_2;

// Adding non-null via a template.
template<typename T>
struct AddNonNull {
  typedef __nonnull T type; // expected-error{{nullability specifier '__nonnull' cannot be applied to non-pointer type 'int'}}
  // expected-error@-1{{nullability specifier '__nonnull' cannot be applied to non-pointer type 'nullptr_t'}}
};

typedef AddNonNull<int *>::type nonnull_int_ptr_1;
typedef AddNonNull<int * __nullable>::type nonnull_int_ptr_2; // FIXME: check that it was overridden
typedef AddNonNull<nullptr_t>::type nonnull_int_ptr_3; // expected-note{{in instantiation of template class}}

typedef AddNonNull<int>::type nonnull_non_pointer_1; // expected-note{{in instantiation of template class 'AddNonNull<int>' requested here}}

// Non-null checking within a template.
template<typename T>
struct AddNonNull2 {
  typedef __nonnull AddNonNull<T> invalid1; // expected-error{{nullability specifier '__nonnull' cannot be applied to non-pointer type 'AddNonNull<T>'}}
  typedef __nonnull AddNonNull2 invalid2; // expected-error{{nullability specifier '__nonnull' cannot be applied to non-pointer type 'AddNonNull2<T>'}}
  typedef __nonnull AddNonNull2<T> invalid3; // expected-error{{nullability specifier '__nonnull' cannot be applied to non-pointer type 'AddNonNull2<T>'}}
  typedef __nonnull typename AddNonNull<T>::type okay1;

  // Don't move past a dependent type even if we know that nullability
  // cannot apply to that specific dependent type.
  typedef __nonnull AddNonNull<T> (*invalid4); // expected-error{{nullability specifier '__nonnull' cannot be applied to non-pointer type 'AddNonNull<T>'}}
};

// Check passing null to a __nonnull argument.
void (*accepts_nonnull_1)(__nonnull int *ptr);
void (*& accepts_nonnull_2)(__nonnull int *ptr) = accepts_nonnull_1;
void (X::* accepts_nonnull_3)(__nonnull int *ptr);
void accepts_nonnull_4(__nonnull int *ptr);
void (&accepts_nonnull_5)(__nonnull int *ptr) = accepts_nonnull_4;

void test_accepts_nonnull_null_pointer_literal(X *x) {
  accepts_nonnull_1(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
  accepts_nonnull_2(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
  (x->*accepts_nonnull_3)(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
  accepts_nonnull_4(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
  accepts_nonnull_5(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
}

template<void FP(__nonnull int*)> 
void test_accepts_nonnull_null_pointer_literal_template() {
  FP(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
}

template void test_accepts_nonnull_null_pointer_literal_template<&accepts_nonnull_4>(); // expected-note{{instantiation of function template specialization}}
