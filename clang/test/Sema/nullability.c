// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-nullability-declspec %s -verify

#if __has_feature(nullability)
#else
#  error nullability feature should be defined
#endif

typedef int * int_ptr;

// Parse nullability type specifiers.
typedef int * __nonnull nonnull_int_ptr; // expected-note{{'__nonnull' specified here}}
typedef int * __nullable nullable_int_ptr;
typedef int * __null_unspecified null_unspecified_int_ptr;

// Redundant nullability type specifiers.
typedef int * __nonnull __nonnull redundant_1; // expected-warning{{duplicate nullability specifier '__nonnull'}}

// Conflicting nullability type specifiers.
typedef int * __nonnull __nullable conflicting_1; // expected-error{{nullability specifier '__nonnull' conflicts with existing specifier '__nullable'}}
typedef int * __null_unspecified __nonnull conflicting_2; // expected-error{{nullability specifier '__null_unspecified' conflicts with existing specifier '__nonnull'}}

// Redundant nullability specifiers via a typedef are okay.
typedef nonnull_int_ptr __nonnull redundant_okay_1;

// Conflicting nullability specifiers via a typedef are not.
typedef nonnull_int_ptr __nullable conflicting_2; // expected-error{{nullability specifier '__nullable' conflicts with existing specifier '__nonnull'}}
typedef nonnull_int_ptr nonnull_int_ptr_typedef;
typedef nonnull_int_ptr_typedef __nullable conflicting_2; // expected-error{{nullability specifier '__nullable' conflicts with existing specifier '__nonnull'}}
typedef nonnull_int_ptr_typedef nonnull_int_ptr_typedef_typedef;
typedef nonnull_int_ptr_typedef_typedef __null_unspecified conflicting_3; // expected-error{{nullability specifier '__null_unspecified' conflicts with existing specifier '__nonnull'}}

// Nullability applies to all pointer types.
typedef int (* __nonnull function_pointer_type_1)(int, int);
typedef int (^ __nonnull block_type_1)(int, int);

// Nullability must be on a pointer type.
typedef int __nonnull int_type_1; // expected-error{{nullability specifier '__nonnull' cannot be applied to non-pointer type 'int'}}

// Nullability can move out to a pointer/block pointer declarator
// (with a suppressed warning).
typedef __nonnull int * nonnull_int_ptr_2;
typedef int __nullable * nullable_int_ptr_2;
typedef __nonnull int (* function_pointer_type_2)(int, int);
typedef __nonnull int (^ block_type_2)(int, int);
typedef __nonnull int * * __nullable nonnull_int_ptr_ptr_1;
typedef __nonnull int *(^ block_type_3)(int, int);
typedef __nonnull int *(* function_pointer_type_3)(int, int);
typedef __nonnull int_ptr (^ block_type_4)(int, int);
typedef __nonnull int_ptr (* function_pointer_type_4)(int, int);

void acceptFunctionPtr(__nonnull int *(*)(void));
void acceptBlockPtr(__nonnull int *(^)(void));

void testBlockFunctionPtrNullability() {
  float *fp;
  fp = (function_pointer_type_3)0; // expected-warning{{from 'function_pointer_type_3' (aka 'int * __nonnull (*)(int, int)')}}
  fp = (block_type_3)0; // expected-error{{from incompatible type 'block_type_3' (aka 'int * __nonnull (^)(int, int)')}}
  fp = (function_pointer_type_4)0; // expected-warning{{from 'function_pointer_type_4' (aka 'int_ptr  __nonnull (*)(int, int)')}}
  fp = (block_type_4)0; // expected-error{{from incompatible type 'block_type_4' (aka 'int_ptr  __nonnull (^)(int, int)')}}

  acceptFunctionPtr(0); // no-warning
  acceptBlockPtr(0); // no-warning
}

// Moving nullability where it creates a conflict.
typedef __nonnull int * __nullable *  conflict_int_ptr_ptr_2; // expected-error{{nullability specifier '__nonnull' cannot be applied to non-pointer type 'int'}}

// Nullability is not part of the canonical type.
typedef int * __nonnull ambiguous_int_ptr;
typedef int * ambiguous_int_ptr;
typedef int * __nullable ambiguous_int_ptr;

// Printing of nullability.
float f;
int * __nonnull ip_1 = &f; // expected-warning{{incompatible pointer types initializing 'int * __nonnull' with an expression of type 'float *'}}

// Check printing of nullability specifiers.
void printing_nullability(void) {
  int * __nonnull iptr;
  float *fptr = iptr; // expected-warning{{incompatible pointer types initializing 'float *' with an expression of type 'int * __nonnull'}}

  int * * __nonnull iptrptr;
  float **fptrptr = iptrptr; // expected-warning{{incompatible pointer types initializing 'float **' with an expression of type 'int ** __nonnull'}}

  int * __nullable * __nonnull iptrptr2;
  float * *fptrptr2 = iptrptr2; // expected-warning{{incompatible pointer types initializing 'float **' with an expression of type 'int * __nullable * __nonnull'}}
}
