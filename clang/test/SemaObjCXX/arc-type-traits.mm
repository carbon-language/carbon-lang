// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -fobjc-runtime-has-weak -verify %s

// Check the results of the various type-trait query functions on
// lifetime-qualified types in ARC.

#define JOIN3(X,Y) X ## Y
#define JOIN2(X,Y) JOIN3(X,Y)
#define JOIN(X,Y) JOIN2(X,Y)

#define TRAIT_IS_TRUE(Trait, Type) char JOIN2(Trait,__LINE__)[Trait(Type)? 1 : -1]
#define TRAIT_IS_FALSE(Trait, Type) char JOIN2(Trait,__LINE__)[Trait(Type)? -1 : 1]
  
// __has_nothrow_assign
TRAIT_IS_TRUE(__has_nothrow_assign, __strong id);
TRAIT_IS_TRUE(__has_nothrow_assign, __weak id);
TRAIT_IS_TRUE(__has_nothrow_assign, __autoreleasing id);
TRAIT_IS_TRUE(__has_nothrow_assign, __unsafe_unretained id);

// __has_nothrow_copy
TRAIT_IS_TRUE(__has_nothrow_copy, __strong id);
TRAIT_IS_TRUE(__has_nothrow_copy, __weak id);
TRAIT_IS_TRUE(__has_nothrow_copy, __autoreleasing id);
TRAIT_IS_TRUE(__has_nothrow_copy, __unsafe_unretained id);

// __has_nothrow_constructor
TRAIT_IS_TRUE(__has_nothrow_constructor, __strong id);
TRAIT_IS_TRUE(__has_nothrow_constructor, __weak id);
TRAIT_IS_TRUE(__has_nothrow_constructor, __autoreleasing id);
TRAIT_IS_TRUE(__has_nothrow_constructor, __unsafe_unretained id);

// __has_trivial_assign
TRAIT_IS_FALSE(__has_trivial_assign, __strong id);
TRAIT_IS_FALSE(__has_trivial_assign, __weak id);
TRAIT_IS_FALSE(__has_trivial_assign, __autoreleasing id);
TRAIT_IS_TRUE(__has_trivial_assign, __unsafe_unretained id);

// __has_trivial_copy
TRAIT_IS_FALSE(__has_trivial_copy, __strong id);
TRAIT_IS_FALSE(__has_trivial_copy, __weak id);
TRAIT_IS_FALSE(__has_trivial_copy, __autoreleasing id);
TRAIT_IS_TRUE(__has_trivial_copy, __unsafe_unretained id);

// __has_trivial_constructor
TRAIT_IS_FALSE(__has_trivial_constructor, __strong id);
TRAIT_IS_FALSE(__has_trivial_constructor, __weak id);
TRAIT_IS_FALSE(__has_trivial_constructor, __autoreleasing id);
TRAIT_IS_TRUE(__has_trivial_constructor, __unsafe_unretained id);

// __has_trivial_destructor
TRAIT_IS_FALSE(__has_trivial_destructor, __strong id);
TRAIT_IS_FALSE(__has_trivial_destructor, __weak id);
TRAIT_IS_TRUE(__has_trivial_destructor, __autoreleasing id);
TRAIT_IS_TRUE(__has_trivial_destructor, __unsafe_unretained id);

// __is_literal
TRAIT_IS_TRUE(__is_literal, __strong id);
TRAIT_IS_TRUE(__is_literal, __weak id);
TRAIT_IS_TRUE(__is_literal, __autoreleasing id);
TRAIT_IS_TRUE(__is_literal, __unsafe_unretained id);

// __is_literal_type
TRAIT_IS_TRUE(__is_literal_type, __strong id);
TRAIT_IS_TRUE(__is_literal_type, __weak id);
TRAIT_IS_TRUE(__is_literal_type, __autoreleasing id);
TRAIT_IS_TRUE(__is_literal_type, __unsafe_unretained id);

// __is_pod
TRAIT_IS_FALSE(__is_pod, __strong id);
TRAIT_IS_FALSE(__is_pod, __weak id);
TRAIT_IS_FALSE(__is_pod, __autoreleasing id);
TRAIT_IS_TRUE(__is_pod, __unsafe_unretained id);

// __is_trivial
TRAIT_IS_FALSE(__is_trivial, __strong id);
TRAIT_IS_FALSE(__is_trivial, __weak id);
TRAIT_IS_FALSE(__is_trivial, __autoreleasing id);
TRAIT_IS_TRUE(__is_trivial, __unsafe_unretained id);

// __is_scalar
TRAIT_IS_FALSE(__is_scalar, __strong id);
TRAIT_IS_FALSE(__is_scalar, __weak id);
TRAIT_IS_FALSE(__is_scalar, __autoreleasing id);
TRAIT_IS_TRUE(__is_scalar, __unsafe_unretained id);

// __is_standard_layout
TRAIT_IS_TRUE(__is_standard_layout, __strong id);
TRAIT_IS_TRUE(__is_standard_layout, __weak id);
TRAIT_IS_TRUE(__is_standard_layout, __autoreleasing id);
TRAIT_IS_TRUE(__is_standard_layout, __unsafe_unretained id);

