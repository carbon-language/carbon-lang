#ifndef FLANG_SUPPORT_GET_VALUE_H

#undef IGNORE_optional
#undef IGNORE_Statement
#undef IGNORE_Scalar
#undef IGNORE_Constant
#undef IGNORE_Indirection
#undef IGNORE_Logical
#undef IGNORE_DefaultChar

// Each include of "GetValue.def" provides a set of helper functions
// whose names are specified by the macros GET_VALUE, HAS_VALUE and 
// GET_OPT_VALUE.
//
// The purpose of those function is to provide easier access to the
// parse-tree by ignoring some wrapper classes/
//
//
// GET_VALUE(x) provides a reference to the value that x is holding.
//
// The following wrapper classes are ignored unless the corresponding
// IGNORES_xxx macro is defined. 
//
//    Scalar<T>
//    Constant<T>
//    Integer<T>
//    Logical<T>
//    DefaultChar<T>
//    Indirection<T>
//    Statement<T>
//    std::optional<T>
//
//
// HAS_VALUE(x) return true if it is legal to call GET_VALUE(x) in case x 
// contains some std::optional<T>  
//
// Example:
//   Constant<std::optional<Indirection<std::optional<int>>>> &x = ... ;
//   if ( HasValue(x) ) {
//     const int &v = getValue(x) ; 
//     ... 
//   } 
// 
// GET_OPT_VALUE(T &x) is equivalent to
//
//    HAS_VALUE(x) ? &GET_VALUE(x) : (Type*) nullptr
//
// here Type is the type of GET_VALUE(x) 
//
// Example:  
//
//  const Scalar<optional<Integer<Expr>>> & z = ... 
//  const Expr *ptr_z = GET_OPT_VALUE(z) ; 
//  if ( ptr_z ) {
//    ... do something with *ptr_z %
//  }
//

// This is the default version that handles all wrapper

#define GET_VALUE GetValue
#define HAS_VALUE HasValue
#define GET_OPT_VALUE GetOptValue
#include "GetValue.def"

// HAS_VALUE and GET_OPT_VALUE are only interesting when
// std::optional is not ignored. 
// We need to give a name to the function but they are pretty much useless
#define IGNORE_optional
#define GET_VALUE GetOptionalValue
#define HAS_VALUE HasOptionalValue__
#define GET_OPT_VALUE GetOptValue__
#include "GetValue.def"
#undef IGNORE_optional

#define IGNORE_Statement
#define GET_VALUE GetStatementValue
#define HAS_VALUE HasStatementValue
#define GET_OPT_VALUE GetOptStatementValue
#include "GetValue.def"
#undef IGNORE_Statement

#define IGNORE_Scalar
#define GET_VALUE GetScalarValue
#define HAS_VALUE HasScalarValue
#define GET_OPT_VALUE GetOptScalarValue
#include "GetValue.def"
#undef IGNORE_Scalar

#define IGNORE_Constant
#define GET_VALUE GetConstantValue
#define HAS_VALUE HasConstantValue
#define GET_OPT_VALUE GetOptConstantValue
#include "GetValue.def"
#undef IGNORE_Constant

#define IGNORE_Indirection
#define GET_VALUE GetIndirectionValue
#define HAS_VALUE HasIndirectionValue
#define GET_OPT_VALUE GetOptIndirectionValue
#include "GetValue.def"
#undef IGNORE_Indirection


#define IGNORE_Logical
#define GET_VALUE GetLogicalValue
#define HAS_VALUE HasLogicalValue
#define GET_OPT_VALUE GetOptLogicalValue
#include "GetValue.def"
#undef IGNORE_Logical

#define IGNORE_DefaultChar
#define GET_VALUE GetDefaultCharValue
#define HAS_VALUE HasDefaultCharValue
#define GET_OPT_VALUE GetOptDefaultCharValue
#include "GetValue.def"
#undef IGNORE_DefaultChar

#endif
