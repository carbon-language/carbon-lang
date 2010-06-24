// Test this without pch.
// RUN: %clang_cc1 -include %S/cxx_exprs.h -std=c++0x -fsyntax-only -verify %s -ast-dump

// Test with pch. Use '-ast-dump' to force deserialization of function bodies.
// RUN: %clang_cc1 -x c++-header -std=c++0x -emit-pch -o %t %S/cxx_exprs.h
// RUN: %clang_cc1 -std=c++0x -include-pch %t -fsyntax-only -verify %s -ast-dump

int integer;
double floating;
char character;
bool boolean;

// CXXStaticCastExpr
static_cast_result void_ptr = &integer;

// CXXDynamicCastExpr
Derived *d;
dynamic_cast_result derived_ptr = d;

// CXXReinterpretCastExpr
reinterpret_cast_result void_ptr2 = &integer;

// CXXConstCastExpr
const_cast_result char_ptr = &character;

// CXXFunctionalCastExpr
functional_cast_result *double_ptr = &floating;

// CXXBoolLiteralExpr
bool_literal_result *bool_ptr = &boolean;
static_assert(true_value, "true_value is true");
static_assert(!false_value, "false_value is false");

// CXXNullPtrLiteralExpr
cxx_null_ptr_result null_ptr = nullptr;

// CXXTypeidExpr
typeid_result1 typeid_1 = 0;
typeid_result2 typeid_2 = 0;
