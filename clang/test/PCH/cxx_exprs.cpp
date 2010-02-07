// Test this without pch.
// RUN: %clang_cc1 -include %S/cxx_exprs.h -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -x c++-header -emit-pch -o %t %S/cxx_exprs.h
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify %s 

int integer;
double floating;
char character;

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
