// Header for PCH test cxx_exprs.cpp

// CXXStaticCastExpr
typedef __typeof__(static_cast<void *>(0)) static_cast_result;

// CXXDynamicCastExpr
struct Base { virtual void f(); };
struct Derived : Base { };
Base *base_ptr;
typedef __typeof__(dynamic_cast<Derived *>(base_ptr)) dynamic_cast_result;

// CXXReinterpretCastExpr
typedef __typeof__(reinterpret_cast<void *>(0)) reinterpret_cast_result;

// CXXConstCastExpr
const char *const_char_ptr_value;
typedef __typeof__(const_cast<char *>(const_char_ptr_value)) const_cast_result;

// CXXFunctionalCastExpr
int int_value;
typedef __typeof__(double(int_value)) functional_cast_result;

// CXXBoolLiteralExpr
typedef __typeof__(true) bool_literal_result;
const bool true_value = true;
const bool false_value = false;

// CXXNullPtrLiteralExpr
typedef __typeof__(nullptr) cxx_null_ptr_result;
