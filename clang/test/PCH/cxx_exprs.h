// Header for PCH test cxx_exprs.cpp

// CXXStaticCastExpr
typedef typeof(static_cast<void *>(0)) static_cast_result;

// CXXDynamicCastExpr
struct Base { virtual void f(); };
struct Derived : Base { };
Base *base_ptr;
typedef typeof(dynamic_cast<Derived *>(base_ptr)) dynamic_cast_result;

// CXXReinterpretCastExpr
typedef typeof(reinterpret_cast<void *>(0)) reinterpret_cast_result;

// CXXConstCastExpr
const char *const_char_ptr_value;
typedef typeof(const_cast<char *>(const_char_ptr_value)) const_cast_result;

// CXXFunctionalCastExpr
int int_value;
typedef typeof(double(int_value)) functional_cast_result;
