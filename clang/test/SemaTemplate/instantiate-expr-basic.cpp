// RUN: clang-cc -fsyntax-only -Wno-unused-value -std=c++0x %s

template <typename T>
struct S {
  void f() {
    __func__; // PredefinedExpr
    10;       // IntegerLiteral
    10.5;     // FloatingLiteral
    'c';      // CharacterLiteral
    "hello";  // StringLiteral
    true;     // CXXBooleanLiteralExpr
    nullptr;  // CXXNullPtrLiteralExpr
    __null;   // GNUNullExpr
  }
};

template struct S<int>;
