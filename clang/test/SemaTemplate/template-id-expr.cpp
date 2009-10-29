// RUN: clang-cc -fsyntax-only -verify %s

// PR5336
template<typename FromCl>
struct isa_impl_cl {
 template<class ToCl>
 static void isa(const FromCl &Val) { }
};

template<class X, class Y>
void isa(const Y &Val) {   return isa_impl_cl<Y>::template isa<X>(Val); }

class Value;
void f0(const Value &Val) { isa<Value>(Val); }
