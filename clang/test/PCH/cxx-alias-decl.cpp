// Test this without pch.
// RUN: %clang_cc1 -x c++ -std=c++0x -include %S/cxx-alias-decl.h -fsyntax-only -emit-llvm -o - %s

// Test with pch.
// RUN: %clang_cc1 -x c++ -std=c++0x -emit-pch -o %t %S/cxx-alias-decl.h
// RUN: %clang_cc1 -x c++ -std=c++0x -include-pch %t -fsyntax-only -emit-llvm -o - %s 

template struct T<S>;
C<A>::A<char> a;

using T1 = decltype(a);
using T1 = D<int, char>;

using T2 = B<A>;
using T2 = S;

using A = int;
template<typename U> using B = S;
template<typename U> using C = T<U>;
template<typename U, typename V> using D = typename T<U>::template A<V>;
