// RUN: %clang_cc1 -fsyntax-only -verify %s

template<class T> class Array { /* ... */ }; 
template<class T> void sort(Array<T>& v) { }

// instantiate sort(Array<int>&) - template-argument deduced
template void sort<>(Array<int>&);

template void sort(Array<long>&);

template<typename T, typename U> void f0(T, U*) { }

template void f0<int>(int, float*);
template void f0<>(double, float*);
