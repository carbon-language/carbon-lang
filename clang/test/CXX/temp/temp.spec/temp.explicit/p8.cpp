// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T>
struct X0 {
  struct MemberClass;
  
  T* f0(T* ptr);
  
  static T* static_member;
};

template class X0<int>; // okay
template class X0<int(int)>; // okay; nothing gets instantiated.

template<typename T>
struct X0<T>::MemberClass {
  T member;
};

template<typename T>
T* X0<T>::f0(T* ptr) {
  return ptr + 1;
}

template<typename T>
T* X0<T>::static_member = 0;

