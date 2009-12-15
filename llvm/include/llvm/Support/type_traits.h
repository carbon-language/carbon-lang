//===- llvm/Support/type_traits.h - Simplfied type traits -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides a template class that determines if a type is a class or
// not. The basic mechanism, based on using the pointer to member function of
// a zero argument to a function was "boosted" from the boost type_traits
// library. See http://www.boost.org/ for all the gory details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_TYPE_TRAITS_H
#define LLVM_SUPPORT_TYPE_TRAITS_H

#include <utility>

// This is actually the conforming implementation which works with abstract
// classes.  However, enough compilers have trouble with it that most will use
// the one in boost/type_traits/object_traits.hpp. This implementation actually
// works with VC7.0, but other interactions seem to fail when we use it.

namespace llvm {
  
namespace dont_use
{
    // These two functions should never be used. They are helpers to
    // the is_class template below. They cannot be located inside
    // is_class because doing so causes at least GCC to think that
    // the value of the "value" enumerator is not constant. Placing
    // them out here (for some strange reason) allows the sizeof
    // operator against them to magically be constant. This is
    // important to make the is_class<T>::value idiom zero cost. it
    // evaluates to a constant 1 or 0 depending on whether the
    // parameter T is a class or not (respectively).
    template<typename T> char is_class_helper(void(T::*)());
    template<typename T> double is_class_helper(...);
}

template <typename T>
struct is_class
{
  // is_class<> metafunction due to Paul Mensonides (leavings@attbi.com). For
  // more details:
  // http://groups.google.com/groups?hl=en&selm=000001c1cc83%24e154d5e0%247772e50c%40c161550a&rnum=1
 public:
    enum { value = sizeof(char) == sizeof(dont_use::is_class_helper<T>(0)) };
};
  
  
/// isPodLike - This is a type trait that is used to determine whether a given
/// type can be copied around with memcpy instead of running ctors etc.
template <typename T>
struct isPodLike {
  // If we don't know anything else, we can (at least) assume that all non-class
  // types are PODs.
  static const bool value = !is_class<T>::value;
};

// std::pair's are pod-like if their elements are.
template<typename T, typename U>
struct isPodLike<std::pair<T, U> > {
  static const bool value = isPodLike<T>::value & isPodLike<U>::value;
};
  

/// \brief Metafunction that determines whether the two given types are 
/// equivalent.
template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};
  
// enable_if_c - Enable/disable a template based on a metafunction
template<bool Cond, typename T = void>
struct enable_if_c {
  typedef T type;
};

template<typename T> struct enable_if_c<false, T> { };
  
// enable_if - Enable/disable a template based on a metafunction
template<typename Cond, typename T = void>
struct enable_if : public enable_if_c<Cond::value, T> { };

namespace dont_use {
  template<typename Base> char base_of_helper(const volatile Base*);
  template<typename Base> double base_of_helper(...);
}

/// is_base_of - Metafunction to determine whether one type is a base class of
/// (or identical to) another type.
template<typename Base, typename Derived>
struct is_base_of {
  static const bool value 
    = is_class<Base>::value && is_class<Derived>::value &&
      sizeof(char) == sizeof(dont_use::base_of_helper<Base>((Derived*)0));
};

// remove_pointer - Metafunction to turn Foo* into Foo.  Defined in
// C++0x [meta.trans.ptr].
template <typename T> struct remove_pointer { typedef T type; };
template <typename T> struct remove_pointer<T*> { typedef T type; };
template <typename T> struct remove_pointer<T*const> { typedef T type; };
template <typename T> struct remove_pointer<T*volatile> { typedef T type; };
template <typename T> struct remove_pointer<T*const volatile> {
    typedef T type; };

template <bool, typename T, typename F>
struct conditional { typedef T type; };

template <typename T, typename F>
struct conditional<false, T, F> { typedef F type; };

}

#endif
