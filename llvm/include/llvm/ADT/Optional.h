//===-- Optional.h - Simple variant for passing optional values ---*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file provides Optional, a template class modeled in the spirit of
//  OCaml's 'opt' variant.  The idea is to strongly type whether or not
//  a value can be optional.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_OPTIONAL_H
#define LLVM_ADT_OPTIONAL_H

#include "llvm/Support/Compiler.h"
#include "llvm/Support/AlignOf.h"
#include <cassert>

#if LLVM_HAS_RVALUE_REFERENCES
#include <utility>
#endif

namespace llvm {

template<typename T>
class Optional {
  AlignedCharArrayUnion<T> storage;
  bool hasVal;
public:
  explicit Optional() : hasVal(false) {}
  Optional(const T &y) : hasVal(true) {
    new (storage.buffer) T(y);
  }
  Optional(const Optional &O) : hasVal(O.hasVal) {
    if (hasVal)
      new (storage.buffer) T(*O);
  }

#if LLVM_HAS_RVALUE_REFERENCES
  Optional(T &&y) : hasVal(true) {
    new (storage.buffer) T(std::forward<T>(y));
  }
#endif

  static inline Optional create(const T* y) {
    return y ? Optional(*y) : Optional();
  }

  Optional &operator=(const T &y) {
    if (hasVal)
      **this = y;
    else {
      new (storage.buffer) T(y);
      hasVal = true;
    }
    return *this;
  }

  Optional &operator=(const Optional &O) {
    if (!O)
      reset();
    else
      *this = *O;
    return *this;
  }

  void reset() {
    if (hasVal) {
      (*this)->~T();
      hasVal = false;
    }
  }

  ~Optional() {
    reset();
  }
  
  const T* getPointer() const { assert(hasVal); return reinterpret_cast<const T*>(storage.buffer); }
  T* getPointer() { assert(hasVal); return reinterpret_cast<T*>(storage.buffer); }
  const T& getValue() const LLVM_LVALUE_FUNCTION { assert(hasVal); return *getPointer(); }
  T& getValue() LLVM_LVALUE_FUNCTION { assert(hasVal); return *getPointer(); }

  operator bool() const { return hasVal; }
  bool hasValue() const { return hasVal; }
  const T* operator->() const { return getPointer(); }
  T* operator->() { return getPointer(); }
  const T& operator*() const LLVM_LVALUE_FUNCTION { assert(hasVal); return *getPointer(); }
  T& operator*() LLVM_LVALUE_FUNCTION { assert(hasVal); return *getPointer(); }

#if LLVM_HAS_RVALUE_REFERENCE_THIS
  T&& getValue() && { assert(hasVal); return std::move(*getPointer()); }
  T&& operator*() && { assert(hasVal); return std::move(*getPointer()); }
#endif
};

template<typename T> struct simplify_type;

template <typename T>
struct simplify_type<const Optional<T> > {
  typedef const T* SimpleType;
  static SimpleType getSimplifiedValue(const Optional<T> &Val) {
    return Val.getPointer();
  }
};

template <typename T>
struct simplify_type<Optional<T> >
  : public simplify_type<const Optional<T> > {};

/// \brief Poison comparison between two \c Optional objects. Clients needs to
/// explicitly compare the underlying values and account for empty \c Optional
/// objects.
///
/// This routine will never be defined. It returns \c void to help diagnose 
/// errors at compile time.
template<typename T, typename U>
void operator==(const Optional<T> &X, const Optional<U> &Y);

/// \brief Poison comparison between two \c Optional objects. Clients needs to
/// explicitly compare the underlying values and account for empty \c Optional
/// objects.
///
/// This routine will never be defined. It returns \c void to help diagnose 
/// errors at compile time.
template<typename T, typename U>
void operator!=(const Optional<T> &X, const Optional<U> &Y);

/// \brief Poison comparison between two \c Optional objects. Clients needs to
/// explicitly compare the underlying values and account for empty \c Optional
/// objects.
///
/// This routine will never be defined. It returns \c void to help diagnose 
/// errors at compile time.
template<typename T, typename U>
void operator<(const Optional<T> &X, const Optional<U> &Y);

/// \brief Poison comparison between two \c Optional objects. Clients needs to
/// explicitly compare the underlying values and account for empty \c Optional
/// objects.
///
/// This routine will never be defined. It returns \c void to help diagnose 
/// errors at compile time.
template<typename T, typename U>
void operator<=(const Optional<T> &X, const Optional<U> &Y);

/// \brief Poison comparison between two \c Optional objects. Clients needs to
/// explicitly compare the underlying values and account for empty \c Optional
/// objects.
///
/// This routine will never be defined. It returns \c void to help diagnose 
/// errors at compile time.
template<typename T, typename U>
void operator>=(const Optional<T> &X, const Optional<U> &Y);

/// \brief Poison comparison between two \c Optional objects. Clients needs to
/// explicitly compare the underlying values and account for empty \c Optional
/// objects.
///
/// This routine will never be defined. It returns \c void to help diagnose 
/// errors at compile time.
template<typename T, typename U>
void operator>(const Optional<T> &X, const Optional<U> &Y);

} // end llvm namespace

#endif
