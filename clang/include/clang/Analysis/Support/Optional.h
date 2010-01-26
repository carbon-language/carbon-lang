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

#ifndef LLVM_CLANG_ANALYSIS_OPTIONAL
#define LLVM_CLANG_ANALYSIS_OPTIONAL

namespace clang {

template<typename T>
class Optional {
  T x;
  unsigned hasVal : 1;
public:
  explicit Optional() : hasVal(false) {}
  Optional(const T &y) : x(y), hasVal(true) {}

  static inline Optional create(const T* y) {
    return y ? Optional(*y) : Optional();
  }

  Optional &operator=(const T &y) {
    x = y;
    hasVal = true;
    return *this;
  }
  
  const T* getPointer() const { assert(hasVal); return &x; }
  const T& getValue() const { assert(hasVal); return x; }

  operator bool() const { return hasVal; }
  bool hasValue() const { return hasVal; }
  const T* operator->() const { return getPointer(); }
  const T& operator*() const { assert(hasVal); return x; }
};
} //end clang namespace

namespace llvm {

template<typename T> struct simplify_type;
  
template <typename T>
struct simplify_type<const ::clang::Optional<T> > {
  typedef const T* SimpleType;
  static SimpleType getSimplifiedValue(const ::clang::Optional<T> &Val) {
    return Val.getPointer();
  }
};

template <typename T>
struct simplify_type< ::clang::Optional<T> >
  : public simplify_type<const ::clang::Optional<T> > {};
} // end llvm namespace

#endif
