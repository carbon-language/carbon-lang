//===- iterator.h - Utilities for using and defining iterators --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_ITERATOR_H
#define LLVM_ADT_ITERATOR_H

#include <iterator>

namespace llvm {

/// \brief CRTP base class for adapting an iterator to a different type.
///
/// This class can be used through CRTP to adapt one iterator into another.
/// Typically this is done through providing in the derived class a custom \c
/// operator* implementation. Other methods can be overridden as well.
///
/// FIXME: Factor out the iterator-facade-like aspects into a base class that
/// can be used for defining completely custom iterators.
template <typename DerivedT, typename WrappedIteratorT, typename T,
          typename PointerT = T *, typename ReferenceT = T &,
          // Don't provide these, they are mostly to act as aliases below.
          typename WrappedTraitsT = std::iterator_traits<WrappedIteratorT>>
class iterator_adaptor_base
    : public std::iterator<typename WrappedTraitsT::iterator_category, T,
                           typename WrappedTraitsT::difference_type, PointerT,
                           ReferenceT> {
protected:
  WrappedIteratorT I;

  iterator_adaptor_base() {}

  template <
      typename U,
      typename = typename std::enable_if<
          !std::is_same<typename std::remove_cv<
                            typename std::remove_reference<U>::type>::type,
                        DerivedT>::value>::type>
  explicit iterator_adaptor_base(U &&u)
      : I(std::forward<U &&>(u)) {}

public:
  typedef typename iterator_adaptor_base::iterator::difference_type
  difference_type;

  DerivedT &operator+=(difference_type n) {
    I += n;
    return *static_cast<DerivedT *>(this);
  }
  DerivedT &operator-=(difference_type n) {
    I -= n;
    return *static_cast<DerivedT *>(this);
  }
  DerivedT operator+(difference_type n) const {
    DerivedT tmp = *this;
    tmp += n;
    return tmp;
  }
  friend DerivedT operator+(difference_type n, const DerivedT &i) {
    return i + n;
  }
  DerivedT operator-(difference_type n) const {
    DerivedT tmp = *this;
    tmp -= n;
    return tmp;
  }
  difference_type operator-(const DerivedT &RHS) const { return I - RHS.I; }

  DerivedT &operator++() {
    ++I;
    return *static_cast<DerivedT *>(this);
  }
  DerivedT &operator--() {
    --I;
    return *static_cast<DerivedT *>(this);
  }
  DerivedT operator++(int) {
    DerivedT tmp = *static_cast<DerivedT *>(this);
    ++*this;
    return tmp;
  }
  DerivedT operator--(int) {
    DerivedT tmp = *static_cast<DerivedT *>(this);
    --*this;
    return tmp;
  }

  bool operator==(const DerivedT &RHS) const { return I == RHS.I; }
  bool operator!=(const DerivedT &RHS) const {
    return !static_cast<const DerivedT *>(this)->operator==(RHS);
  }

  bool operator<(const DerivedT &RHS) const { return I < RHS.I; }
  bool operator>(const DerivedT &RHS) const {
    return !static_cast<const DerivedT *>(this)->operator<(RHS) &&
           !static_cast<const DerivedT *>(this)->operator==(RHS);
  }
  bool operator<=(const DerivedT &RHS) const {
    return !static_cast<const DerivedT *>(this)->operator>(RHS);
  }
  bool operator>=(const DerivedT &RHS) const {
    return !static_cast<const DerivedT *>(this)->operator<(RHS);
  }

  ReferenceT operator*() const { return *I; }
  PointerT operator->() const {
    return static_cast<const DerivedT *>(this)->operator*();
  }
  ReferenceT operator[](difference_type n) const {
    return *static_cast<const DerivedT *>(this)->operator+(n);
  }
};

/// \brief An iterator type that allows iterating over the pointees via some
/// other iterator.
///
/// The typical usage of this is to expose a type that iterates over Ts, but
/// which is implemented with some iterator over T*s:
///
/// \code
///   typedef pointee_iterator<SmallVectorImpl<T *>::iterator> iterator;
/// \endcode
template <
    typename WrappedIteratorT,
    typename T = typename std::remove_pointer<
        typename std::iterator_traits<WrappedIteratorT>::value_type>::type>
struct pointee_iterator
    : iterator_adaptor_base<pointee_iterator<WrappedIteratorT>,
                            WrappedIteratorT, T> {
  pointee_iterator() {}
  template <typename U>
  pointee_iterator(U &&u)
      : pointee_iterator::iterator_adaptor_base(std::forward<U &&>(u)) {}

  T &operator*() const { return **this->I; }
};

}

#endif
