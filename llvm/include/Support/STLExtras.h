//===- STLExtras.h - Useful functions when working with the STL -*- C++ -*-===//
//
// This file contains some templates that are useful if you are working with the
// STL at all.
//
// No library is required when using these functinons.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_STLEXTRAS_H
#define SUPPORT_STLEXTRAS_H

#include <functional>
#include "Support/iterator"

//===----------------------------------------------------------------------===//
//     Extra additions to <functional>
//===----------------------------------------------------------------------===//

// bind_obj - Often times you want to apply the member function of an object
// as a unary functor.  This macro is shorthand that makes it happen less
// verbosely.
//
// Example:
//  struct Summer { void accumulate(int x); }
//  vector<int> Numbers;
//  Summer MyS;
//  for_each(Numbers.begin(), Numbers.end(),
//           bind_obj(&MyS, &Summer::accumulate));
//
// TODO: When I get lots of extra time, convert this from an evil macro
//
#define bind_obj(OBJ, METHOD) std::bind1st(std::mem_fun(METHOD), OBJ)


// bitwise_or - This is a simple functor that applys operator| on its two 
// arguments to get a boolean result.
//
template<class Ty>
struct bitwise_or : public std::binary_function<Ty, Ty, bool> {
  bool operator()(const Ty& left, const Ty& right) const {
    return left | right;
  }
};


// deleter - Very very very simple method that is used to invoke operator
// delete on something.  It is used like this: 
//
//   for_each(V.begin(), B.end(), deleter<Interval>);
//
template <class T> 
static inline void deleter(T *Ptr) { 
  delete Ptr; 
}



//===----------------------------------------------------------------------===//
//     Extra additions to <iterator>
//===----------------------------------------------------------------------===//

// mapped_iterator - This is a simple iterator adapter that causes a function to
// be dereferenced whenever operator* is invoked on the iterator.
//
// It turns out that this is disturbingly similar to boost::transform_iterator
//
#if 1
template <class RootIt, class UnaryFunc>
class mapped_iterator {
  RootIt current;
  UnaryFunc Fn;
public:
  typedef typename std::iterator_traits<RootIt>::iterator_category
          iterator_category;
  typedef typename std::iterator_traits<RootIt>::difference_type
          difference_type;
  typedef typename UnaryFunc::result_type value_type;

  typedef void pointer;
  //typedef typename UnaryFunc::result_type *pointer;
  typedef void reference;        // Can't modify value returned by fn

  typedef RootIt iterator_type;
  typedef mapped_iterator<RootIt, UnaryFunc> _Self;

  inline RootIt &getCurrent() const { return current; }

  inline explicit mapped_iterator(const RootIt &I, UnaryFunc F)
    : current(I), Fn(F) {}
  inline mapped_iterator(const mapped_iterator &It)
    : current(It.current), Fn(It.Fn) {}

  inline value_type operator*() const {   // All this work to do this 
    return Fn(*current);         // little change
  }

  _Self& operator++() { ++current; return *this; }
  _Self& operator--() { --current; return *this; }
  _Self  operator++(int) { _Self __tmp = *this; ++current; return __tmp; }
  _Self  operator--(int) { _Self __tmp = *this; --current; return __tmp; }
  _Self  operator+    (difference_type n) const { return _Self(current + n); }
  _Self& operator+=   (difference_type n) { current += n; return *this; }
  _Self  operator-    (difference_type n) const { return _Self(current - n); }
  _Self& operator-=   (difference_type n) { current -= n; return *this; }
  reference operator[](difference_type n) const { return *(*this + n); }  

  inline bool operator!=(const _Self &X) const { return !operator==(X); }
  inline bool operator==(const _Self &X) const { return current == X.current; }
  inline bool operator< (const _Self &X) const { return current <  X.current; }

  inline difference_type operator-(const _Self &X) const {
    return current - X.current;
  }
};

template <class _Iterator, class Func>
inline mapped_iterator<_Iterator, Func> 
operator+(typename mapped_iterator<_Iterator, Func>::difference_type N,
          const mapped_iterator<_Iterator, Func>& X) {
  return mapped_iterator<_Iterator, Func>(X.getCurrent() - N);
}

#else

// This fails to work, because some iterators are not classes, for example
// vector iterators are commonly value_type **'s
template <class RootIt, class UnaryFunc>
class mapped_iterator : public RootIt {
  UnaryFunc Fn;
public:
  typedef typename UnaryFunc::result_type value_type;
  typedef typename UnaryFunc::result_type *pointer;
  typedef void reference;        // Can't modify value returned by fn

  typedef mapped_iterator<RootIt, UnaryFunc> _Self;
  typedef RootIt super;
  inline explicit mapped_iterator(const RootIt &I) : super(I) {}
  inline mapped_iterator(const super &It) : super(It) {}

  inline value_type operator*() const {     // All this work to do 
    return Fn(super::operator*());   // this little thing
  }
};
#endif

// map_iterator - Provide a convenient way to create mapped_iterators, just like
// make_pair is useful for creating pairs...
//
template <class ItTy, class FuncTy>
inline mapped_iterator<ItTy, FuncTy> map_iterator(const ItTy &I, FuncTy F) {
  return mapped_iterator<ItTy, FuncTy>(I, F);
}


//===----------------------------------------------------------------------===//
//     Extra additions to <algorithm>
//===----------------------------------------------------------------------===//

// apply_until - Apply a functor to a sequence continually, unless the
// functor returns true.  Return true if the functor returned true, return false
// if the functor never returned true.
//
template <class InputIt, class Function>
bool apply_until(InputIt First, InputIt Last, Function Func) {
  for ( ; First != Last; ++First)
    if (Func(*First)) return true;
  return false;
}


// reduce - Reduce a sequence values into a single value, given an initial
// value and an operator.
//
template <class InputIt, class Function, class ValueType>
ValueType reduce(InputIt First, InputIt Last, Function Func, ValueType Value) {
  for ( ; First != Last; ++First)
    Value = Func(*First, Value);
  return Value;
}

#if 1   // This is likely to be more efficient

// reduce_apply - Reduce the result of applying a function to each value in a
// sequence, given an initial value, an operator, a function, and a sequence.
//
template <class InputIt, class Function, class ValueType, class TransFunc>
inline ValueType reduce_apply(InputIt First, InputIt Last, Function Func, 
			      ValueType Value, TransFunc XForm) {
  for ( ; First != Last; ++First)
    Value = Func(XForm(*First), Value);
  return Value;
}

#else  // This is arguably more elegant

// reduce_apply - Reduce the result of applying a function to each value in a
// sequence, given an initial value, an operator, a function, and a sequence.
//
template <class InputIt, class Function, class ValueType, class TransFunc>
inline ValueType reduce_apply2(InputIt First, InputIt Last, Function Func, 
			       ValueType Value, TransFunc XForm) {
  return reduce(map_iterator(First, XForm), map_iterator(Last, XForm),
		Func, Value);
}
#endif


// reduce_apply_bool - Reduce the result of applying a (bool returning) function
// to each value in a sequence.  All of the bools returned by the mapped
// function are bitwise or'd together, and the result is returned.
//
template <class InputIt, class Function>
inline bool reduce_apply_bool(InputIt First, InputIt Last, Function Func) {
  return reduce_apply(First, Last, bitwise_or<bool>(), false, Func);
}


// map - This function maps the specified input sequence into the specified
// output iterator, applying a unary function in between.
//
template <class InIt, class OutIt, class Functor>
inline OutIt mapto(InIt Begin, InIt End, OutIt Dest, Functor F) {
  return copy(map_iterator(Begin, F), map_iterator(End, F), Dest);
}
#endif
