//===- llvm/ADT/STLExtras.h - Useful STL related functions ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains some templates that are useful if you are working with the
// STL at all.
//
// No library is required when using these functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STLEXTRAS_H
#define LLVM_ADT_STLEXTRAS_H

#include "llvm/Support/Compiler.h"
#include <cstddef> // for std::size_t
#include <cstdlib> // for qsort
#include <functional>
#include <iterator>
#include <memory>
#include <utility> // for std::pair

namespace llvm {

//===----------------------------------------------------------------------===//
//     Extra additions to <functional>
//===----------------------------------------------------------------------===//

template<class Ty>
struct identity : public std::unary_function<Ty, Ty> {
  Ty &operator()(Ty &self) const {
    return self;
  }
  const Ty &operator()(const Ty &self) const {
    return self;
  }
};

template<class Ty>
struct less_ptr : public std::binary_function<Ty, Ty, bool> {
  bool operator()(const Ty* left, const Ty* right) const {
    return *left < *right;
  }
};

template<class Ty>
struct greater_ptr : public std::binary_function<Ty, Ty, bool> {
  bool operator()(const Ty* left, const Ty* right) const {
    return *right < *left;
  }
};

/// An efficient, type-erasing, non-owning reference to a callable. This is
/// intended for use as the type of a function parameter that is not used
/// after the function in question returns.
///
/// This class does not own the callable, so it is not in general safe to store
/// a function_ref.
template<typename Fn> class function_ref;

#if LLVM_HAS_VARIADIC_TEMPLATES

template<typename Ret, typename ...Params>
class function_ref<Ret(Params...)> {
  Ret (*callback)(intptr_t callable, Params ...params);
  intptr_t callable;

  template<typename Callable>
  static Ret callback_fn(intptr_t callable, Params ...params) {
    return (*reinterpret_cast<Callable*>(callable))(
        std::forward<Params>(params)...);
  }

public:
  template<typename Callable>
  function_ref(Callable &&callable)
      : callback(callback_fn<typename std::remove_reference<Callable>::type>),
        callable(reinterpret_cast<intptr_t>(&callable)) {}
  Ret operator()(Params ...params) const {
    return callback(callable, std::forward<Params>(params)...);
  }
};

#else

template<typename Ret>
class function_ref<Ret()> {
  Ret (*callback)(intptr_t callable);
  intptr_t callable;

  template<typename Callable>
  static Ret callback_fn(intptr_t callable) {
    return (*reinterpret_cast<Callable*>(callable))();
  }

public:
  template<typename Callable>
  function_ref(Callable &&callable)
      : callback(callback_fn<typename std::remove_reference<Callable>::type>),
        callable(reinterpret_cast<intptr_t>(&callable)) {}
  Ret operator()() const { return callback(callable); }
};

template<typename Ret, typename Param1>
class function_ref<Ret(Param1)> {
  Ret (*callback)(intptr_t callable, Param1 param1);
  intptr_t callable;

  template<typename Callable>
  static Ret callback_fn(intptr_t callable, Param1 param1) {
    return (*reinterpret_cast<Callable*>(callable))(
        std::forward<Param1>(param1));
  }

public:
  template<typename Callable>
  function_ref(Callable &&callable)
      : callback(callback_fn<typename std::remove_reference<Callable>::type>),
        callable(reinterpret_cast<intptr_t>(&callable)) {}
  Ret operator()(Param1 param1) {
    return callback(callable, std::forward<Param1>(param1));
  }
};

template<typename Ret, typename Param1, typename Param2>
class function_ref<Ret(Param1, Param2)> {
  Ret (*callback)(intptr_t callable, Param1 param1, Param2 param2);
  intptr_t callable;

  template<typename Callable>
  static Ret callback_fn(intptr_t callable, Param1 param1, Param2 param2) {
    return (*reinterpret_cast<Callable*>(callable))(
        std::forward<Param1>(param1),
        std::forward<Param2>(param2));
  }

public:
  template<typename Callable>
  function_ref(Callable &&callable)
      : callback(callback_fn<typename std::remove_reference<Callable>::type>),
        callable(reinterpret_cast<intptr_t>(&callable)) {}
  Ret operator()(Param1 param1, Param2 param2) {
    return callback(callable,
                    std::forward<Param1>(param1),
                    std::forward<Param2>(param2));
  }
};

template<typename Ret, typename Param1, typename Param2, typename Param3>
class function_ref<Ret(Param1, Param2, Param3)> {
  Ret (*callback)(intptr_t callable, Param1 param1, Param2 param2, Param3 param3);
  intptr_t callable;

  template<typename Callable>
  static Ret callback_fn(intptr_t callable, Param1 param1, Param2 param2,
                         Param3 param3) {
    return (*reinterpret_cast<Callable*>(callable))(
        std::forward<Param1>(param1),
        std::forward<Param2>(param2),
        std::forward<Param3>(param3));
  }

public:
  template<typename Callable>
  function_ref(Callable &&callable)
      : callback(callback_fn<typename std::remove_reference<Callable>::type>),
        callable(reinterpret_cast<intptr_t>(&callable)) {}
  Ret operator()(Param1 param1, Param2 param2, Param3 param3) {
    return callback(callable,
                    std::forward<Param1>(param1),
                    std::forward<Param2>(param2),
                    std::forward<Param3>(param3));
  }
};

#endif

// deleter - Very very very simple method that is used to invoke operator
// delete on something.  It is used like this:
//
//   for_each(V.begin(), B.end(), deleter<Interval>);
//
template <class T>
inline void deleter(T *Ptr) {
  delete Ptr;
}



//===----------------------------------------------------------------------===//
//     Extra additions to <iterator>
//===----------------------------------------------------------------------===//

// mapped_iterator - This is a simple iterator adapter that causes a function to
// be dereferenced whenever operator* is invoked on the iterator.
//
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

  inline const RootIt &getCurrent() const { return current; }
  inline const UnaryFunc &getFunc() const { return Fn; }

  inline explicit mapped_iterator(const RootIt &I, UnaryFunc F)
    : current(I), Fn(F) {}

  inline value_type operator*() const {   // All this work to do this
    return Fn(*current);         // little change
  }

  _Self& operator++() { ++current; return *this; }
  _Self& operator--() { --current; return *this; }
  _Self  operator++(int) { _Self __tmp = *this; ++current; return __tmp; }
  _Self  operator--(int) { _Self __tmp = *this; --current; return __tmp; }
  _Self  operator+    (difference_type n) const {
    return _Self(current + n, Fn);
  }
  _Self& operator+=   (difference_type n) { current += n; return *this; }
  _Self  operator-    (difference_type n) const {
    return _Self(current - n, Fn);
  }
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
  return mapped_iterator<_Iterator, Func>(X.getCurrent() - N, X.getFunc());
}


// map_iterator - Provide a convenient way to create mapped_iterators, just like
// make_pair is useful for creating pairs...
//
template <class ItTy, class FuncTy>
inline mapped_iterator<ItTy, FuncTy> map_iterator(const ItTy &I, FuncTy F) {
  return mapped_iterator<ItTy, FuncTy>(I, F);
}

//===----------------------------------------------------------------------===//
//     Extra additions to <utility>
//===----------------------------------------------------------------------===//

/// \brief Function object to check whether the first component of a std::pair
/// compares less than the first component of another std::pair.
struct less_first {
  template <typename T> bool operator()(const T &lhs, const T &rhs) const {
    return lhs.first < rhs.first;
  }
};

/// \brief Function object to check whether the second component of a std::pair
/// compares less than the second component of another std::pair.
struct less_second {
  template <typename T> bool operator()(const T &lhs, const T &rhs) const {
    return lhs.second < rhs.second;
  }
};

//===----------------------------------------------------------------------===//
//     Extra additions for arrays
//===----------------------------------------------------------------------===//

/// Find the length of an array.
template <class T, std::size_t N>
LLVM_CONSTEXPR inline size_t array_lengthof(T (&)[N]) {
  return N;
}

/// Adapt std::less<T> for array_pod_sort.
template<typename T>
inline int array_pod_sort_comparator(const void *P1, const void *P2) {
  if (std::less<T>()(*reinterpret_cast<const T*>(P1),
                     *reinterpret_cast<const T*>(P2)))
    return -1;
  if (std::less<T>()(*reinterpret_cast<const T*>(P2),
                     *reinterpret_cast<const T*>(P1)))
    return 1;
  return 0;
}

/// get_array_pod_sort_comparator - This is an internal helper function used to
/// get type deduction of T right.
template<typename T>
inline int (*get_array_pod_sort_comparator(const T &))
             (const void*, const void*) {
  return array_pod_sort_comparator<T>;
}


/// array_pod_sort - This sorts an array with the specified start and end
/// extent.  This is just like std::sort, except that it calls qsort instead of
/// using an inlined template.  qsort is slightly slower than std::sort, but
/// most sorts are not performance critical in LLVM and std::sort has to be
/// template instantiated for each type, leading to significant measured code
/// bloat.  This function should generally be used instead of std::sort where
/// possible.
///
/// This function assumes that you have simple POD-like types that can be
/// compared with std::less and can be moved with memcpy.  If this isn't true,
/// you should use std::sort.
///
/// NOTE: If qsort_r were portable, we could allow a custom comparator and
/// default to std::less.
template<class IteratorTy>
inline void array_pod_sort(IteratorTy Start, IteratorTy End) {
  // Don't dereference start iterator of empty sequence.
  if (Start == End) return;
  qsort(&*Start, End-Start, sizeof(*Start),
        get_array_pod_sort_comparator(*Start));
}

template <class IteratorTy>
inline void array_pod_sort(
    IteratorTy Start, IteratorTy End,
    int (*Compare)(
        const typename std::iterator_traits<IteratorTy>::value_type *,
        const typename std::iterator_traits<IteratorTy>::value_type *)) {
  // Don't dereference start iterator of empty sequence.
  if (Start == End) return;
  qsort(&*Start, End - Start, sizeof(*Start),
        reinterpret_cast<int (*)(const void *, const void *)>(Compare));
}

//===----------------------------------------------------------------------===//
//     Extra additions to <algorithm>
//===----------------------------------------------------------------------===//

/// For a container of pointers, deletes the pointers and then clears the
/// container.
template<typename Container>
void DeleteContainerPointers(Container &C) {
  for (typename Container::iterator I = C.begin(), E = C.end(); I != E; ++I)
    delete *I;
  C.clear();
}

/// In a container of pairs (usually a map) whose second element is a pointer,
/// deletes the second elements and then clears the container.
template<typename Container>
void DeleteContainerSeconds(Container &C) {
  for (typename Container::iterator I = C.begin(), E = C.end(); I != E; ++I)
    delete I->second;
  C.clear();
}

//===----------------------------------------------------------------------===//
//     Extra additions to <memory>
//===----------------------------------------------------------------------===//

#if LLVM_HAS_VARIADIC_TEMPLATES

// Implement make_unique according to N3656.

/// \brief Constructs a `new T()` with the given args and returns a
///        `unique_ptr<T>` which owns the object.
///
/// Example:
///
///     auto p = make_unique<int>();
///     auto p = make_unique<std::tuple<int, int>>(0, 1);
template <class T, class... Args>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Args &&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

/// \brief Constructs a `new T[n]` with the given args and returns a
///        `unique_ptr<T[]>` which owns the object.
///
/// \param n size of the new array.
///
/// Example:
///
///     auto p = make_unique<int[]>(2); // value-initializes the array with 0's.
template <class T>
typename std::enable_if<std::is_array<T>::value && std::extent<T>::value == 0,
                        std::unique_ptr<T>>::type
make_unique(size_t n) {
  return std::unique_ptr<T>(new typename std::remove_extent<T>::type[n]());
}

/// This function isn't used and is only here to provide better compile errors.
template <class T, class... Args>
typename std::enable_if<std::extent<T>::value != 0>::type
make_unique(Args &&...) LLVM_DELETED_FUNCTION;

#else

template <class T>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique() {
  return std::unique_ptr<T>(new T());
}

template <class T, class Arg1>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Arg1 &&arg1) {
  return std::unique_ptr<T>(new T(std::forward<Arg1>(arg1)));
}

template <class T, class Arg1, class Arg2>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Arg1 &&arg1, Arg2 &&arg2) {
  return std::unique_ptr<T>(
      new T(std::forward<Arg1>(arg1), std::forward<Arg2>(arg2)));
}

template <class T, class Arg1, class Arg2, class Arg3>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Arg1 &&arg1, Arg2 &&arg2, Arg3 &&arg3) {
  return std::unique_ptr<T>(new T(std::forward<Arg1>(arg1),
                                  std::forward<Arg2>(arg2),
                                  std::forward<Arg3>(arg3)));
}

template <class T, class Arg1, class Arg2, class Arg3, class Arg4>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Arg1 &&arg1, Arg2 &&arg2, Arg3 &&arg3, Arg4 &&arg4) {
  return std::unique_ptr<T>(
      new T(std::forward<Arg1>(arg1), std::forward<Arg2>(arg2),
            std::forward<Arg3>(arg3), std::forward<Arg4>(arg4)));
}

template <class T, class Arg1, class Arg2, class Arg3, class Arg4, class Arg5>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Arg1 &&arg1, Arg2 &&arg2, Arg3 &&arg3, Arg4 &&arg4, Arg5 &&arg5) {
  return std::unique_ptr<T>(
      new T(std::forward<Arg1>(arg1), std::forward<Arg2>(arg2),
            std::forward<Arg3>(arg3), std::forward<Arg4>(arg4),
            std::forward<Arg5>(arg5)));
}

template <class T, class Arg1, class Arg2, class Arg3, class Arg4, class Arg5,
          class Arg6>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Arg1 &&arg1, Arg2 &&arg2, Arg3 &&arg3, Arg4 &&arg4, Arg5 &&arg5,
            Arg6 &&arg6) {
  return std::unique_ptr<T>(
      new T(std::forward<Arg1>(arg1), std::forward<Arg2>(arg2),
            std::forward<Arg3>(arg3), std::forward<Arg4>(arg4),
            std::forward<Arg5>(arg5), std::forward<Arg6>(arg6)));
}

template <class T, class Arg1, class Arg2, class Arg3, class Arg4, class Arg5,
          class Arg6, class Arg7>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Arg1 &&arg1, Arg2 &&arg2, Arg3 &&arg3, Arg4 &&arg4, Arg5 &&arg5,
            Arg6 &&arg6, Arg7 &&arg7) {
  return std::unique_ptr<T>(
      new T(std::forward<Arg1>(arg1), std::forward<Arg2>(arg2),
            std::forward<Arg3>(arg3), std::forward<Arg4>(arg4),
            std::forward<Arg5>(arg5), std::forward<Arg6>(arg6),
            std::forward<Arg7>(arg7)));
}

template <class T, class Arg1, class Arg2, class Arg3, class Arg4, class Arg5,
          class Arg6, class Arg7, class Arg8>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Arg1 &&arg1, Arg2 &&arg2, Arg3 &&arg3, Arg4 &&arg4, Arg5 &&arg5,
            Arg6 &&arg6, Arg7 &&arg7, Arg8 &&arg8) {
  return std::unique_ptr<T>(
      new T(std::forward<Arg1>(arg1), std::forward<Arg2>(arg2),
            std::forward<Arg3>(arg3), std::forward<Arg4>(arg4),
            std::forward<Arg5>(arg5), std::forward<Arg6>(arg6),
            std::forward<Arg7>(arg7), std::forward<Arg8>(arg8)));
}

template <class T, class Arg1, class Arg2, class Arg3, class Arg4, class Arg5,
          class Arg6, class Arg7, class Arg8, class Arg9>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Arg1 &&arg1, Arg2 &&arg2, Arg3 &&arg3, Arg4 &&arg4, Arg5 &&arg5,
            Arg6 &&arg6, Arg7 &&arg7, Arg8 &&arg8, Arg9 &&arg9) {
  return std::unique_ptr<T>(
      new T(std::forward<Arg1>(arg1), std::forward<Arg2>(arg2),
            std::forward<Arg3>(arg3), std::forward<Arg4>(arg4),
            std::forward<Arg5>(arg5), std::forward<Arg6>(arg6),
            std::forward<Arg7>(arg7), std::forward<Arg8>(arg8),
            std::forward<Arg9>(arg9)));
}

template <class T, class Arg1, class Arg2, class Arg3, class Arg4, class Arg5,
          class Arg6, class Arg7, class Arg8, class Arg9, class Arg10>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Arg1 &&arg1, Arg2 &&arg2, Arg3 &&arg3, Arg4 &&arg4, Arg5 &&arg5,
            Arg6 &&arg6, Arg7 &&arg7, Arg8 &&arg8, Arg9 &&arg9, Arg10 &&arg10) {
  return std::unique_ptr<T>(
      new T(std::forward<Arg1>(arg1), std::forward<Arg2>(arg2),
            std::forward<Arg3>(arg3), std::forward<Arg4>(arg4),
            std::forward<Arg5>(arg5), std::forward<Arg6>(arg6),
            std::forward<Arg7>(arg7), std::forward<Arg8>(arg8),
            std::forward<Arg9>(arg9), std::forward<Arg10>(arg10)));
}

template <class T>
typename std::enable_if<std::is_array<T>::value &&std::extent<T>::value == 0,
                        std::unique_ptr<T>>::type
make_unique(size_t n) {
  return std::unique_ptr<T>(new typename std::remove_extent<T>::type[n]());
}

#endif

struct FreeDeleter {
  void operator()(void* v) {
    ::free(v);
  }
};

template<typename First, typename Second>
struct pair_hash {
  size_t operator()(const std::pair<First, Second> &P) const {
    return std::hash<First>()(P.first) * 31 + std::hash<Second>()(P.second);
  }
};

} // End llvm namespace

#endif
