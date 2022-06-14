//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// Test nested types:

// template <InputIterator Iter>
// class move_iterator {
// public:
//  using iterator_type     = Iterator;
//  using iterator_concept  = input_iterator_tag; // From C++20
//  using iterator_category = see below; // not always present starting from C++20
//  using value_type        = iter_value_t<Iterator>; // Until C++20, iterator_traits<Iterator>::value_type
//  using difference_type   = iter_difference_t<Iterator>; // Until C++20, iterator_traits<Iterator>::difference_type;
//  using pointer           = Iterator;
//  using reference         = iter_rvalue_reference_t<Iterator>; // Until C++20, value_type&&
// };

#include <functional>
#include <iterator>
#include <type_traits>

#include "test_macros.h"
#include "test_iterators.h"

struct FooIter {
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type = void*;
  using difference_type = void*;
  using pointer = void*;
  using reference = char&;
  bool& operator*() const;
};

#if TEST_STD_VER > 17
template <>
struct std::indirectly_readable_traits<FooIter> {
  using value_type = int;
};
template <>
struct std::incrementable_traits<FooIter> {
  using difference_type = char;
};
#endif

#if TEST_STD_VER > 17
// Not using `FooIter::value_type`.
static_assert(std::is_same_v<typename std::move_iterator<FooIter>::value_type, int>);
// Not using `FooIter::difference_type`.
static_assert(std::is_same_v<typename std::move_iterator<FooIter>::difference_type, char>);
static_assert(std::is_same_v<typename std::reverse_iterator<FooIter>::reference, bool&>);

#else
static_assert(std::is_same<typename std::reverse_iterator<FooIter>::reference, char&>::value, "");
#endif

template <class ValueType, class Reference>
struct DummyIt {
  typedef std::forward_iterator_tag iterator_category;
  typedef ValueType value_type;
  typedef std::ptrdiff_t difference_type;
  typedef ValueType* pointer;
  typedef Reference reference;

  Reference operator*() const;
};

template <class It>
void test() {
  typedef std::move_iterator<It> R;
  typedef std::iterator_traits<It> T;
  static_assert((std::is_same<typename R::iterator_type, It>::value), "");
  static_assert((std::is_same<typename R::difference_type, typename T::difference_type>::value), "");
  static_assert((std::is_same<typename R::pointer, It>::value), "");
  static_assert((std::is_same<typename R::value_type, typename T::value_type>::value), "");

#if TEST_STD_VER > 17
  static_assert((std::is_same_v<typename R::reference, std::iter_rvalue_reference_t<It>>), "");
#else
  static_assert((std::is_same<typename R::reference, typename R::value_type&&>::value), "");
#endif

#if TEST_STD_VER > 17
  if constexpr (std::is_same_v<typename T::iterator_category, std::contiguous_iterator_tag>) {
    static_assert((std::is_same<typename R::iterator_category, std::random_access_iterator_tag>::value), "");
  } else {
    static_assert((std::is_same<typename R::iterator_category, typename T::iterator_category>::value), "");
  }
#else
  static_assert((std::is_same<typename R::iterator_category, typename T::iterator_category>::value), "");
#endif

#if TEST_STD_VER > 17
  static_assert(std::is_same_v<typename R::iterator_concept, std::input_iterator_tag>);
#endif
}

int main(int, char**) {
  test<cpp17_input_iterator<char*> >();
  test<forward_iterator<char*> >();
  test<bidirectional_iterator<char*> >();
  test<random_access_iterator<char*> >();
  test<char*>();

#if TEST_STD_VER >= 11
  {
      typedef DummyIt<int, int> T;
      typedef std::move_iterator<T> It;
      static_assert(std::is_same<It::reference, int>::value, "");
  }
  {
      typedef DummyIt<int, std::reference_wrapper<int> > T;
      typedef std::move_iterator<T> It;
      static_assert(std::is_same<It::reference, std::reference_wrapper<int> >::value, "");
  }
  {
      // Check that move_iterator uses whatever reference type it's given
      // when it's not a reference.
      typedef DummyIt<int, long > T;
      typedef std::move_iterator<T> It;
      static_assert(std::is_same<It::reference, long>::value, "");
  }
  {
      typedef DummyIt<int, int&> T;
      typedef std::move_iterator<T> It;
      static_assert(std::is_same<It::reference, int&&>::value, "");
  }
  {
      typedef DummyIt<int, int&&> T;
      typedef std::move_iterator<T> It;
      static_assert(std::is_same<It::reference, int&&>::value, "");
  }
#endif

#if TEST_STD_VER > 17
  test<contiguous_iterator<char*>>();
  static_assert(std::is_same_v<typename std::move_iterator<forward_iterator<char*>>::iterator_concept, std::input_iterator_tag>);
  static_assert(std::is_same_v<typename std::move_iterator<bidirectional_iterator<char*>>::iterator_concept, std::input_iterator_tag>);
  static_assert(std::is_same_v<typename std::move_iterator<random_access_iterator<char*>>::iterator_concept, std::input_iterator_tag>);
  static_assert(std::is_same_v<typename std::move_iterator<contiguous_iterator<char*>>::iterator_concept, std::input_iterator_tag>);
  static_assert(std::is_same_v<typename std::move_iterator<char*>::iterator_concept, std::input_iterator_tag>);
#endif

  return 0;
}
