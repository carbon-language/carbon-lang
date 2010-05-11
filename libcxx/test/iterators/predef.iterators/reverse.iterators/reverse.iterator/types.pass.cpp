//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// Test nested types and data member:

// template <BidirectionalIterator Iter> 
// class reverse_iterator { 
// protected: 
//   Iter current; 
// public: 
//   typedef Iter iterator_type; 
//   typedef Iter::value_type value_type; 
//   typedef Iter::difference_type difference_type; 
//   typedef Iter::reference reference; 
//   typedef Iter::pointer pointer; 
// };

#include <iterator>
#include <type_traits>

#include "../../../iterators.h"

template <class It>
struct find_current
    : private std::reverse_iterator<It>
{
    void test() {++(this->current);}
};

template <class It>
void
test()
{
    typedef std::reverse_iterator<It> R;
    typedef std::iterator_traits<It> T;
    find_current<It> q;
    q.test();
    static_assert((std::is_same<typename R::iterator_type, It>::value), "");
    static_assert((std::is_same<typename R::value_type, typename T::value_type>::value), "");
    static_assert((std::is_same<typename R::difference_type, typename T::difference_type>::value), "");
    static_assert((std::is_same<typename R::reference, typename T::reference>::value), "");
    static_assert((std::is_same<typename R::pointer, It>::value), "");
    static_assert((std::is_same<typename R::iterator_category, typename T::iterator_category>::value), "");
}

int main()
{
    test<bidirectional_iterator<char*> >();
    test<random_access_iterator<char*> >();
    test<char*>();
}
