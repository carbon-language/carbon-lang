//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class BidirectionalIterator,
//           class charT = typename iterator_traits< BidirectionalIterator>::value_type,
//           class traits = regex_traits<charT>>
// class regex_iterator
// {
// public:
//     typedef basic_regex<charT, traits>           regex_type;
//     typedef match_results<BidirectionalIterator> value_type;
//     typedef ptrdiff_t                            difference_type;
//     typedef const value_type*                    pointer;
//     typedef const value_type&                    reference;
//     typedef forward_iterator_tag                 iterator_category;

#include <regex>
#include <type_traits>
#include "test_macros.h"

template <class CharT>
void
test()
{
    typedef std::regex_iterator<const CharT*> I;
    static_assert((std::is_same<typename I::regex_type, std::basic_regex<CharT> >::value), "");
    static_assert((std::is_same<typename I::value_type, std::match_results<const CharT*> >::value), "");
    static_assert((std::is_same<typename I::difference_type, std::ptrdiff_t>::value), "");
    static_assert((std::is_same<typename I::pointer, const std::match_results<const CharT*>*>::value), "");
    static_assert((std::is_same<typename I::reference, const std::match_results<const CharT*>&>::value), "");
    static_assert((std::is_same<typename I::iterator_category, std::forward_iterator_tag>::value), "");
}

int main(int, char**)
{
    test<char>();
    test<wchar_t>();

  return 0;
}
