//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// template<class NotAnIterator>
// struct iterator_traits
// { 
// };

#include <iterator>

struct not_an_iterator
{
};

int main()
{
    typedef std::iterator_traits<not_an_iterator> It;
}
