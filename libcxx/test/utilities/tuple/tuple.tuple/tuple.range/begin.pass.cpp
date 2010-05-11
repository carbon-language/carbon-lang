//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class InputIterator>
//   InputIterator begin(const tuple<InputIterator, InputIterator>& t);

// template <class InputIterator>
//   InputIterator end(const tuple<InputIterator, InputIterator>& t);

#include <tuple>
#include <iterator>
#include <cassert>

int main()
{
    {
        typedef std::tuple<int*, int*> T;
        int array[5] = {0, 1, 2, 3, 4};
        const T t(std::begin(array), std::end(array));
        assert(begin(t) == std::begin(array));
        assert(end(t) == std::end(array));
    }
}
