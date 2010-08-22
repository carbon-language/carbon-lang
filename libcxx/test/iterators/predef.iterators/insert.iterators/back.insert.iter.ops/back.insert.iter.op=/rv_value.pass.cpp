//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// back_insert_iterator

// requires CopyConstructible<Cont::value_type>
//   back_insert_iterator<Cont>&
//   operator=(Cont::value_type&& value);

#include <iterator>

#ifdef _LIBCPP_MOVE

#include <vector>
#include <memory>
#include <cassert>

template <class C>
void
test(C c)
{
    std::back_insert_iterator<C> i(c);
    i = typename C::value_type();
    assert(c.back() == typename C::value_type());
}

#endif  // _LIBCPP_MOVE

int main()
{
#ifdef _LIBCPP_MOVE
    test(std::vector<std::unique_ptr<int> >());
#endif
}
