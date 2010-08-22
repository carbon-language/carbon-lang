//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// front_insert_iterator

// front_insert_iterator<Cont>&
//   operator=(Cont::value_type&& value);

#include <iterator>

#ifdef _LIBCPP_MOVE

#include <list>
#include <memory>
#include <cassert>

template <class C>
void
test(C c)
{
    std::front_insert_iterator<C> i(c);
    i = typename C::value_type();
    assert(c.front() == typename C::value_type());
}

#endif  // _LIBCPP_MOVE

int main()
{
#ifdef _LIBCPP_MOVE
    test(std::list<std::unique_ptr<int> >());
#endif
}
