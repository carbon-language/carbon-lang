//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// reference operator*() const;

#include <iterator>
#include <cassert>
#ifdef _LIBCPP_MOVE
#include <memory>
#endif

class A
{
    int data_;
public:
    A() : data_(1) {}
    ~A() {data_ = -1;}

    friend bool operator==(const A& x, const A& y)
        {return x.data_ == y.data_;}
};

template <class It>
void
test(It i, typename std::iterator_traits<It>::value_type x)
{
    std::move_iterator<It> r(i);
    assert(*r == x);
    typename std::iterator_traits<It>::value_type x2 = *r;
    assert(x2 == x);
}

#ifdef _LIBCPP_MOVE

struct do_nothing
{
    void operator()(void*) const {}
};

#endif

int main()
{
    A a;
    test(&a, A());
#ifdef _LIBCPP_MOVE
    int i;
    std::unique_ptr<int, do_nothing> p(&i);
    test(&p, std::unique_ptr<int, do_nothing>(&i));
#endif
}
