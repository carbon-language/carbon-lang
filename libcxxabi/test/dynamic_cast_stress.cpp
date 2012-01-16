//===------------------------- dynamic_cast_stress.cpp --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <tuple>
#include <chrono>
#include <iostream>

template <std::size_t Indx, std::size_t Depth>
struct C
    : public virtual C<Indx, Depth-1>,
      public virtual C<Indx+1, Depth-1>
{
    virtual ~C() {}
};

template <std::size_t Indx>
struct C<Indx, 0>
{
    virtual ~C() {}
};

template <std::size_t Indx, std::size_t Depth>
struct B
    : public virtual C<Indx, Depth-1>,
      public virtual C<Indx+1, Depth-1>
{
};

template <class Indx, std::size_t Depth>
struct makeB;

template <std::size_t ...Indx, std::size_t Depth>
struct makeB<std::__tuple_indices<Indx...>, Depth>
    : public B<Indx, Depth>...
{
};

template <std::size_t Width, std::size_t Depth>
struct A
    : public makeB<typename std::__make_tuple_indices<Width>::type, Depth>
{
};

void test()
{
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration<double, std::micro> US;
    const std::size_t Width = 20;
    const std::size_t Depth = 7;
    A<Width, Depth> a;
    typedef B<Width/2, Depth> Destination;
//    typedef A<Width, Depth> Destination;
    auto t0 = Clock::now();
    Destination* b = dynamic_cast<Destination*>((C<Width/2, 0>*)&a);
    auto t1 = Clock::now();
    std::cout << US(t1-t0).count() << " microseconds\n";
    assert(b != 0);
}

int main()
{
    test();
}

/*
Timing results I'm seeing (median of 3 microseconds):

                          libc++abi    gcc's dynamic_cast
B<Width/2, Depth> -O3      50.694         93.190           libc++abi 84% faster
B<Width/2, Depth> -Os      55.235         94.103           libc++abi 70% faster
A<Width, Depth>   -O3      14.895         33.134           libc++abi 122% faster
A<Width, Depth>   -Os      16.515         31.553           libc++abi 91% faster

*/
