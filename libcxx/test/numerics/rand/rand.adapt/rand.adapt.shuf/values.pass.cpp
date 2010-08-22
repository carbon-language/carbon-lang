//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <random>

// template<class Engine, size_t k>
// class shuffle_order_engine
// {
// public:
//     // types
//     typedef typename Engine::result_type result_type;
//
//     // engine characteristics
//     static constexpr size_t table_size = k;
//     static constexpr result_type min() { return Engine::min; }
//     static constexpr result_type max() { return Engine::max; }

#include <random>
#include <type_traits>
#include <cassert>

void
test1()
{
    typedef std::knuth_b E;
    static_assert(E::table_size == 256, "");
    /*static_*/assert((E::min() == 1)/*, ""*/);
    /*static_*/assert((E::max() == 2147483646)/*, ""*/);
}

int main()
{
    test1();
}
