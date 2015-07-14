//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++98, c++03

// <atomic>

// constexpr atomic<T>::atomic(T value)

#include <atomic>
#include <type_traits>
#include <cassert>

struct UserType {
    int i;

    UserType() noexcept {}
    constexpr explicit UserType(int d) noexcept : i(d) {}

    friend bool operator==(const UserType& x, const UserType& y) {
        return x.i == y.i;
    }
};

template <class Tp>
void test() {
    typedef std::atomic<Tp> Atomic;
    static_assert(std::is_literal_type<Atomic>::value, "");
    constexpr Tp t(42);
    {
        constexpr Atomic a(t);
        assert(a == t);
    }
    {
        constexpr Atomic a{t};
        assert(a == t);
    }
    {
        constexpr Atomic a = ATOMIC_VAR_INIT(t);
        assert(a == t);
    }
}


int main()
{
    test<int>();
    test<UserType>();
}
