//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// REQUIRES-ANY: c++98, c++03

// <utility>

// template <class T1, class T2> struct pair

// pair& operator=(pair const& p);

#include <utility>
#include <memory>
#include <cassert>


struct NonAssignable {
  NonAssignable() {}
private:
  NonAssignable& operator=(NonAssignable const&);
};

int main()
{
    // Test that we don't constrain the assignment operator in C++03 mode.
    // Since we don't have access control SFINAE having pair evaluate SFINAE
    // may cause a hard error.
    typedef std::pair<int, NonAssignable> P;
    static_assert(std::is_copy_assignable<P>::value, "");
}
