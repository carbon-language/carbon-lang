//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// Test unique_ptr::pointer type

#include <memory>
#include <type_traits>

struct Deleter
{
    struct pointer {};
};

int main()
{
    {
    typedef std::unique_ptr<int> P;
    static_assert((std::is_same<P::pointer, int*>::value), "");
    }
    {
    typedef std::unique_ptr<int, Deleter> P;
    static_assert((std::is_same<P::pointer, Deleter::pointer>::value), "");
    }
}
