//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// XFAIL: c++98, c++03

// <memory>

// unique_ptr

// Test unique_ptr(pointer, deleter) ctor

// unique_ptr<T, const D&>(pointer, D()) should not compile

#include <memory>

class Deleter
{
public:
    Deleter() {}
    void operator()(int* p) const {delete [] p;}
};

int main()
{
    int* p = nullptr;
    std::unique_ptr<int[], const Deleter&> s(p, Deleter()); // expected-error@memory:* {{static_assert failed "rvalue deleter bound to reference"}}
}
