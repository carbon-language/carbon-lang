//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// Test unique_ptr default ctor

// default unique_ptr ctor should require default Deleter ctor


#include <memory>

class Deleter
{
    // expected-error@memory:* {{base class 'Deleter' has private default constructor}}
    // expected-note@memory:* + {{in instantiation of member function}}
    Deleter() {} // expected-note {{implicitly declared private here}}

public:

    Deleter(Deleter&) {}
    Deleter& operator=(Deleter&) { return *this; }

    void operator()(void*) const {}
};

int main()
{
    std::unique_ptr<int[], Deleter> p;
}
