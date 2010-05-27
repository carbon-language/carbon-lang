//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <exception>

// class nested_exception;

// template <class E> void rethrow_if_nested(const E& e);

#include <exception>
#include <cstdlib>
#include <cassert>

class A
{
    int data_;
public:
    explicit A(int data) : data_(data) {}

    friend bool operator==(const A& x, const A& y) {return x.data_ == y.data_;}
};

class B
    : public std::nested_exception
{
    int data_;
public:
    explicit B(int data) : data_(data) {}
    B(const B& b) : data_(b.data_) {}

    friend bool operator==(const B& x, const B& y) {return x.data_ == y.data_;}
};

int main()
{
    {
        try
        {
            A a(3);
            std::rethrow_if_nested(a);
            assert(true);
        }
        catch (...)
        {
            assert(false);
        }
    }
    {
        try
        {
            throw B(5);
        }
        catch (const B& b0)
        {
            try
            {
                B b = b0;
                std::rethrow_if_nested(b);
                assert(false);
            }
            catch (const B& b)
            {
                assert(b == B(5));
            }
        }
    }
}
