//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// template<class Y> explicit shared_ptr(auto_ptr<Y>&& r);

#include <memory>
#include <new>
#include <cstdlib>
#include <cassert>

bool throw_next = false;

void* operator new(std::size_t s) throw(std::bad_alloc)
{
    if (throw_next)
        throw std::bad_alloc();
    return std::malloc(s);
}

void  operator delete(void* p) throw()
{
    std::free(p);
}

struct B
{
    static int count;

    B() {++count;}
    B(const B&) {++count;}
    virtual ~B() {--count;}
};

int B::count = 0;

struct A
    : public B
{
    static int count;

    A() {++count;}
    A(const A&) {++count;}
    ~A() {--count;}
};

int A::count = 0;

int main()
{
    {
    std::auto_ptr<A> ptr(new A);
    A* raw_ptr = ptr.get();
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    std::shared_ptr<B> p(std::move(ptr));
#else
    std::shared_ptr<B> p(ptr);
#endif
    assert(A::count == 1);
    assert(B::count == 1);
    assert(p.use_count() == 1);
    assert(p.get() == raw_ptr);
    assert(ptr.get() == 0);
    }
    assert(A::count == 0);
    {
    std::auto_ptr<A> ptr(new A);
    A* raw_ptr = ptr.get();
    throw_next = true;
    try
    {
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
        std::shared_ptr<B> p(std::move(ptr));
#else
        std::shared_ptr<B> p(ptr);
#endif
        assert(false);
    }
    catch (...)
    {
        assert(A::count == 1);
        assert(B::count == 1);
        assert(ptr.get() == raw_ptr);
    }
    }
    assert(A::count == 0);
}
