//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <memory>

// template <class OuterAlloc, class... InnerAllocs>
//   class scoped_allocator_adaptor

// template <class T, class... Args> void construct(T* p, Args&&... args);

#include <scoped_allocator>
#include <cassert>
#include <string>

#include "test_macros.h"
#include "allocators.h"

struct B
{
    static bool constructed;

    typedef A1<B> allocator_type;

    explicit B(std::allocator_arg_t, const allocator_type& a, int i)
    {
        assert(a.id() == 5);
        assert(i == 6);
        constructed = true;
    }
};

bool B::constructed = false;

struct C
{
    static bool constructed;

    typedef std::scoped_allocator_adaptor<A2<C>> allocator_type;

    explicit C(std::allocator_arg_t, const allocator_type& a, int i)
    {
        assert(a.id() == 7);
        assert(i == 8);
        constructed = true;
    }
};

bool C::constructed = false;

struct D
{
    static bool constructed;

    typedef std::scoped_allocator_adaptor<A2<D>> allocator_type;

    explicit D(int i, int j, const allocator_type& a)
    {
        assert(i == 1);
        assert(j == 2);
        assert(a.id() == 3);
        constructed = true;
    }
};

bool D::constructed = false;

struct E
{
    static bool constructed;

    typedef std::scoped_allocator_adaptor<A1<E>> allocator_type;

    explicit E(int i, int j, const allocator_type& a)
    {
        assert(i == 1);
        assert(j == 2);
        assert(a.id() == 50);
        constructed = true;
    }
};

bool E::constructed = false;

struct F
{
    static bool constructed;

    typedef std::scoped_allocator_adaptor<A2<F>> allocator_type;

    explicit F(int i, int j)
    {
        assert(i == 1);
        assert(j == 2);
    }

    explicit F(int i, int j, const allocator_type& a)
    {
        assert(i == 1);
        assert(j == 2);
        assert(a.id() == 50);
        constructed = true;
    }
};

bool F::constructed = false;

struct G
{
    static bool constructed;

    typedef std::allocator<G> allocator_type;

    G(std::allocator_arg_t, allocator_type&&) { assert(false); }
    G(allocator_type&) { constructed = true; }
};

bool G::constructed = false;

int main(int, char**)
{

    {
        typedef std::scoped_allocator_adaptor<A1<std::string>> A;
        A a;
        char buf[100];
        typedef std::string S;
        S* s = (S*)buf;
        a.construct(s, 4, 'c');
        assert(*s == "cccc");
        s->~S();
    }

    {
        typedef std::scoped_allocator_adaptor<A1<B>> A;
        A a(A1<B>(5));
        char buf[100];
        typedef B S;
        S* s = (S*)buf;
        a.construct(s, 6);
        assert(S::constructed);
        s->~S();
    }

    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<C>> A;
        A a(A1<int>(5), A2<C>(7));
        char buf[100];
        typedef C S;
        S* s = (S*)buf;
        a.construct(s, 8);
        assert(S::constructed);
        s->~S();
    }

    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<D>> A;
        A a(A1<int>(5), A2<D>(3));
        char buf[100];
        typedef D S;
        S* s = (S*)buf;
        a.construct(s, 1, 2);
        assert(S::constructed);
        s->~S();
    }

    {
        typedef std::scoped_allocator_adaptor<A3<E>, A2<E>> K;
        typedef std::scoped_allocator_adaptor<K, A1<E>> A;
        A a(K(), A1<E>(50));
        char buf[100];
        typedef E S;
        S* s = (S*)buf;
        A3<E>::constructed = false;
        a.construct(s, 1, 2);
        assert(S::constructed);
        assert(A3<E>::constructed);
        s->~S();
    }

    {
        typedef std::scoped_allocator_adaptor<A3<F>, A2<F>> K;
        typedef std::scoped_allocator_adaptor<K, A1<F>> A;
        A a(K(), A1<F>(50));
        char buf[100];
        typedef F S;
        S* s = (S*)buf;
        A3<F>::constructed = false;
        a.construct(s, 1, 2);
        assert(!S::constructed);
        assert(A3<F>::constructed);
        s->~S();
    }

    // LWG 2586
    // Test that is_constructible uses an lvalue ref so the correct constructor
    // is picked.
    {
        std::scoped_allocator_adaptor<G::allocator_type> sa;
        G* ptr = sa.allocate(1);
        sa.construct(ptr);
        assert(G::constructed);
        sa.deallocate(ptr, 1);
    }

  return 0;
}
