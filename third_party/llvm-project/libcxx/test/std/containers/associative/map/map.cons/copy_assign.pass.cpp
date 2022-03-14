//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class map

// map& operator=(const map& m);

#include <map>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iterator>
#include <vector>

#include "test_macros.h"
#include "../../../test_compare.h"
#include "test_allocator.h"
#include "min_allocator.h"

#if TEST_STD_VER >= 11
std::vector<int> ca_allocs;
std::vector<int> ca_deallocs;

template <class T>
class counting_allocatorT {
public:
    typedef T value_type;
    int foo{0};
    counting_allocatorT(int f) noexcept : foo(f) {}

    using propagate_on_container_copy_assignment = std::true_type;
    template <class U> counting_allocatorT(const counting_allocatorT<U>& other) noexcept {foo = other.foo;}
    template <class U> bool operator==(const counting_allocatorT<U>& other) const noexcept { return foo == other.foo; }
    template <class U> bool operator!=(const counting_allocatorT<U>& other) const noexcept { return foo != other.foo; }

    T* allocate(size_t n) const {
        ca_allocs.push_back(foo);
        void * const pv = ::malloc(n * sizeof(T));
        return static_cast<T *>(pv);
    }
    void deallocate(T* p, size_t) const noexcept {
        ca_deallocs.push_back(foo);
        free(p);
    }
};

template <class T>
class counting_allocatorF {
public:
    typedef T value_type;
    int foo{0};
    counting_allocatorF(int f) noexcept : foo(f) {}

    using propagate_on_container_copy_assignment = std::false_type;
    template <class U> counting_allocatorF(const counting_allocatorF<U>& other) noexcept {foo = other.foo;}
    template <class U> bool operator==(const counting_allocatorF<U>& other) const noexcept { return foo == other.foo; }
    template <class U> bool operator!=(const counting_allocatorF<U>& other) const noexcept { return foo != other.foo; }

    T* allocate(size_t n) const {
        ca_allocs.push_back(foo);
        void * const pv = ::malloc(n * sizeof(T));
        return static_cast<T *>(pv);
    }
    void deallocate(T* p, size_t) const noexcept {
        ca_deallocs.push_back(foo);
        free(p);
    }
};

bool balanced_allocs() {
    std::vector<int> temp1, temp2;

    std::printf("Allocations = %zu, deallocations = %zu\n", ca_allocs.size(),
                ca_deallocs.size());
    if (ca_allocs.size() != ca_deallocs.size())
        return false;

    temp1 = ca_allocs;
    std::sort(temp1.begin(), temp1.end());
    temp2.clear();
    std::unique_copy(temp1.begin(), temp1.end(), std::back_inserter<std::vector<int>>(temp2));
    std::printf("There were %zu different allocators\n", temp2.size());

    for (std::vector<int>::const_iterator it = temp2.begin(); it != temp2.end(); ++it ) {
        std::ptrdiff_t const allocs = std::count(ca_allocs.begin(), ca_allocs.end(), *it);
        std::ptrdiff_t const deallocs = std::count(ca_deallocs.begin(), ca_deallocs.end(), *it);
        std::printf("%d: %td vs %td\n", *it, allocs, deallocs);
        if (allocs != deallocs)
            return false;
    }

    temp1 = ca_allocs;
    std::sort(temp1.begin(), temp1.end());
    temp2.clear();
    std::unique_copy(temp1.begin(), temp1.end(), std::back_inserter<std::vector<int>>(temp2));
    std::printf("There were %zu different (de)allocators\n", temp2.size());

    for (std::vector<int>::const_iterator it = ca_deallocs.begin(); it != ca_deallocs.end(); ++it ) {
        std::ptrdiff_t const allocs = std::count(ca_allocs.begin(), ca_allocs.end(), *it);
        std::ptrdiff_t const deallocs = std::count(ca_deallocs.begin(), ca_deallocs.end(), *it);
        std::printf("%d: %td vs %td\n", *it, allocs, deallocs);
        if (allocs != deallocs)
            return false;
    }

    return true;
}
#endif

int main(int, char**)
{
    {
        typedef std::pair<const int, double> V;
        V ar[] =
        {
            V(1, 1),
            V(1, 1.5),
            V(1, 2),
            V(2, 1),
            V(2, 1.5),
            V(2, 2),
            V(3, 1),
            V(3, 1.5),
            V(3, 2)
        };
        typedef test_less<int> C;
        typedef test_allocator<V> A;
        std::map<int, double, C, A> mo(ar, ar+sizeof(ar)/sizeof(ar[0]), C(5), A(2));
        std::map<int, double, C, A> m(ar, ar+sizeof(ar)/sizeof(ar[0])/2, C(3), A(7));
        m = mo;
        assert(m.get_allocator() == A(7));
        assert(m.key_comp() == C(5));
        assert(m.size() == 3);
        assert(std::distance(m.begin(), m.end()) == 3);
        assert(*m.begin() == V(1, 1));
        assert(*std::next(m.begin()) == V(2, 1));
        assert(*std::next(m.begin(), 2) == V(3, 1));

        assert(mo.get_allocator() == A(2));
        assert(mo.key_comp() == C(5));
        assert(mo.size() == 3);
        assert(std::distance(mo.begin(), mo.end()) == 3);
        assert(*mo.begin() == V(1, 1));
        assert(*std::next(mo.begin()) == V(2, 1));
        assert(*std::next(mo.begin(), 2) == V(3, 1));
    }
    {
        typedef std::pair<const int, double> V;
        const V ar[] =
        {
            V(1, 1),
            V(2, 1),
            V(3, 1),
        };
        std::map<int, double> m(ar, ar+sizeof(ar)/sizeof(ar[0]));
        std::map<int, double> *p = &m;
        m = *p;

        assert(m.size() == 3);
        assert(std::equal(m.begin(), m.end(), ar));
    }
    {
        typedef std::pair<const int, double> V;
        V ar[] =
        {
            V(1, 1),
            V(1, 1.5),
            V(1, 2),
            V(2, 1),
            V(2, 1.5),
            V(2, 2),
            V(3, 1),
            V(3, 1.5),
            V(3, 2)
        };
        typedef test_less<int> C;
        typedef other_allocator<V> A;
        std::map<int, double, C, A> mo(ar, ar+sizeof(ar)/sizeof(ar[0]), C(5), A(2));
        std::map<int, double, C, A> m(ar, ar+sizeof(ar)/sizeof(ar[0])/2, C(3), A(7));
        m = mo;
        assert(m.get_allocator() == A(2));
        assert(m.key_comp() == C(5));
        assert(m.size() == 3);
        assert(std::distance(m.begin(), m.end()) == 3);
        assert(*m.begin() == V(1, 1));
        assert(*std::next(m.begin()) == V(2, 1));
        assert(*std::next(m.begin(), 2) == V(3, 1));

        assert(mo.get_allocator() == A(2));
        assert(mo.key_comp() == C(5));
        assert(mo.size() == 3);
        assert(std::distance(mo.begin(), mo.end()) == 3);
        assert(*mo.begin() == V(1, 1));
        assert(*std::next(mo.begin()) == V(2, 1));
        assert(*std::next(mo.begin(), 2) == V(3, 1));
    }
#if TEST_STD_VER >= 11
    {
        typedef std::pair<const int, double> V;
        V ar[] =
        {
            V(1, 1),
            V(1, 1.5),
            V(1, 2),
            V(2, 1),
            V(2, 1.5),
            V(2, 2),
            V(3, 1),
            V(3, 1.5),
            V(3, 2)
        };
        typedef test_less<int> C;
        typedef min_allocator<V> A;
        std::map<int, double, C, A> mo(ar, ar+sizeof(ar)/sizeof(ar[0]), C(5), A());
        std::map<int, double, C, A> m(ar, ar+sizeof(ar)/sizeof(ar[0])/2, C(3), A());
        m = mo;
        assert(m.get_allocator() == A());
        assert(m.key_comp() == C(5));
        assert(m.size() == 3);
        assert(std::distance(m.begin(), m.end()) == 3);
        assert(*m.begin() == V(1, 1));
        assert(*std::next(m.begin()) == V(2, 1));
        assert(*std::next(m.begin(), 2) == V(3, 1));

        assert(mo.get_allocator() == A());
        assert(mo.key_comp() == C(5));
        assert(mo.size() == 3);
        assert(std::distance(mo.begin(), mo.end()) == 3);
        assert(*mo.begin() == V(1, 1));
        assert(*std::next(mo.begin()) == V(2, 1));
        assert(*std::next(mo.begin(), 2) == V(3, 1));
    }
    {
        typedef std::pair<const int, double> V;
        V ar[] =
        {
            V(1, 1),
            V(1, 1.5),
            V(1, 2),
            V(2, 1),
            V(2, 1.5),
            V(2, 2),
            V(3, 1),
            V(3, 1.5),
            V(3, 2)
        };
        typedef test_less<int> C;
        typedef min_allocator<V> A;
        std::map<int, double, C, A> mo(ar, ar+sizeof(ar)/sizeof(ar[0]), C(5), A());
        std::map<int, double, C, A> m(ar, ar+sizeof(ar)/sizeof(ar[0])/2, C(3), A());
        m = mo;
        assert(m.get_allocator() == A());
        assert(m.key_comp() == C(5));
        assert(m.size() == 3);
        assert(std::distance(m.begin(), m.end()) == 3);
        assert(*m.begin() == V(1, 1));
        assert(*std::next(m.begin()) == V(2, 1));
        assert(*std::next(m.begin(), 2) == V(3, 1));

        assert(mo.get_allocator() == A());
        assert(mo.key_comp() == C(5));
        assert(mo.size() == 3);
        assert(std::distance(mo.begin(), mo.end()) == 3);
        assert(*mo.begin() == V(1, 1));
        assert(*std::next(mo.begin()) == V(2, 1));
        assert(*std::next(mo.begin(), 2) == V(3, 1));
    }

    assert(balanced_allocs());
    {
        typedef std::pair<const int, double> V;
        V ar[] =
        {
            V(1, 1),
            V(1, 1.5),
            V(1, 2),
            V(2, 1),
            V(2, 1.5),
            V(2, 2),
            V(3, 1),
            V(3, 1.5),
            V(3, 2)
        };
        typedef test_less<int> C;
        typedef counting_allocatorT<V> A;
        std::map<int, double, C, A> mo(ar, ar+sizeof(ar)/sizeof(ar[0]), C(5), A(1));
        std::map<int, double, C, A> m(ar, ar+sizeof(ar)/sizeof(ar[0])/2, C(3), A(2));
        m = mo;
        assert(m.key_comp() == C(5));
        assert(m.size() == 3);
        assert(std::distance(m.begin(), m.end()) == 3);
        assert(*m.begin() == V(1, 1));
        assert(*std::next(m.begin()) == V(2, 1));
        assert(*std::next(m.begin(), 2) == V(3, 1));

        assert(mo.key_comp() == C(5));
        assert(mo.size() == 3);
        assert(std::distance(mo.begin(), mo.end()) == 3);
        assert(*mo.begin() == V(1, 1));
        assert(*std::next(mo.begin()) == V(2, 1));
        assert(*std::next(mo.begin(), 2) == V(3, 1));
    }
    assert(balanced_allocs());
    {
        typedef std::pair<const int, double> V;
        V ar[] =
        {
            V(1, 1),
            V(1, 1.5),
            V(1, 2),
            V(2, 1),
            V(2, 1.5),
            V(2, 2),
            V(3, 1),
            V(3, 1.5),
            V(3, 2)
        };
        typedef test_less<int> C;
        typedef counting_allocatorF<V> A;
        std::map<int, double, C, A> mo(ar, ar+sizeof(ar)/sizeof(ar[0]), C(5), A(100));
        std::map<int, double, C, A> m(ar, ar+sizeof(ar)/sizeof(ar[0])/2, C(3), A(200));
        m = mo;
        assert(m.key_comp() == C(5));
        assert(m.size() == 3);
        assert(std::distance(m.begin(), m.end()) == 3);
        assert(*m.begin() == V(1, 1));
        assert(*std::next(m.begin()) == V(2, 1));
        assert(*std::next(m.begin(), 2) == V(3, 1));

        assert(mo.key_comp() == C(5));
        assert(mo.size() == 3);
        assert(std::distance(mo.begin(), mo.end()) == 3);
        assert(*mo.begin() == V(1, 1));
        assert(*std::next(mo.begin()) == V(2, 1));
        assert(*std::next(mo.begin(), 2) == V(3, 1));
    }
    assert(balanced_allocs());
#endif

  return 0;
}
