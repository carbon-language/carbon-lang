//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SORTABLE_HELPERS_H
#define SORTABLE_HELPERS_H

#include <type_traits>

#include "test_macros.h"

struct TrivialSortable {
    int value;
    TEST_CONSTEXPR TrivialSortable() : value(0) {}
    TEST_CONSTEXPR TrivialSortable(int v) : value(v) {}
    friend TEST_CONSTEXPR bool operator<(const TrivialSortable& a, const TrivialSortable& b) {
        return a.value / 10 < b.value / 10;
    }
    static TEST_CONSTEXPR bool less(const TrivialSortable& a, const TrivialSortable& b) {
        return a.value < b.value;
    }
};

struct NonTrivialSortable {
    int value;
    TEST_CONSTEXPR NonTrivialSortable() : value(0) {}
    TEST_CONSTEXPR NonTrivialSortable(int v) : value(v) {}
    TEST_CONSTEXPR NonTrivialSortable(const NonTrivialSortable& rhs) : value(rhs.value) {}
    TEST_CONSTEXPR_CXX14 NonTrivialSortable& operator=(const NonTrivialSortable& rhs) { value = rhs.value; return *this; }
    friend TEST_CONSTEXPR bool operator<(const NonTrivialSortable& a, const NonTrivialSortable& b) {
        return a.value / 10 < b.value / 10;
    }
    static TEST_CONSTEXPR bool less(const NonTrivialSortable& a, const NonTrivialSortable& b) {
        return a.value < b.value;
    }
};


struct TrivialSortableWithComp {
    int value;
    TEST_CONSTEXPR TrivialSortableWithComp() : value(0) {}
    TEST_CONSTEXPR TrivialSortableWithComp(int v) : value(v) {}
    struct Comparator {
        TEST_CONSTEXPR bool operator()(const TrivialSortableWithComp& a, const TrivialSortableWithComp& b) const {
            return a.value / 10 < b.value / 10;
        }
    };
    static TEST_CONSTEXPR bool less(const TrivialSortableWithComp& a, const TrivialSortableWithComp& b) {
        return a.value < b.value;
    }
};

struct NonTrivialSortableWithComp {
    int value;
    TEST_CONSTEXPR NonTrivialSortableWithComp() : value(0) {}
    TEST_CONSTEXPR NonTrivialSortableWithComp(int v) : value(v) {}
    TEST_CONSTEXPR NonTrivialSortableWithComp(const NonTrivialSortableWithComp& rhs) : value(rhs.value) {}
    TEST_CONSTEXPR_CXX14 NonTrivialSortableWithComp& operator=(const NonTrivialSortableWithComp& rhs) { value = rhs.value; return *this; }
    struct Comparator {
        TEST_CONSTEXPR bool operator()(const NonTrivialSortableWithComp& a, const NonTrivialSortableWithComp& b) const {
            return a.value / 10 < b.value / 10;
        }
    };
    static TEST_CONSTEXPR bool less(const NonTrivialSortableWithComp& a, const NonTrivialSortableWithComp& b) {
        return a.value < b.value;
    }
};

static_assert(std::is_trivially_copyable<TrivialSortable>::value, "");
static_assert(std::is_trivially_copyable<TrivialSortableWithComp>::value, "");
static_assert(!std::is_trivially_copyable<NonTrivialSortable>::value, "");
static_assert(!std::is_trivially_copyable<NonTrivialSortableWithComp>::value, "");

#endif // SORTABLE_HELPERS_H
