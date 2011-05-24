//===----------------------------- test_guard.cpp -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "cxxabi.h"

#include <cassert>

namespace test1 {
    static int run_count = 0;
    int increment() {
        ++run_count;
        return 0;
    }
    void helper() {
        static int a = increment();
    }
    void test() {
        static int a = increment();
        assert(run_count == 1);
        static int b = increment();
        assert(run_count == 2);
        helper();
        assert(run_count == 3);
        helper();
        assert(run_count == 3);
    }
}

namespace test2 {
    static int run_count = 0;
    int increment() {
        ++run_count;
        throw 0;
    }
    void helper() {
        try {
            static int a = increment();
            assert(0);
        } catch (...) {}
    }
    void test() {
        helper();
        assert(run_count == 1);
        helper();
        assert(run_count == 2);
    }
}

int main()
{
    test1::test();
    test2::test();
}
