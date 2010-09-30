//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <atomic>

// struct atomic_flag

// bool atomic_flag_test_and_set(volatile atomic_flag*);
// bool atomic_flag_test_and_set(atomic_flag*);

#include <atomic>
#include <cassert>

int main()
{
    {
        std::atomic_flag f;
        f.clear();
        assert(atomic_flag_test_and_set(&f) == 0);
        assert(f.test_and_set() == 1);
    }
    {
        volatile std::atomic_flag f;
        f.clear();
        assert(atomic_flag_test_and_set(&f) == 0);
        assert(f.test_and_set() == 1);
    }
}
