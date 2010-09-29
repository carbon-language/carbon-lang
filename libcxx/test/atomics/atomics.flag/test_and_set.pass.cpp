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

// bool test_and_set(memory_order = memory_order_seq_cst);
// bool test_and_set(memory_order = memory_order_seq_cst) volatile;

#include <atomic>
#include <cassert>

int main()
{
    std::atomic_flag f;
    assert(f.test_and_set() == 0);
    assert(f.test_and_set() == 1);
}
