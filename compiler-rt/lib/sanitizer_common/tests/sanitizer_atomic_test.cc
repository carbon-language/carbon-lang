//===-- sanitizer_atomic_test.cc ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_atomic.h"
#include "gtest/gtest.h"

namespace __sanitizer {

// Clang crashes while compiling this test for Android:
// http://llvm.org/bugs/show_bug.cgi?id=15587
#if !SANITIZER_ANDROID
template<typename T>
void CheckAtomicCompareExchange() {
  typedef typename T::Type Type;
  {
    Type old_val = 42;
    Type new_val = 24;
    Type var = old_val;
    EXPECT_TRUE(atomic_compare_exchange_strong((T*)&var, &old_val, new_val,
                                               memory_order_relaxed));
    EXPECT_FALSE(atomic_compare_exchange_strong((T*)&var, &old_val, new_val,
                                                memory_order_relaxed));
    EXPECT_EQ(new_val, old_val);
  }
  {
    Type old_val = 42;
    Type new_val = 24;
    Type var = old_val;
    EXPECT_TRUE(atomic_compare_exchange_weak((T*)&var, &old_val, new_val,
                                             memory_order_relaxed));
    EXPECT_FALSE(atomic_compare_exchange_weak((T*)&var, &old_val, new_val,
                                              memory_order_relaxed));
    EXPECT_EQ(new_val, old_val);
  }
}

TEST(SanitizerCommon, AtomicCompareExchangeTest) {
  CheckAtomicCompareExchange<atomic_uint8_t>();
  CheckAtomicCompareExchange<atomic_uint16_t>();
  CheckAtomicCompareExchange<atomic_uint32_t>();
  CheckAtomicCompareExchange<atomic_uint64_t>();
  CheckAtomicCompareExchange<atomic_uintptr_t>();
}
#endif  //!SANITIZER_ANDROID

}  // namespace __sanitizer
