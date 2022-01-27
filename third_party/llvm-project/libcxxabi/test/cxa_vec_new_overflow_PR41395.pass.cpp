//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// PR41395 isn't fixed until the dylib shipped with macOS 10.15
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14}}

#include "cxxabi.h"
#include <new>
#include <cassert>

void dummy_ctor(void*) { assert(false && "should not be called"); }
void dummy_dtor(void*) { assert(false && "should not be called"); }

void *dummy_alloc(size_t) { assert(false && "should not be called"); }
void dummy_dealloc(void*) { assert(false && "should not be called"); }
void dummy_dealloc_sized(void*, size_t) { assert(false && "should not be called"); }


bool check_mul_overflows(size_t x, size_t y) {
  size_t tmp = x * y;
  if (tmp / x != y)
    return true;
  return false;
}

bool check_add_overflows(size_t x, size_t y) {
  size_t tmp = x + y;
  if (tmp < x)
    return true;

  return false;
}

void test_overflow_in_multiplication() {
  const size_t elem_count = std::size_t(1) << (sizeof(std::size_t) * 8 - 2);
  const size_t elem_size = 8;
  const size_t padding = 0;
  assert(check_mul_overflows(elem_count, elem_size));

  try {
    __cxxabiv1::__cxa_vec_new(elem_count, elem_size, padding, dummy_ctor,
                              dummy_dtor);
    assert(false && "allocation should fail");
  } catch (std::bad_array_new_length const&) {
    // OK
  } catch (...) {
    assert(false && "unexpected exception");
  }

  try {
    __cxxabiv1::__cxa_vec_new2(elem_count, elem_size, padding, dummy_ctor,
                              dummy_dtor, &dummy_alloc, &dummy_dealloc);
    assert(false && "allocation should fail");
  } catch (std::bad_array_new_length const&) {
    // OK
  } catch (...) {
    assert(false && "unexpected exception");
  }

  try {
    __cxxabiv1::__cxa_vec_new3(elem_count, elem_size, padding, dummy_ctor,
                               dummy_dtor, &dummy_alloc, &dummy_dealloc_sized);
    assert(false && "allocation should fail");
  } catch (std::bad_array_new_length const&) {
    // OK
  } catch (...) {
    assert(false && "unexpected exception");
  }
}

void test_overflow_in_addition() {
  const size_t elem_size = 4;
  const size_t elem_count = static_cast<size_t>(-1) / 4u;
#if defined(_LIBCXXABI_ARM_EHABI)
  const size_t padding = 8;
#else
  const size_t padding = sizeof(std::size_t);
#endif
  assert(!check_mul_overflows(elem_count, elem_size));
  assert(check_add_overflows(elem_count * elem_size, padding));
  try {
    __cxxabiv1::__cxa_vec_new(elem_count, elem_size, padding, dummy_ctor,
                              dummy_dtor);
    assert(false && "allocation should fail");
  } catch (std::bad_array_new_length const&) {
    // OK
  } catch (...) {
    assert(false && "unexpected exception");
  }


  try {
    __cxxabiv1::__cxa_vec_new2(elem_count, elem_size, padding, dummy_ctor,
                               dummy_dtor, &dummy_alloc, &dummy_dealloc);
    assert(false && "allocation should fail");
  } catch (std::bad_array_new_length const&) {
    // OK
  } catch (...) {
    assert(false && "unexpected exception");
  }

  try {
    __cxxabiv1::__cxa_vec_new3(elem_count, elem_size, padding, dummy_ctor,
                               dummy_dtor, &dummy_alloc, &dummy_dealloc_sized);
    assert(false && "allocation should fail");
  } catch (std::bad_array_new_length const&) {
    // OK
  } catch (...) {
    assert(false && "unexpected exception");
  }
}

int main(int, char**) {
  test_overflow_in_multiplication();
  test_overflow_in_addition();

  return 0;
}
