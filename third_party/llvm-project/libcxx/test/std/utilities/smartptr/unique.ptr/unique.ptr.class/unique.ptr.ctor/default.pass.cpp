//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

//=============================================================================
// TESTING std::unique_ptr::unique_ptr()
//
// Concerns:
//   1 The default constructor works for any default constructible deleter types.
//   2 The stored type 'T' is allowed to be incomplete.
//
// Plan
//  1 Default construct unique_ptr's with various deleter types (C-1)
//  2 Default construct a unique_ptr with an incomplete element_type and
//    various deleter types (C-1,2)

#include <memory>
#include <cassert>
#include "test_macros.h"

#include "test_macros.h"
#include "deleter_types.h"
#include "unique_ptr_test_helper.h"

#if TEST_STD_VER >= 11
TEST_CONSTINIT std::unique_ptr<int> global_static_unique_ptr_single;
TEST_CONSTINIT std::unique_ptr<int[]> global_static_unique_ptr_runtime;

struct NonDefaultDeleter {
  NonDefaultDeleter() = delete;
  void operator()(void*) const {}
};
#endif

template <class ElemType>
void test_sfinae() {
#if TEST_STD_VER >= 11
  { // the constructor does not participate in overload resolution when
    // the deleter is a pointer type
    using U = std::unique_ptr<ElemType, void (*)(void*)>;
    static_assert(!std::is_default_constructible<U>::value, "");
  }
  { // the constructor does not participate in overload resolution when
    // the deleter is not default constructible
    using Del = CDeleter<ElemType>;
    using U1 = std::unique_ptr<ElemType, NonDefaultDeleter>;
    using U2 = std::unique_ptr<ElemType, Del&>;
    using U3 = std::unique_ptr<ElemType, Del const&>;
    static_assert(!std::is_default_constructible<U1>::value, "");
    static_assert(!std::is_default_constructible<U2>::value, "");
    static_assert(!std::is_default_constructible<U3>::value, "");
  }
#endif
}

template <class ElemType>
void test_basic() {
#if TEST_STD_VER >= 11
  {
    using U1 = std::unique_ptr<ElemType>;
    using U2 = std::unique_ptr<ElemType, Deleter<ElemType> >;
    static_assert(std::is_nothrow_default_constructible<U1>::value, "");
    static_assert(std::is_nothrow_default_constructible<U2>::value, "");
  }
#endif
  {
    std::unique_ptr<ElemType> p;
    assert(p.get() == 0);
  }
  {
    std::unique_ptr<ElemType, NCDeleter<ElemType> > p;
    assert(p.get() == 0);
    assert(p.get_deleter().state() == 0);
    p.get_deleter().set_state(5);
    assert(p.get_deleter().state() == 5);
  }
  {
    std::unique_ptr<ElemType, DefaultCtorDeleter<ElemType> > p;
    assert(p.get() == 0);
    assert(p.get_deleter().state() == 0);
  }
}

DEFINE_AND_RUN_IS_INCOMPLETE_TEST({
  doIncompleteTypeTest(0);
  doIncompleteTypeTest<IncompleteType, Deleter<IncompleteType> >(0);
} {
  doIncompleteTypeTest<IncompleteType[]>(0);
  doIncompleteTypeTest<IncompleteType[], Deleter<IncompleteType[]> >(0);
})

int main(int, char**) {
  {
    test_sfinae<int>();
    test_basic<int>();
  }
  {
    test_sfinae<int[]>();
    test_basic<int[]>();
  }

  return 0;
}
