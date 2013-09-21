//===-- sanitizer_procmaps_test.cc ----------------------------------------===//
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
#include "sanitizer_common/sanitizer_procmaps.h"
//#include "sanitizer_common/sanitizer_internal_defs.h"
//#include "sanitizer_common/sanitizer_libc.h"
#include "gtest/gtest.h"

namespace __sanitizer {

#ifdef SANITIZER_LINUX
TEST(ProcMaps, CodeRange) {
  uptr start, end;
  bool res = GetCodeRangeForFile("[vdso]", &start, &end);
  EXPECT_EQ(res, true);
  EXPECT_GT(start, (uptr)0);
  EXPECT_LT(start, end);
}
#endif

}  // namespace __sanitizer
