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
#include "gtest/gtest.h"

#include <stdlib.h>

namespace __sanitizer {

#if SANITIZER_LINUX && !SANITIZER_ANDROID
TEST(MemoryMappingLayout, CodeRange) {
  uptr start, end;
  bool res = GetCodeRangeForFile("[vdso]", &start, &end);
  EXPECT_EQ(res, true);
  EXPECT_GT(start, 0U);
  EXPECT_LT(start, end);
}
#endif

static void noop() {}

TEST(MemoryMappingLayout, DumpListOfModules) {
  MemoryMappingLayout memory_mapping(false);
  const uptr kMaxModules = 10;
  LoadedModule *modules =
      (LoadedModule *)malloc(kMaxModules * sizeof(LoadedModule));
  uptr n_modules = memory_mapping.DumpListOfModules(modules, kMaxModules, 0);
  EXPECT_GT(n_modules, 0U);
  bool found = false;
  for (uptr i = 0; i < n_modules; ++i) {
    if (modules[i].containsAddress((uptr)&noop)) {
      // Verify that the module name is sane.
      if (strstr(modules[i].full_name(), "Sanitizer") != 0)
        found = true;
    }
  }
  EXPECT_TRUE(found);
  free(modules);
}

}  // namespace __sanitizer
