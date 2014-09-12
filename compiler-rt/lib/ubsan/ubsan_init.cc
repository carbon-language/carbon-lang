//===-- ubsan_init.cc -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Initialization of UBSan runtime.
//
//===----------------------------------------------------------------------===//

#include "ubsan_init.h"
#include "ubsan_flags.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_suppressions.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

using namespace __ubsan;

static bool ubsan_inited;

void __ubsan::InitIfNecessary() {
#if !SANITIZER_CAN_USE_PREINIT_ARRAY
  // No need to lock mutex if we're initializing from preinit array.
  static StaticSpinMutex init_mu;
  SpinMutexLock l(&init_mu);
#endif
  if (LIKELY(ubsan_inited))
   return;
  if (0 == internal_strcmp(SanitizerToolName, "SanitizerTool")) {
    // WARNING: If this condition holds, then either UBSan runs in a standalone
    // mode, or initializer for another sanitizer hasn't run yet. In a latter
    // case, another sanitizer will overwrite "SanitizerToolName" and reparse
    // common flags. It means, that we are not allowed to *use* common flags
    // in this function.
    SanitizerToolName = "UndefinedBehaviorSanitizer";
    InitializeCommonFlags();
  }
  // Initialize UBSan-specific flags.
  InitializeFlags();
  SuppressionContext::InitIfNecessary();
  ubsan_inited = true;
}

#if SANITIZER_CAN_USE_PREINIT_ARRAY
__attribute__((section(".preinit_array"), used))
void (*__local_ubsan_preinit)(void) = __ubsan::InitIfNecessary;
#else
// Use a dynamic initializer.
class UbsanInitializer {
 public:
  UbsanInitializer() {
    InitIfNecessary();
  }
};
static UbsanInitializer ubsan_initializer;
#endif  // SANITIZER_CAN_USE_PREINIT_ARRAY
