#pragma once
#include <x86intrin.h>

#include "sanitizer_common/sanitizer_internal_defs.h"
#include "xray_defs.h"

namespace __xray {

ALWAYS_INLINE uint64_t readTSC(uint8_t &CPU) XRAY_NEVER_INSTRUMENT {
  unsigned LongCPU;
  uint64_t TSC = __rdtscp(&LongCPU);
  CPU = LongCPU;
  return TSC;
}

}
