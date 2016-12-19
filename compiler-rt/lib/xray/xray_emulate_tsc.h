#pragma once
#include <time.h>

#include "sanitizer_common/sanitizer_internal_defs.h"
#include "xray_defs.h"

namespace __xray {

static constexpr uint64_t NanosecondsPerSecond = 1000ULL * 1000 * 1000;

ALWAYS_INLINE uint64_t readTSC(uint8_t &CPU) XRAY_NEVER_INSTRUMENT {
  timespec TS;
  int result = clock_gettime(CLOCK_REALTIME, &TS);
  if (result != 0) {
    Report("clock_gettime(2) returned %d, errno=%d.", result, int(errno));
    TS.tv_sec = 0;
    TS.tv_nsec = 0;
  }
  CPU = 0;
  return TS.tv_sec * NanosecondsPerSecond + TS.tv_nsec;
}

}
