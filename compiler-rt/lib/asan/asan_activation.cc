//===-- asan_activation.cc --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// ASan activation/deactivation logic.
//===----------------------------------------------------------------------===//

#include "asan_activation.h"
#include "asan_allocator.h"
#include "asan_flags.h"
#include "asan_internal.h"
#include "sanitizer_common/sanitizer_flags.h"

namespace __asan {

static struct AsanDeactivatedFlags {
  int quarantine_size;
  int max_redzone;
  int malloc_context_size;
  bool poison_heap;
} asan_deactivated_flags;

static bool asan_is_deactivated;

void AsanStartDeactivated() {
  VReport(1, "Deactivating ASan\n");
  // Save flag values.
  asan_deactivated_flags.quarantine_size = flags()->quarantine_size;
  asan_deactivated_flags.max_redzone = flags()->max_redzone;
  asan_deactivated_flags.poison_heap = flags()->poison_heap;
  asan_deactivated_flags.malloc_context_size =
      common_flags()->malloc_context_size;

  flags()->quarantine_size = 0;
  flags()->max_redzone = 16;
  flags()->poison_heap = false;
  common_flags()->malloc_context_size = 0;

  asan_is_deactivated = true;
}

void AsanActivate() {
  if (!asan_is_deactivated) return;
  VReport(1, "Activating ASan\n");

  // Restore flag values.
  // FIXME: this is not atomic, and there may be other threads alive.
  flags()->quarantine_size = asan_deactivated_flags.quarantine_size;
  flags()->max_redzone = asan_deactivated_flags.max_redzone;
  flags()->poison_heap = asan_deactivated_flags.poison_heap;
  common_flags()->malloc_context_size =
      asan_deactivated_flags.malloc_context_size;

  ParseExtraActivationFlags();

  ReInitializeAllocator();

  asan_is_deactivated = false;
  VReport(
      1,
      "quarantine_size %d, max_redzone %d, poison_heap %d, malloc_context_size "
      "%d\n",
      flags()->quarantine_size, flags()->max_redzone, flags()->poison_heap,
      common_flags()->malloc_context_size);
}

}  // namespace __asan
