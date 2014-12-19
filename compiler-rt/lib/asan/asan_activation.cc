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
#include "asan_poisoning.h"
#include "asan_stack.h"
#include "sanitizer_common/sanitizer_flags.h"

namespace __asan {

static struct AsanDeactivatedFlags {
  int quarantine_size;
  int max_redzone;
  int malloc_context_size;
  bool poison_heap;
  bool alloc_dealloc_mismatch;
  bool allocator_may_return_null;

  void CopyFrom(const Flags *f, const CommonFlags *cf) {
    quarantine_size = f->quarantine_size;
    max_redzone = f->max_redzone;
    malloc_context_size = cf->malloc_context_size;
    poison_heap = f->poison_heap;
    alloc_dealloc_mismatch = f->alloc_dealloc_mismatch;
    allocator_may_return_null = cf->allocator_may_return_null;
  }

  void OverrideFromActivationFlags() {
    Flags f;
    CommonFlags cf;

    // Copy the current activation flags.
    f.quarantine_size = quarantine_size;
    f.max_redzone = max_redzone;
    cf.malloc_context_size = malloc_context_size;
    f.poison_heap = poison_heap;
    f.alloc_dealloc_mismatch = alloc_dealloc_mismatch;
    cf.allocator_may_return_null = allocator_may_return_null;

    // Check if activation flags need to be overriden.
    // FIXME: Add diagnostic to check that activation flags string doesn't
    // contain any other flags.
    char buf[100];
    GetExtraActivationFlags(buf, sizeof(buf));
    ParseCommonFlagsFromString(&cf, buf);
    ParseFlagsFromString(&f, buf);

    CopyFrom(&f, &cf);
  }

  void Print() {
    Report("quarantine_size %d, max_redzone %d, poison_heap %d, "
           "malloc_context_size %d, alloc_dealloc_mismatch %d, "
           "allocator_may_return_null %d\n",
           quarantine_size, max_redzone, poison_heap, malloc_context_size,
           alloc_dealloc_mismatch, allocator_may_return_null);
  }
} asan_deactivated_flags;

static bool asan_is_deactivated;

void AsanStartDeactivated() {
  VReport(1, "Deactivating ASan\n");
  // Save flag values.
  asan_deactivated_flags.CopyFrom(flags(), common_flags());

  // FIXME: Don't overwrite commandline flags. Instead, make the flags store
  // the original values calculated during flag parsing, and re-initialize
  // the necessary runtime objects.
  flags()->quarantine_size = 0;
  flags()->max_redzone = 16;
  flags()->poison_heap = false;
  common_flags()->malloc_context_size = 0;
  flags()->alloc_dealloc_mismatch = false;
  common_flags()->allocator_may_return_null = true;

  asan_is_deactivated = true;
}

void AsanActivate() {
  if (!asan_is_deactivated) return;
  VReport(1, "Activating ASan\n");

  asan_deactivated_flags.OverrideFromActivationFlags();

  // Restore flag values.
  // FIXME: this is not atomic, and there may be other threads alive.
  flags()->max_redzone = asan_deactivated_flags.max_redzone;
  flags()->alloc_dealloc_mismatch =
      asan_deactivated_flags.alloc_dealloc_mismatch;

  SetCanPoisonMemory(asan_deactivated_flags.poison_heap);
  SetMallocContextSize(asan_deactivated_flags.malloc_context_size);
  ReInitializeAllocator(asan_deactivated_flags.allocator_may_return_null,
                        asan_deactivated_flags.quarantine_size);

  asan_is_deactivated = false;
  if (common_flags()->verbosity) {
    Report("Activated with flags:\n");
    asan_deactivated_flags.Print();
  }
}

}  // namespace __asan
