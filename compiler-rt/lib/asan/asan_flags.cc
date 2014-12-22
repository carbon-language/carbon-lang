//===-- asan_flags.cc -------------------------------------------*- C++ -*-===//
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
// ASan flag parsing logic.
//===----------------------------------------------------------------------===//

#include "asan_activation.h"
#include "asan_flags.h"
#include "asan_interface_internal.h"
#include "asan_stack.h"
#include "lsan/lsan_common.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"

namespace __asan {

Flags asan_flags_dont_use_directly;  // use via flags().

static const char *MaybeCallAsanDefaultOptions() {
  return (&__asan_default_options) ? __asan_default_options() : "";
}

static const char *MaybeUseAsanDefaultOptionsCompileDefinition() {
#ifdef ASAN_DEFAULT_OPTIONS
// Stringize the macro value.
# define ASAN_STRINGIZE(x) #x
# define ASAN_STRINGIZE_OPTIONS(options) ASAN_STRINGIZE(options)
  return ASAN_STRINGIZE_OPTIONS(ASAN_DEFAULT_OPTIONS);
#else
  return "";
#endif
}

void ParseFlagsFromString(Flags *f, const char *str) {
  // Please write meaningful flag descriptions when adding new flags.
  ParseFlag(str, &f->quarantine_size, "quarantine_size",
            "Size (in bytes) of quarantine used to detect use-after-free "
            "errors. Lower value may reduce memory usage but increase the "
            "chance of false negatives.");
  ParseFlag(str, &f->redzone, "redzone",
            "Minimal size (in bytes) of redzones around heap objects. "
            "Requirement: redzone >= 16, is a power of two.");
  ParseFlag(str, &f->max_redzone, "max_redzone",
            "Maximal size (in bytes) of redzones around heap objects.");

  ParseFlag(str, &f->debug, "debug",
      "If set, prints some debugging information and does additional checks.");
  ParseFlag(str, &f->report_globals, "report_globals",
      "Controls the way to handle globals (0 - don't detect buffer overflow on "
      "globals, 1 - detect buffer overflow, 2 - print data about registered "
      "globals).");

  ParseFlag(str, &f->check_initialization_order,
      "check_initialization_order",
      "If set, attempts to catch initialization order issues.");

  ParseFlag(str, &f->replace_str, "replace_str",
      "If set, uses custom wrappers and replacements for libc string functions "
      "to find more errors.");

  ParseFlag(str, &f->replace_intrin, "replace_intrin",
      "If set, uses custom wrappers for memset/memcpy/memmove intinsics.");
  ParseFlag(str, &f->mac_ignore_invalid_free, "mac_ignore_invalid_free",
      "Ignore invalid free() calls to work around some bugs. Used on OS X "
      "only.");
  ParseFlag(str, &f->detect_stack_use_after_return,
      "detect_stack_use_after_return",
      "Enables stack-use-after-return checking at run-time.");
  ParseFlag(str, &f->min_uar_stack_size_log, "min_uar_stack_size_log",
      "Minimum fake stack size log.");
  ParseFlag(str, &f->max_uar_stack_size_log, "max_uar_stack_size_log",
      "Maximum fake stack size log.");
  ParseFlag(str, &f->uar_noreserve, "uar_noreserve",
      "Use mmap with 'norserve' flag to allocate fake stack.");
  ParseFlag(str, &f->max_malloc_fill_size, "max_malloc_fill_size",
      "ASan allocator flag. max_malloc_fill_size is the maximal amount of "
      "bytes that will be filled with malloc_fill_byte on malloc.");
  ParseFlag(str, &f->malloc_fill_byte, "malloc_fill_byte",
      "Value used to fill the newly allocated memory.");
  ParseFlag(str, &f->exitcode, "exitcode",
      "Override the program exit status if the tool found an error.");
  ParseFlag(str, &f->allow_user_poisoning, "allow_user_poisoning",
      "If set, user may manually mark memory regions as poisoned or "
      "unpoisoned.");
  ParseFlag(str, &f->sleep_before_dying, "sleep_before_dying",
      "Number of seconds to sleep between printing an error report and "
      "terminating the program. Useful for debugging purposes (e.g. when one "
      "needs to attach gdb).");

  ParseFlag(str, &f->check_malloc_usable_size, "check_malloc_usable_size",
      "Allows the users to work around the bug in Nvidia drivers prior to "
      "295.*.");

  ParseFlag(str, &f->unmap_shadow_on_exit, "unmap_shadow_on_exit",
      "If set, explicitly unmaps the (huge) shadow at exit.");
  ParseFlag(str, &f->abort_on_error, "abort_on_error",
      "If set, the tool calls abort() instead of _exit() after printing the "
      "error report.");
  ParseFlag(str, &f->print_stats, "print_stats",
      "Print various statistics after printing an error message or if "
      "atexit=1.");
  ParseFlag(str, &f->print_legend, "print_legend",
      "Print the legend for the shadow bytes.");
  ParseFlag(str, &f->atexit, "atexit",
      "If set, prints ASan exit stats even after program terminates "
      "successfully.");

  ParseFlag(str, &f->print_full_thread_history,
      "print_full_thread_history",
      "If set, prints thread creation stacks for the threads involved in the "
      "report and their ancestors up to the main thread.");

  ParseFlag(str, &f->poison_heap, "poison_heap",
      "Poison (or not) the heap memory on [de]allocation. Zero value is useful "
      "for benchmarking the allocator or instrumentator.");

  ParseFlag(str, &f->poison_array_cookie, "poison_array_cookie",
      "Poison (or not) the array cookie after operator new[].");

  ParseFlag(str, &f->poison_partial, "poison_partial",
      "If true, poison partially addressable 8-byte aligned words "
      "(default=true). This flag affects heap and global buffers, but not "
      "stack buffers.");

  ParseFlag(str, &f->alloc_dealloc_mismatch, "alloc_dealloc_mismatch",
      "Report errors on malloc/delete, new/free, new/delete[], etc.");

  ParseFlag(str, &f->new_delete_type_mismatch, "new_delete_type_mismatch",
      "Report errors on mismatch betwen size of new and delete.");

  ParseFlag(str, &f->strict_memcmp, "strict_memcmp",
      "If true, assume that memcmp(p1, p2, n) always reads n bytes before "
      "comparing p1 and p2.");

  ParseFlag(str, &f->strict_init_order, "strict_init_order",
      "If true, assume that dynamic initializers can never access globals from "
      "other modules, even if the latter are already initialized.");

  ParseFlag(str, &f->start_deactivated, "start_deactivated",
      "If true, ASan tweaks a bunch of other flags (quarantine, redzone, heap "
      "poisoning) to reduce memory consumption as much as possible, and "
      "restores them to original values when the first instrumented module is "
      "loaded into the process. This is mainly intended to be used on "
      "Android. ");

  ParseFlag(str, &f->detect_invalid_pointer_pairs,
      "detect_invalid_pointer_pairs",
      "If non-zero, try to detect operations like <, <=, >, >= and - on "
      "invalid pointer pairs (e.g. when pointers belong to different objects). "
      "The bigger the value the harder we try.");

  ParseFlag(str, &f->detect_container_overflow,
      "detect_container_overflow",
      "If true, honor the container overflow  annotations. "
      "See https://code.google.com/p/address-sanitizer/wiki/ContainerOverflow");

  ParseFlag(str, &f->detect_odr_violation, "detect_odr_violation",
            "If >=2, detect violation of One-Definition-Rule (ODR); "
            "If ==1, detect ODR-violation only if the two variables "
            "have different sizes");

  ParseFlag(str, &f->dump_instruction_bytes, "dump_instruction_bytes",
      "If true, dump 16 bytes starting at the instruction that caused SEGV");
}

void InitializeFlags(Flags *f) {
  CommonFlags *cf = common_flags();
  SetCommonFlagsDefaults();
  cf->detect_leaks = CAN_SANITIZE_LEAKS;
  cf->external_symbolizer_path = GetEnv("ASAN_SYMBOLIZER_PATH");
  cf->malloc_context_size = kDefaultMallocContextSize;
  cf->intercept_tls_get_addr = true;
  cf->coverage = false;

  internal_memset(f, 0, sizeof(*f));
  f->quarantine_size = (ASAN_LOW_MEMORY) ? 1UL << 26 : 1UL << 28;
  f->redzone = 16;
  f->max_redzone = 2048;
  f->debug = false;
  f->report_globals = 1;
  f->check_initialization_order = false;
  f->replace_str = true;
  f->replace_intrin = true;
  f->mac_ignore_invalid_free = false;
  f->detect_stack_use_after_return = false;  // Also needs the compiler flag.
  f->min_uar_stack_size_log = 16;  // We can't do smaller anyway.
  f->max_uar_stack_size_log = 20;  // 1Mb per size class, i.e. ~11Mb per thread.
  f->uar_noreserve = false;
  f->max_malloc_fill_size = 0x1000;  // By default, fill only the first 4K.
  f->malloc_fill_byte = 0xbe;
  f->exitcode = ASAN_DEFAULT_FAILURE_EXITCODE;
  f->allow_user_poisoning = true;
  f->sleep_before_dying = 0;
  f->check_malloc_usable_size = true;
  f->unmap_shadow_on_exit = false;
  f->abort_on_error = false;
  f->print_stats = false;
  f->print_legend = true;
  f->atexit = false;
  f->print_full_thread_history = true;
  f->poison_heap = true;
  f->poison_array_cookie = true;
  f->poison_partial = true;
  // Turn off alloc/dealloc mismatch checker on Mac and Windows for now.
  // https://code.google.com/p/address-sanitizer/issues/detail?id=131
  // https://code.google.com/p/address-sanitizer/issues/detail?id=309
  // TODO(glider,timurrrr): Fix known issues and enable this back.
  f->alloc_dealloc_mismatch = (SANITIZER_MAC == 0) && (SANITIZER_WINDOWS == 0);
  f->new_delete_type_mismatch = true;
  f->strict_memcmp = true;
  f->strict_init_order = false;
  f->start_deactivated = false;
  f->detect_invalid_pointer_pairs = 0;
  f->detect_container_overflow = true;
  f->detect_odr_violation = 2;
  f->dump_instruction_bytes = false;

  // Override from compile definition.
  const char *compile_def = MaybeUseAsanDefaultOptionsCompileDefinition();
  ParseCommonFlagsFromString(compile_def);
  ParseFlagsFromString(f, compile_def);

  // Override from user-specified string.
  const char *default_options = MaybeCallAsanDefaultOptions();
  ParseCommonFlagsFromString(default_options);
  ParseFlagsFromString(f, default_options);
  VReport(1, "Using the defaults from __asan_default_options: %s\n",
          MaybeCallAsanDefaultOptions());

  // Override from command line.
  if (const char *env = GetEnv("ASAN_OPTIONS")) {
    ParseCommonFlagsFromString(env);
    ParseFlagsFromString(f, env);
    VReport(1, "Parsed ASAN_OPTIONS: %s\n", env);
  }

  // Let activation flags override current settings. On Android they come
  // from a system property. On other platforms this is no-op.
  if (!flags()->start_deactivated) {
    char buf[100];
    GetExtraActivationFlags(buf, sizeof(buf));
    ParseCommonFlagsFromString(buf);
    ParseFlagsFromString(f, buf);
    if (buf[0] != '\0')
      VReport(1, "Parsed activation flags: %s\n", buf);
  }

  if (common_flags()->help) {
    PrintFlagDescriptions();
  }

  // Flag validation:
  if (!CAN_SANITIZE_LEAKS && cf->detect_leaks) {
    Report("%s: detect_leaks is not supported on this platform.\n",
           SanitizerToolName);
    cf->detect_leaks = false;
  }
  // Make "strict_init_order" imply "check_initialization_order".
  // TODO(samsonov): Use a single runtime flag for an init-order checker.
  if (f->strict_init_order) {
    f->check_initialization_order = true;
  }
  CHECK_LE((uptr)cf->malloc_context_size, kStackTraceMax);
  CHECK_LE(f->min_uar_stack_size_log, f->max_uar_stack_size_log);
  CHECK_GE(f->redzone, 16);
  CHECK_GE(f->max_redzone, f->redzone);
  CHECK_LE(f->max_redzone, 2048);
  CHECK(IsPowerOfTwo(f->redzone));
  CHECK(IsPowerOfTwo(f->max_redzone));
}

}  // namespace __asan

#if !SANITIZER_SUPPORTS_WEAK_HOOKS
extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
const char* __asan_default_options() { return ""; }
}  // extern "C"
#endif
