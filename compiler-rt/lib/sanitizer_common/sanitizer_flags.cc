//===-- sanitizer_flags.cc ------------------------------------------------===//
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

#include "sanitizer_flags.h"

#include "sanitizer_common.h"
#include "sanitizer_libc.h"
#include "sanitizer_list.h"

namespace __sanitizer {

CommonFlags common_flags_dont_use;

struct FlagDescription {
  const char *name;
  const char *description;
  FlagDescription *next;
};

IntrusiveList<FlagDescription> flag_descriptions;

void SetCommonFlagsDefaults(CommonFlags *f) {
  f->symbolize = true;
  f->external_symbolizer_path = 0;
  f->allow_addr2line = false;
  f->strip_path_prefix = "";
  f->fast_unwind_on_fatal = false;
  f->fast_unwind_on_malloc = true;
  f->handle_ioctl = false;
  f->malloc_context_size = 1;
  f->log_path = "stderr";
  f->verbosity = 0;
  f->detect_leaks = true;
  f->leak_check_at_exit = true;
  f->allocator_may_return_null = false;
  f->print_summary = true;
  f->check_printf = true;
  // TODO(glider): tools may want to set different defaults for handle_segv.
  f->handle_segv = SANITIZER_NEEDS_SEGV;
  f->allow_user_segv_handler = false;
  f->use_sigaltstack = true;
  f->detect_deadlocks = false;
  f->clear_shadow_mmap_threshold = 64 * 1024;
  f->color = "auto";
  f->legacy_pthread_cond = false;
  f->intercept_tls_get_addr = false;
}

void ParseCommonFlagsFromString(CommonFlags *f, const char *str) {
  ParseFlag(str, &f->symbolize, "symbolize",
      "If set, use the online symbolizer from common sanitizer runtime to turn "
      "virtual addresses to file/line locations.");
  ParseFlag(str, &f->external_symbolizer_path, "external_symbolizer_path",
      "Path to external symbolizer. If empty, the tool will search $PATH for "
      "the symbolizer.");
  ParseFlag(str, &f->allow_addr2line, "allow_addr2line",
      "If set, allows online symbolizer to run addr2line binary to symbolize "
      "stack traces (addr2line will only be used if llvm-symbolizer binary is "
      "unavailable.");
  ParseFlag(str, &f->strip_path_prefix, "strip_path_prefix",
      "Strips this prefix from file paths in error reports.");
  ParseFlag(str, &f->fast_unwind_on_fatal, "fast_unwind_on_fatal",
      "If available, use the fast frame-pointer-based unwinder on fatal "
      "errors.");
  ParseFlag(str, &f->fast_unwind_on_malloc, "fast_unwind_on_malloc",
      "If available, use the fast frame-pointer-based unwinder on "
      "malloc/free.");
  ParseFlag(str, &f->handle_ioctl, "handle_ioctl",
      "Intercept and handle ioctl requests.");
  ParseFlag(str, &f->malloc_context_size, "malloc_context_size",
      "Max number of stack frames kept for each allocation/deallocation.");
  ParseFlag(str, &f->log_path, "log_path",
      "Write logs to \"log_path.pid\". The special values are \"stdout\" and "
      "\"stderr\". The default is \"stderr\".");
  ParseFlag(str, &f->verbosity, "verbosity",
      "Verbosity level (0 - silent, 1 - a bit of output, 2+ - more output).");
  ParseFlag(str, &f->detect_leaks, "detect_leaks",
      "Enable memory leak detection.");
  ParseFlag(str, &f->leak_check_at_exit, "leak_check_at_exit",
      "Invoke leak checking in an atexit handler. Has no effect if "
      "detect_leaks=false, or if __lsan_do_leak_check() is called before the "
      "handler has a chance to run.");
  ParseFlag(str, &f->allocator_may_return_null, "allocator_may_return_null",
      "If false, the allocator will crash instead of returning 0 on "
      "out-of-memory.");
  ParseFlag(str, &f->print_summary, "print_summary",
      "If false, disable printing error summaries in addition to error "
      "reports.");
  ParseFlag(str, &f->check_printf, "check_printf",
      "Check printf arguments.");
  ParseFlag(str, &f->handle_segv, "handle_segv",
      "If set, registers the tool's custom SEGV handler (both SIGBUS and "
      "SIGSEGV on OSX).");
  ParseFlag(str, &f->allow_user_segv_handler, "allow_user_segv_handler",
      "If set, allows user to register a SEGV handler even if the tool "
      "registers one.");
  ParseFlag(str, &f->use_sigaltstack, "use_sigaltstack",
      "If set, uses alternate stack for signal handling.");
  ParseFlag(str, &f->detect_deadlocks, "detect_deadlocks",
      "If set, deadlock detection is enabled.");
  ParseFlag(str, &f->clear_shadow_mmap_threshold,
            "clear_shadow_mmap_threshold",
      "Large shadow regions are zero-filled using mmap(NORESERVE) instead of "
      "memset(). This is the threshold size in bytes.");
  ParseFlag(str, &f->color, "color",
      "Colorize reports: (always|never|auto).");
  ParseFlag(str, &f->legacy_pthread_cond, "legacy_pthread_cond",
      "Enables support for dynamic libraries linked with libpthread 2.2.5.");
  ParseFlag(str, &f->intercept_tls_get_addr, "intercept_tls_get_addr",
            "Intercept __tls_get_addr.");
  ParseFlag(str, &f->help, "help", "Print the flag descriptions.");
  ParseFlag(str, &f->mmap_limit_mb, "mmap_limit_mb",
            "Limit the amount of mmap-ed memory (excluding shadow) in Mb; "
            "not a user-facing flag, used mosly for testing the tools");

  // Do a sanity check for certain flags.
  if (f->malloc_context_size < 1)
    f->malloc_context_size = 1;
}

static bool GetFlagValue(const char *env, const char *name,
                         const char **value, int *value_length) {
  if (env == 0)
    return false;
  const char *pos = 0;
  for (;;) {
    pos = internal_strstr(env, name);
    if (pos == 0)
      return false;
    if (pos != env && ((pos[-1] >= 'a' && pos[-1] <= 'z') || pos[-1] == '_')) {
      // Seems to be middle of another flag name or value.
      env = pos + 1;
      continue;
    }
    break;
  }
  pos += internal_strlen(name);
  const char *end;
  if (pos[0] != '=') {
    end = pos;
  } else {
    pos += 1;
    if (pos[0] == '"') {
      pos += 1;
      end = internal_strchr(pos, '"');
    } else if (pos[0] == '\'') {
      pos += 1;
      end = internal_strchr(pos, '\'');
    } else {
      // Read until the next space or colon.
      end = pos + internal_strcspn(pos, " :");
    }
    if (end == 0)
      end = pos + internal_strlen(pos);
  }
  *value = pos;
  *value_length = end - pos;
  return true;
}

static bool StartsWith(const char *flag, int flag_length, const char *value) {
  if (!flag || !value)
    return false;
  int value_length = internal_strlen(value);
  return (flag_length >= value_length) &&
         (0 == internal_strncmp(flag, value, value_length));
}

static LowLevelAllocator allocator_for_flags;

// The linear scan is suboptimal, but the number of flags is relatively small.
bool FlagInDescriptionList(const char *name) {
  IntrusiveList<FlagDescription>::Iterator it(&flag_descriptions);
  while (it.hasNext()) {
    if (!internal_strcmp(it.next()->name, name)) return true;
  }
  return false;
}

void AddFlagDescription(const char *name, const char *description) {
  if (FlagInDescriptionList(name)) return;
  FlagDescription *new_description = new(allocator_for_flags) FlagDescription;
  new_description->name = name;
  new_description->description = description;
  flag_descriptions.push_back(new_description);
}

// TODO(glider): put the descriptions inside CommonFlags.
void PrintFlagDescriptions() {
  IntrusiveList<FlagDescription>::Iterator it(&flag_descriptions);
  Printf("Available flags for %s:\n", SanitizerToolName);
  while (it.hasNext()) {
    FlagDescription *descr = it.next();
    Printf("\t%s\n\t\t- %s\n", descr->name, descr->description);
  }
}

void ParseFlag(const char *env, bool *flag,
               const char *name, const char *descr) {
  const char *value;
  int value_length;
  AddFlagDescription(name, descr);
  if (!GetFlagValue(env, name, &value, &value_length))
    return;
  if (StartsWith(value, value_length, "0") ||
      StartsWith(value, value_length, "no") ||
      StartsWith(value, value_length, "false"))
    *flag = false;
  if (StartsWith(value, value_length, "1") ||
      StartsWith(value, value_length, "yes") ||
      StartsWith(value, value_length, "true"))
    *flag = true;
}

void ParseFlag(const char *env, int *flag,
               const char *name, const char *descr) {
  const char *value;
  int value_length;
  AddFlagDescription(name, descr);
  if (!GetFlagValue(env, name, &value, &value_length))
    return;
  *flag = static_cast<int>(internal_atoll(value));
}

void ParseFlag(const char *env, uptr *flag,
               const char *name, const char *descr) {
  const char *value;
  int value_length;
  AddFlagDescription(name, descr);
  if (!GetFlagValue(env, name, &value, &value_length))
    return;
  *flag = static_cast<uptr>(internal_atoll(value));
}

void ParseFlag(const char *env, const char **flag,
               const char *name, const char *descr) {
  const char *value;
  int value_length;
  AddFlagDescription(name, descr);
  if (!GetFlagValue(env, name, &value, &value_length))
    return;
  // Copy the flag value. Don't use locks here, as flags are parsed at
  // tool startup.
  char *value_copy = (char*)(allocator_for_flags.Allocate(value_length + 1));
  internal_memcpy(value_copy, value, value_length);
  value_copy[value_length] = '\0';
  *flag = value_copy;
}

}  // namespace __sanitizer
