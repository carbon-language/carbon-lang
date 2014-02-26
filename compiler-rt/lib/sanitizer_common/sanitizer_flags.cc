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

namespace __sanitizer {

CommonFlags common_flags_dont_use;

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
  f->detect_leaks = false;
  f->leak_check_at_exit = true;
  f->allocator_may_return_null = false;
  f->print_summary = true;
  f->check_printf = false;
  // TODO(glider): tools may want to set different defaults for handle_segv.
  f->handle_segv = SANITIZER_NEEDS_SEGV;
  f->allow_user_segv_handler = false;
  f->use_sigaltstack = false;
  f->detect_deadlocks = false;
  f->clear_shadow_mmap_threshold = 64 * 1024;
  f->color = "auto";
}

void ParseCommonFlagsFromString(CommonFlags *f, const char *str) {
  ParseFlag(str, &f->symbolize, "symbolize");
  ParseFlag(str, &f->external_symbolizer_path, "external_symbolizer_path");
  ParseFlag(str, &f->allow_addr2line, "allow_addr2line");
  ParseFlag(str, &f->strip_path_prefix, "strip_path_prefix");
  ParseFlag(str, &f->fast_unwind_on_fatal, "fast_unwind_on_fatal");
  ParseFlag(str, &f->fast_unwind_on_malloc, "fast_unwind_on_malloc");
  ParseFlag(str, &f->handle_ioctl, "handle_ioctl");
  ParseFlag(str, &f->malloc_context_size, "malloc_context_size");
  ParseFlag(str, &f->log_path, "log_path");
  ParseFlag(str, &f->verbosity, "verbosity");
  ParseFlag(str, &f->detect_leaks, "detect_leaks");
  ParseFlag(str, &f->leak_check_at_exit, "leak_check_at_exit");
  ParseFlag(str, &f->allocator_may_return_null, "allocator_may_return_null");
  ParseFlag(str, &f->print_summary, "print_summary");
  ParseFlag(str, &f->check_printf, "check_printf");
  ParseFlag(str, &f->handle_segv, "handle_segv");
  ParseFlag(str, &f->allow_user_segv_handler, "allow_user_segv_handler");
  ParseFlag(str, &f->use_sigaltstack, "use_sigaltstack");
  ParseFlag(str, &f->detect_deadlocks, "detect_deadlocks");
  ParseFlag(str, &f->clear_shadow_mmap_threshold,
            "clear_shadow_mmap_threshold");
  ParseFlag(str, &f->color, "color");

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

void ParseFlag(const char *env, bool *flag, const char *name) {
  const char *value;
  int value_length;
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

void ParseFlag(const char *env, int *flag, const char *name) {
  const char *value;
  int value_length;
  if (!GetFlagValue(env, name, &value, &value_length))
    return;
  *flag = static_cast<int>(internal_atoll(value));
}

void ParseFlag(const char *env, uptr *flag, const char *name) {
  const char *value;
  int value_length;
  if (!GetFlagValue(env, name, &value, &value_length))
    return;
  *flag = static_cast<uptr>(internal_atoll(value));
}

static LowLevelAllocator allocator_for_flags;

void ParseFlag(const char *env, const char **flag, const char *name) {
  const char *value;
  int value_length;
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
