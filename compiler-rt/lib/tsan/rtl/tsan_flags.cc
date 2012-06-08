//===-- tsan_flags.cc -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_libc.h"
#include "tsan_flags.h"
#include "tsan_rtl.h"
#include "tsan_mman.h"

namespace __tsan {

static void Flag(const char *env, bool *flag, const char *name);
static void Flag(const char *env, int *flag, const char *name);
static void Flag(const char *env, const char **flag, const char *name);

Flags *flags() {
  return &CTX()->flags;
}

// Can be overriden in frontend.
void WEAK OverrideFlags(Flags *f) {
  (void)f;
}

void InitializeFlags(Flags *f, const char *env) {
  real_memset(f, 0, sizeof(*f));

  // Default values.
  f->enable_annotations = true;
  f->suppress_equal_stacks = true;
  f->suppress_equal_addresses = true;
  f->report_thread_leaks = true;
  f->report_signal_unsafe = true;
  f->force_seq_cst_atomics = false;
  f->strip_path_prefix = "";
  f->suppressions = "";
  f->exitcode = 66;
  f->log_fileno = 2;
  f->atexit_sleep_ms = 1000;
  f->verbosity = 0;
  f->profile_memory = "";
  f->flush_memory_ms = 0;
  f->stop_on_start = false;
  f->running_on_valgrind = false;


  // Let a frontend override.
  OverrideFlags(f);

  // Override from command line.
  Flag(env, &f->enable_annotations, "enable_annotations");
  Flag(env, &f->suppress_equal_stacks, "suppress_equal_stacks");
  Flag(env, &f->suppress_equal_addresses, "suppress_equal_addresses");
  Flag(env, &f->report_thread_leaks, "report_thread_leaks");
  Flag(env, &f->report_signal_unsafe, "report_signal_unsafe");
  Flag(env, &f->force_seq_cst_atomics, "force_seq_cst_atomics");
  Flag(env, &f->strip_path_prefix, "strip_path_prefix");
  Flag(env, &f->suppressions, "suppressions");
  Flag(env, &f->exitcode, "exitcode");
  Flag(env, &f->log_fileno, "log_fileno");
  Flag(env, &f->atexit_sleep_ms, "atexit_sleep_ms");
  Flag(env, &f->verbosity, "verbosity");
  Flag(env, &f->profile_memory, "profile_memory");
  Flag(env, &f->flush_memory_ms, "flush_memory_ms");
  Flag(env, &f->stop_on_start, "stop_on_start");
}

static const char *GetFlagValue(const char *env, const char *name,
                                const char **end) {
  if (env == 0)
    return *end = 0;
  const char *pos = internal_strstr(env, name);
  if (pos == 0)
    return *end = 0;
  pos += internal_strlen(name);
  if (pos[0] != '=')
    return *end = pos;
  pos += 1;
  if (pos[0] == '"') {
    pos += 1;
    *end = internal_strchr(pos, '"');
  } else if (pos[0] == '\'') {
    pos += 1;
    *end = internal_strchr(pos, '\'');
  } else {
    *end = internal_strchr(pos, ' ');
  }
  if (*end == 0)
    *end = pos + internal_strlen(pos);
  return pos;
}

static void Flag(const char *env, bool *flag, const char *name) {
  const char *end = 0;
  const char *val = GetFlagValue(env, name, &end);
  if (val == 0)
    return;
  int len = end - val;
  if (len == 1 && val[0] == '0')
    *flag = false;
  else if (len == 1 && val[0] == '1')
    *flag = true;
}

static void Flag(const char *env, int *flag, const char *name) {
  const char *end = 0;
  const char *val = GetFlagValue(env, name, &end);
  if (val == 0)
    return;
  bool minus = false;
  if (val != end && val[0] == '-') {
    minus = true;
    val += 1;
  }
  int v = 0;
  for (; val != end; val++) {
    if (val[0] < '0' || val[0] > '9')
      break;
    v = v * 10 + val[0] - '0';
  }
  if (minus)
    v = -v;
  *flag = v;
}

static void Flag(const char *env, const char **flag, const char *name) {
  const char *end = 0;
  const char *val = GetFlagValue(env, name, &end);
  if (val == 0)
    return;
  int len = end - val;
  char *f = (char*)internal_alloc(MBlockFlag, len + 1);
  internal_memcpy(f, val, len);
  f[len] = 0;
  *flag = f;
}

}  // namespace __tsan
