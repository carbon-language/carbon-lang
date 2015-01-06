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

// If set, the tool will install its own SEGV signal handler by default.
#ifndef SANITIZER_NEEDS_SEGV
# define SANITIZER_NEEDS_SEGV 1
#endif

void CommonFlags::SetDefaults() {
#define COMMON_FLAG(Type, Name, DefaultValue, Description) Name = DefaultValue;
#include "sanitizer_flags.inc"
#undef COMMON_FLAG
}

void CommonFlags::ParseFromString(const char *str) {
#define COMMON_FLAG(Type, Name, DefaultValue, Description)                     \
  ParseFlag(str, &Name, #Name, Description);
#include "sanitizer_flags.inc"
#undef COMMON_FLAG
  // Do a sanity check for certain flags.
  if (malloc_context_size < 1)
    malloc_context_size = 1;
}

void CommonFlags::CopyFrom(const CommonFlags &other) {
  internal_memcpy(this, &other, sizeof(*this));
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
    const char *name_end = pos + internal_strlen(name);
    if ((pos != env &&
         ((pos[-1] >= 'a' && pos[-1] <= 'z') || pos[-1] == '_')) ||
        *name_end != '=') {
      // Seems to be middle of another flag name or value.
      env = pos + 1;
      continue;
    }
    pos = name_end;
    break;
  }
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
      end = pos + internal_strcspn(pos, " :\r\n\t");
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
