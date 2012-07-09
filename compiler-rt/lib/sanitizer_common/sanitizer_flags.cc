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

static char *GetFlagValue(const char *env, const char *name) {
  if (env == 0)
    return 0;
  const char *pos = internal_strstr(env, name);
  const char *end;
  if (pos == 0)
    return 0;
  pos += internal_strlen(name);
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
      end = internal_strchr(pos, ' ');
    }
    if (end == 0)
      end = pos + internal_strlen(pos);
  }
  int len = end - pos;
  char *f = (char*)InternalAlloc(len + 1);
  internal_memcpy(f, pos, len);
  f[len] = '\0';
  return f;
}

void ParseFlag(const char *env, bool *flag, const char *name) {
  char *val = GetFlagValue(env, name);
  if (val == 0)
    return;
  if (0 == internal_strcmp(val, "0") ||
      0 == internal_strcmp(val, "no") ||
      0 == internal_strcmp(val, "false"))
    *flag = false;
  if (0 == internal_strcmp(val, "1") ||
      0 == internal_strcmp(val, "yes") ||
      0 == internal_strcmp(val, "true"))
    *flag = true;
  InternalFree(val);
}

void ParseFlag(const char *env, int *flag, const char *name) {
  char *val = GetFlagValue(env, name);
  if (val == 0)
    return;
  *flag = internal_atoll(val);
  InternalFree(val);
}

void ParseFlag(const char *env, const char **flag, const char *name) {
  const char *val = GetFlagValue(env, name);
  if (val == 0)
    return;
  *flag = val;
}

}  // namespace __sanitizer
