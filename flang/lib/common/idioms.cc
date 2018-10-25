// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "idioms.h"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace Fortran::common {

[[noreturn]] void die(const char *msg, ...) {
  va_list ap;
  va_start(ap, msg);
  std::fputs("\nfatal internal error: ", stderr);
  std::vfprintf(stderr, msg, ap);
  va_end(ap);
  fputc('\n', stderr);
  std::abort();
}

// Convert the int index of an enumerator to a string.
// enumNames is a list of the names, separated by commas with optional spaces.
// This is intended for use from the expansion of ENUM_CLASS.
std::string EnumIndexToString(int index, const char *enumNames) {
  const char *p{enumNames};
  for (; index > 0; --index, ++p) {
    for (; *p && *p != ','; ++p) {
    }
  }
  for (; *p == ' '; ++p) {
  }
  CHECK(*p != '\0');
  const char *q = p;
  for (; *q && *q != ' ' && *q != ','; ++q) {
  }
  return std::string(p, q - p);
}
}
