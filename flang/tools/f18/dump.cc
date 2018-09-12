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

// This file defines Dump routines available for calling from the debugger.
// Each is based on operator<< for that type. There are overloadings for
// reference and pointer, and for dumping to a provided ostream or cerr.

#ifdef DEBUG

#include <iostream>

#define DEFINE_DUMP(ns, name) \
  namespace ns { \
  class name; \
  std::ostream &operator<<(std::ostream &, const name &); \
  } \
  void Dump(std::ostream &os, const ns::name &x) { os << x << '\n'; } \
  void Dump(std::ostream &os, const ns::name *x) { \
    if (x == nullptr) \
      os << "null\n"; \
    else \
      Dump(os, *x); \
  } \
  void Dump(const ns::name &x) { Dump(std::cerr, x); } \
  void Dump(const ns::name *x) { Dump(std::cerr, *x); }

namespace Fortran {
DEFINE_DUMP(parser, Name)
DEFINE_DUMP(parser, CharBlock)
DEFINE_DUMP(semantics, Symbol)
DEFINE_DUMP(semantics, Scope)
DEFINE_DUMP(semantics, IntrinsicTypeSpec)
DEFINE_DUMP(semantics, DerivedTypeSpec)
DEFINE_DUMP(semantics, DeclTypeSpec)
}  // namespace Fortran

#endif
