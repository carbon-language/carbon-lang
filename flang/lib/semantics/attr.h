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

#ifndef FORTRAN_SEMANTICS_ATTR_H_
#define FORTRAN_SEMANTICS_ATTR_H_

#include "../common/enum-set.h"
#include "../common/idioms.h"
#include <cinttypes>
#include <string>

namespace Fortran::semantics {

// All available attributes.
ENUM_CLASS(Attr, ABSTRACT, ALLOCATABLE, ASYNCHRONOUS, BIND_C, CONTIGUOUS,
    DEFERRED, ELEMENTAL, EXTERNAL, IMPURE, INTENT_IN, INTENT_INOUT, INTENT_OUT,
    INTRINSIC, MODULE, NON_OVERRIDABLE, NON_RECURSIVE, NOPASS, OPTIONAL,
    PARAMETER, PASS, POINTER, PRIVATE, PROTECTED, PUBLIC, PURE, RECURSIVE, SAVE,
    TARGET, VALUE, VOLATILE)

// Set of attributes
class Attrs : public common::EnumSet<Attr, Attr_enumSize> {
private:
  using enumSetType = common::EnumSet<Attr, Attr_enumSize>;

public:
  using enumSetType::enumSetType;
  constexpr bool HasAny(const Attrs &x) const { return !(*this & x).none(); }
  constexpr bool HasAll(const Attrs &x) const { return (~*this & x).none(); }
  // Internal error if any of these attributes are not in allowed.
  void CheckValid(const Attrs &allowed) const;

private:
  friend std::ostream &operator<<(std::ostream &, const Attrs &);
};

// Return string representation of attr that matches Fortran source.
std::string AttrToString(Attr attr);

std::ostream &operator<<(std::ostream &o, Attr attr);
std::ostream &operator<<(std::ostream &o, const Attrs &attrs);
}
#endif  // FORTRAN_SEMANTICS_ATTR_H_
