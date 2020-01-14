//===-- lib/parser/char-block.cc --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//----------------------------------------------------------------------------//

#include "char-block.h"
#include <ostream>

namespace Fortran::parser {

std::ostream &operator<<(std::ostream &os, const CharBlock &x) {
  return os << x.ToString();
}

}
