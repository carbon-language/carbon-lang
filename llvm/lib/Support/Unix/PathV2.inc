//===- llvm/Support/Unix/PathV2.cpp - Unix Path Implementation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Unix specific implementation of the PathV2 API.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

#include "Unix.h"

namespace llvm {
namespace sys  {
namespace path {

error_code current_path(SmallVectorImpl<char> &result) {
  long size = ::pathconf(".", _PC_PATH_MAX);
  result.reserve(size + 1);
  result.set_size(size + 1);

  if (::getcwd(result.data(), result.size()) == 0)
    return error_code(errno, system_category());

  result.set_size(strlen(result.data()));
  return make_error_code(errc::success);
}

} // end namespace path
} // end namespace sys
} // end namespace llvm
