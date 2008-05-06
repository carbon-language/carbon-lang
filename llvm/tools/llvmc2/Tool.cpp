//===--- Tools.cpp - The LLVM Compiler Driver -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Tool abstract base class - implementation of the auxiliary functions.
//
//===----------------------------------------------------------------------===//

#include "Tool.h"

#include "llvm/ADT/StringExtras.h"

void llvmcc::Tool::UnpackValues (const std::string& from,
                                 std::vector<std::string>& to) {
  llvm::SplitString(from, to, ",");
}
