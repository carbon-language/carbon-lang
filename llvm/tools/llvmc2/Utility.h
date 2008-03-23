//===--- Utility.h - The LLVM Compiler Driver -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Various helper and utility functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMCC_UTILITY_H
#define LLVM_TOOLS_LLVMCC_UTILITY_H

#include <string>
#include <vector>

namespace llvmcc {

  int ExecuteProgram (const std::string& name,
                      const std::vector<std::string>& arguments);

}

#endif // LLVM_TOOLS_LLVMCC_UTILITY_H
