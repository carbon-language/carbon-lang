//===-- ProfileWriter.cpp - serialize profiling data ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_TOOLS_LLVM_BOLT_PROFILE_WRITER_H
#define LLVM_TOOLS_LLVM_BOLT_PROFILE_WRITER_H

#include "BinaryBasicBlock.h"
#include "BinaryContext.h"
#include "BinaryFunction.h"
#include "ProfileYAMLMapping.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/raw_ostream.h"
#include <system_error>

namespace llvm {
namespace bolt {

class ProfileWriter {
  ProfileWriter() = delete;

  std::string FileName;

  std::error_code write(BinaryFunction &BF);

  std::unique_ptr<raw_fd_ostream> OS;

  void printBinaryFunctionProfile(const BinaryFunction &BF);

  void printBinaryFunctionsProfile(std::map<uint64_t, BinaryFunction> &BFs);

public:
  explicit ProfileWriter(const std::string &FileName)
    : FileName(FileName) {
  }

  /// Write profile for functions.
  std::error_code writeProfile(std::map<uint64_t, BinaryFunction> &Functions);
};

} // namespace bolt
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_BOLT_PROFILE_WRITER_H
