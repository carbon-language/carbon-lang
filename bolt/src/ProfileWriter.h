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
#include "RewriteInstance.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/raw_ostream.h"
#include <system_error>

namespace llvm {
namespace bolt {

class ProfileWriter {
  ProfileWriter() = delete;

  std::string FileName;

  std::unique_ptr<raw_fd_ostream> OS;

public:
  explicit ProfileWriter(const std::string &FileName)
    : FileName(FileName) {
  }

  /// Save execution profile for that instance.
  std::error_code writeProfile(const RewriteInstance &RI);
};

} // namespace bolt
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_BOLT_PROFILE_WRITER_H
