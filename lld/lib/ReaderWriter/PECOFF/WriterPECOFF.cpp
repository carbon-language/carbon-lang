//===- lib/ReaderWriter/PECOFF/WriterPECOFF.cpp ---------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/Writer.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"

namespace lld {
namespace pecoff {

class ExecutableWriter : public Writer {
 public:
  ExecutableWriter(const TargetInfo &) {}

  virtual error_code writeFile(const File &linkedFile, StringRef path) {
    // TODO: implement this
    return error_code::success();
  }
};

} // end namespace pecoff

std::unique_ptr<Writer> createWriterPECOFF(const TargetInfo &info) {
  return std::unique_ptr<Writer>(new pecoff::ExecutableWriter(info));
}

} // end namespace lld
