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


namespace lld {
std::unique_ptr<Writer> createWriterPECOFF(const TargetInfo &) {
  llvm_unreachable("PE/COFF support not implemented yet");
  return nullptr;
}
} // end namespace lld
