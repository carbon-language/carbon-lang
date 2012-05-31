//===- lib/ReaderWriter/PECOFF/WriterPECOFF.cpp ---------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/WriterPECOFF.h"

#include "llvm/Support/Debug.h"


namespace lld {
namespace pe_coff {

// define PE/COFF writer class here


} // namespace pe_coff

Writer* createWriterPECOFF(const WriterOptionsPECOFF &options) {
  assert(0 && "PE/COFF support not implemented yet");
  return nullptr;
}

WriterOptionsPECOFF::WriterOptionsPECOFF() {
}

WriterOptionsPECOFF::~WriterOptionsPECOFF() {
}

} // namespace lld

