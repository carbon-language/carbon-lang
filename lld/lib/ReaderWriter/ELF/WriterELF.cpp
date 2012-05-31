//===- lib/ReaderWriter/ELF/WriterELF.cpp ---------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/WriterELF.h"

#include "llvm/Support/Debug.h"


namespace lld {
namespace elf {

// define ELF writer class here


} // namespace elf

Writer* createWriterELF(const WriterOptionsELF &options) {
  assert(0 && "ELF support not implemented yet");
  return nullptr;
}

WriterOptionsELF::WriterOptionsELF() {
}

WriterOptionsELF::~WriterOptionsELF() {
}

} // namespace lld

