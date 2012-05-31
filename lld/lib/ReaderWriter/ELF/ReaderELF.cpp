//===- lib/ReaderWriter/ELF/ReaderELF.cpp --------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/ReaderELF.h"
#include "lld/Core/File.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"

#include <map>
#include <vector>


namespace lld {

ReaderOptionsELF::ReaderOptionsELF() {
}

ReaderOptionsELF::~ReaderOptionsELF() {
}



Reader* createReaderELF(const ReaderOptionsELF &options) {
  assert(0 && "ELF Reader not yet implemented");
  return nullptr;
}


} // namespace

