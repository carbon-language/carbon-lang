//===- Reader.h - Create object file readers ------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_READER_H_
#define LLD_READER_READER_H_

#include "llvm/ADT/OwningPtr.h"

#include <memory>

namespace llvm {
  class error_code;
  class MemoryBuffer;
}

namespace lld {
  class File;

  llvm::error_code parseCOFFObjectFile(std::unique_ptr<llvm::MemoryBuffer> MB,
                                       std::unique_ptr<File> &Result);
}

#endif
