//===- lib/ReaderWriter/Writer.cpp ----------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/File.h"
#include "lld/ReaderWriter/Writer.h"

namespace lld {
Writer::Writer() {
}

Writer::~Writer() {
}

bool Writer::createImplicitFiles(std::vector<std::unique_ptr<File> > &) {
  return true;
}
} // end namespace lld
