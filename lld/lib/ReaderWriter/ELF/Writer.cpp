//===- lib/ReaderWriter/ELF/WriterELF.cpp ---------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/Writer.h"
#include "DynamicLibraryWriter.h"
#include "ExecutableWriter.h"

using namespace llvm;
using namespace llvm::object;

namespace lld {

std::unique_ptr<Writer> createWriterELF(TargetHandlerBase *handler) {
  return std::move(handler->getWriter());
}

} // namespace lld
