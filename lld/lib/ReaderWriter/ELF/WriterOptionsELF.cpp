//===- lib/ReaderWriter/ELF/WriterOptionsELF.cpp ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/WriterELF.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/system_error.h"


namespace lld {

StringRef WriterOptionsELF::entryPoint() const {
  if (_type == llvm::ELF::ET_EXEC)
    return _entryPoint;
  return StringRef();
}

} // namespace lld
