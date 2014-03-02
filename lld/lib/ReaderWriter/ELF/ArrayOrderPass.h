//===- lib/ReaderWriter/ELF/ArrayOrderPass.h ------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_ARRAY_ORDER_PASS_H
#define LLD_READER_WRITER_ELF_ARRAY_ORDER_PASS_H

#include "lld/Core/Pass.h"

namespace lld {
namespace elf {
/// \brief This pass sorts atoms in .{init,fini}_array.<priority> sections.
class ArrayOrderPass : public Pass {
public:
  ArrayOrderPass() : Pass() {}
  virtual void perform(std::unique_ptr<MutableFile> &mergedFile) override;
};
}
}

#endif
