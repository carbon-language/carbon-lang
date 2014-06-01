//===- lib/ReaderWriter/ELF/Mips/MipsCtorsOrderPass.h ---------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_MIPS_MIPS_CTORS_ORDER_PASS_H
#define LLD_READER_WRITER_ELF_MIPS_MIPS_CTORS_ORDER_PASS_H

#include "lld/Core/Pass.h"

namespace lld {
namespace elf {
/// \brief This pass sorts atoms in .{ctors,dtors}.<priority> sections.
class MipsCtorsOrderPass : public Pass {
public:
  void perform(std::unique_ptr<MutableFile> &mergedFile) override;
};
}
}

#endif
