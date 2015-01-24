//===- lib/ReaderWriter/ELF/PPC/PPCLinkingContext.h -----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_PPC_PPC_LINKING_CONTEXT_H
#define LLD_READER_WRITER_ELF_PPC_PPC_LINKING_CONTEXT_H

#include "lld/ReaderWriter/ELFLinkingContext.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/ELF.h"

namespace lld {
namespace elf {

class PPCLinkingContext final : public ELFLinkingContext {
public:
  static std::unique_ptr<ELFLinkingContext> create(llvm::Triple);
  PPCLinkingContext(llvm::Triple triple);

  /// \brief PPC has no relative relocations defined
  bool isRelativeReloc(const Reference &) const override { return false; }
};

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_PPC_PPC_LINKING_CONTEXT_H
