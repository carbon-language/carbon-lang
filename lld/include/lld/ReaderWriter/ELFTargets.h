//===- lld/ReaderWriter/ELFTargets.h --------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_TARGETS_H
#define LLD_READER_WRITER_ELF_TARGETS_H

#include "ELFLinkingContext.h"

namespace lld {
namespace elf {

std::unique_ptr<ELFLinkingContext> createAArch64LinkingContext(llvm::Triple);
std::unique_ptr<ELFLinkingContext> createARMLinkingContext(llvm::Triple);
std::unique_ptr<ELFLinkingContext> createExampleLinkingContext(llvm::Triple);
std::unique_ptr<ELFLinkingContext> createHexagonLinkingContext(llvm::Triple);
std::unique_ptr<ELFLinkingContext> createMipsLinkingContext(llvm::Triple);
std::unique_ptr<ELFLinkingContext> createX86LinkingContext(llvm::Triple);
std::unique_ptr<ELFLinkingContext> createX86_64LinkingContext(llvm::Triple);

} // end namespace elf
} // end namespace lld

#endif
