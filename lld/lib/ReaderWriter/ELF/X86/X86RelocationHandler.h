//===- lib/ReaderWriter/ELF/X86/X86RelocationHandler.h --------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef X86_X86_RELOCATION_HANDLER_H
#define X86_X86_RELOCATION_HANDLER_H

#include "lld/ReaderWriter/ELFLinkingContext.h"

namespace lld {
namespace elf {

class X86TargetRelocationHandler final : public TargetRelocationHandler {
public:
  std::error_code applyRelocation(ELFWriter &, llvm::FileOutputBuffer &,
                                  const AtomLayout &,
                                  const Reference &) const override;
};

} // end namespace elf
} // end namespace lld

#endif // X86_X86_RELOCATION_HANDLER_H
