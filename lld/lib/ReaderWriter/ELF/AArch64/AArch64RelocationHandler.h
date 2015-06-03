//===- lib/ReaderWriter/ELF/AArch64/AArch64RelocationHandler.h ------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef AARCH64_RELOCATION_HANDLER_H
#define AARCH64_RELOCATION_HANDLER_H

#include "lld/ReaderWriter/ELFLinkingContext.h"

namespace lld {
namespace elf {

class AArch64TargetLayout;

class AArch64TargetRelocationHandler final : public TargetRelocationHandler {
public:
  AArch64TargetRelocationHandler(AArch64TargetLayout &layout)
      : _layout(layout) {}

  std::error_code applyRelocation(ELFWriter &, llvm::FileOutputBuffer &,
                                  const AtomLayout &,
                                  const Reference &) const override;

private:
  AArch64TargetLayout &_layout;
};

} // end namespace elf
} // end namespace lld

#endif // AArch64_RELOCATION_HANDLER_H
