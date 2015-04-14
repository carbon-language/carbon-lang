//===- lib/ReaderWriter/ELF/X86_64/X86_64RelocationHandler.h --------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef X86_64_RELOCATION_HANDLER_H
#define X86_64_RELOCATION_HANDLER_H

#include "lld/ReaderWriter/ELFLinkingContext.h"

namespace lld {
namespace elf {
class X86_64TargetLayout;

class X86_64TargetRelocationHandler final : public TargetRelocationHandler {
public:
  X86_64TargetRelocationHandler(X86_64TargetLayout &layout)
      : _tlsSize(0), _layout(layout) {}

  std::error_code applyRelocation(ELFWriter &, llvm::FileOutputBuffer &,
                                  const AtomLayout &,
                                  const Reference &) const override;

private:
  // Cached size of the TLS segment.
  mutable uint64_t _tlsSize;
  X86_64TargetLayout &_layout;
};

} // end namespace elf
} // end namespace lld

#endif // X86_64_RELOCATION_HANDLER_H
