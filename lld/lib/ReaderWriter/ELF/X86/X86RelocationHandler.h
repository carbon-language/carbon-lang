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

#include "X86TargetHandler.h"

namespace lld {
namespace elf {
template <class ELFT> class X86TargetLayout;
typedef llvm::object::ELFType<llvm::support::little, 2, false> X86ELFType;

class X86TargetRelocationHandler final
    : public TargetRelocationHandler<X86ELFType> {
public:
  using TargetRelocationHandler<X86ELFType>::TargetRelocationHandler;

  std::error_code applyRelocation(ELFWriter &, llvm::FileOutputBuffer &,
                                  const lld::AtomLayout &,
                                  const Reference &) const override;
};

} // end namespace elf
} // end namespace lld

#endif // X86_X86_RELOCATION_HANDLER_H
