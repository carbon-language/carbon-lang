//===--------- lib/ReaderWriter/ELF/ARM/ARMRelocationHandler.h ------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_ARM_ARM_RELOCATION_HANDLER_H
#define LLD_READER_WRITER_ELF_ARM_ARM_RELOCATION_HANDLER_H

#include "ARMTargetHandler.h"

namespace lld {
namespace elf {
typedef llvm::object::ELFType<llvm::support::little, 2, false> ARMELFType;

template <class ELFT> class ARMTargetLayout;

class ARMTargetRelocationHandler final
    : public TargetRelocationHandler {
public:
  std::error_code applyRelocation(ELFWriter &, llvm::FileOutputBuffer &,
                                  const lld::AtomLayout &,
                                  const Reference &) const override;
};

} // end namespace elf
} // end namespace lld

#endif // LLD_READER_WRITER_ELF_ARM_ARM_RELOCATION_HANDLER_H
