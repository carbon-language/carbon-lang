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

#include "AArch64TargetHandler.h"

namespace lld {
namespace elf {
typedef llvm::object::ELFType<llvm::support::little, 2, true> AArch64ELFType;

template <class ELFT> class AArch64TargetLayout;

class AArch64TargetRelocationHandler final
    : public TargetRelocationHandler<AArch64ELFType> {
public:
  AArch64TargetRelocationHandler(AArch64TargetLayout<AArch64ELFType> &layout)
      : _tlsSize(0), _AArch64Layout(layout) {}

  std::error_code applyRelocation(ELFWriter &, llvm::FileOutputBuffer &,
                                  const lld::AtomLayout &,
                                  const Reference &) const override;

  virtual int64_t relocAddend(const Reference &) const;

  static const Registry::KindStrings kindStrings[];

private:
  // Cached size of the TLS segment.
  mutable uint64_t _tlsSize;
  AArch64TargetLayout<AArch64ELFType> &_AArch64Layout;
};

} // end namespace elf
} // end namespace lld

#endif // AArch64_RELOCATION_HANDLER_H
