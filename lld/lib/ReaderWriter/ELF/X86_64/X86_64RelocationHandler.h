//===- lib/ReaderWriter/ELF/X86_64/X86_64RelocationHandler.h
//------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef X86_64_RELOCATION_HANDLER_H
#define X86_64_RELOCATION_HANDLER_H

#include "X86_64TargetHandler.h"

namespace lld {
namespace elf {
typedef llvm::object::ELFType<llvm::support::little, 8, true> X86_64ELFType;
class X86_64TargetInfo;

class X86_64TargetRelocationHandler LLVM_FINAL
    : public TargetRelocationHandler<X86_64ELFType> {
public:
  X86_64TargetRelocationHandler(const X86_64TargetInfo &ti)
      : _tlsSize(0), _targetInfo(ti) {}

  virtual ErrorOr<void> applyRelocation(ELFWriter &, llvm::FileOutputBuffer &,
                                        const lld::AtomLayout &,
                                        const Reference &) const;

  virtual int64_t relocAddend(const Reference &) const;

private:
  // Cached size of the TLS segment.
  mutable uint64_t _tlsSize;
  const X86_64TargetInfo &_targetInfo;
};

} // end namespace elf
} // end namespace lld

#endif
