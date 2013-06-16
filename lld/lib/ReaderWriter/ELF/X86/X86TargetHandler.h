//===- lib/ReaderWriter/ELF/X86/X86TargetHandler.h ------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_X86_TARGET_HANDLER_H
#define LLD_READER_WRITER_ELF_X86_TARGET_HANDLER_H

#include "DefaultTargetHandler.h"
#include "TargetLayout.h"

namespace lld {
namespace elf {
typedef llvm::object::ELFType<llvm::support::little, 4, false> X86ELFType;
class X86TargetInfo;

class X86TargetRelocationHandler LLVM_FINAL
    : public TargetRelocationHandler<X86ELFType> {
public:
  X86TargetRelocationHandler(const X86TargetInfo &ti) : _targetInfo(ti) {}

  virtual ErrorOr<void> applyRelocation(ELFWriter &, llvm::FileOutputBuffer &,
                                        const lld::AtomLayout &,
                                        const Reference &)const;

private:
  const X86TargetInfo &_targetInfo;
};

class X86TargetHandler LLVM_FINAL
    : public DefaultTargetHandler<X86ELFType> {
public:
  X86TargetHandler(X86TargetInfo &targetInfo);

  virtual TargetLayout<X86ELFType> &targetLayout() {
    return _targetLayout;
  }

  virtual const X86TargetRelocationHandler &getRelocationHandler() const {
    return _relocationHandler;
  }

private:
  X86TargetRelocationHandler _relocationHandler;
  TargetLayout<X86ELFType> _targetLayout;
};
} // end namespace elf
} // end namespace lld

#endif
