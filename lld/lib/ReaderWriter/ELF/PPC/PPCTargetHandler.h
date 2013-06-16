//===- lib/ReaderWriter/ELF/PPC/PPCTargetHandler.h ------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_PPC_TARGET_HANDLER_H
#define LLD_READER_WRITER_ELF_PPC_TARGET_HANDLER_H

#include "DefaultTargetHandler.h"
#include "TargetLayout.h"

namespace lld {
namespace elf {
typedef llvm::object::ELFType<llvm::support::big, 4, false> PPCELFType;
class PPCTargetInfo;

class PPCTargetRelocationHandler LLVM_FINAL
    : public TargetRelocationHandler<PPCELFType> {
public:
  PPCTargetRelocationHandler(const PPCTargetInfo &ti) : _targetInfo(ti) {}

  virtual ErrorOr<void> applyRelocation(ELFWriter &, llvm::FileOutputBuffer &,
                                        const lld::AtomLayout &,
                                        const Reference &)const;

private:
  const PPCTargetInfo &_targetInfo;
};

class PPCTargetHandler LLVM_FINAL
    : public DefaultTargetHandler<PPCELFType> {
public:
  PPCTargetHandler(PPCTargetInfo &targetInfo);

  virtual TargetLayout<PPCELFType> &targetLayout() {
    return _targetLayout;
  }

  virtual const PPCTargetRelocationHandler &getRelocationHandler() const {
    return _relocationHandler;
  }

private:
  PPCTargetRelocationHandler _relocationHandler;
  TargetLayout<PPCELFType> _targetLayout;
};
} // end namespace elf
} // end namespace lld

#endif
