//===- lib/ReaderWriter/ELF/Hexagon/HexagonTargetHandler.h ----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_HEXAGON_TARGET_HANDLER_H
#define LLD_READER_WRITER_ELF_HEXAGON_TARGET_HANDLER_H

#include "DefaultTargetHandler.h"
#include "TargetLayout.h"

namespace lld {
namespace elf {
typedef llvm::object::ELFType<llvm::support::little, 4, false> HexagonELFType;
class HexagonTargetInfo;

class HexagonTargetRelocationHandler LLVM_FINAL
    : public TargetRelocationHandler<HexagonELFType> {
public:
  HexagonTargetRelocationHandler(const HexagonTargetInfo &ti) : _targetInfo(ti) {}

  virtual ErrorOr<void> applyRelocation(ELFWriter &, llvm::FileOutputBuffer &,
                                        const AtomLayout &,
                                        const Reference &)const;

private:
  const HexagonTargetInfo &_targetInfo;
};

class HexagonTargetHandler LLVM_FINAL
    : public DefaultTargetHandler<HexagonELFType> {
public:
  HexagonTargetHandler(HexagonTargetInfo &targetInfo);

  virtual TargetLayout<HexagonELFType> &targetLayout() {
    return _targetLayout;
  }

  virtual const HexagonTargetRelocationHandler &getRelocationHandler() const {
    return _relocationHandler;
  }

private:
  HexagonTargetRelocationHandler _relocationHandler;
  TargetLayout<HexagonELFType> _targetLayout;
};
} // end namespace elf
} // end namespace lld

#endif
