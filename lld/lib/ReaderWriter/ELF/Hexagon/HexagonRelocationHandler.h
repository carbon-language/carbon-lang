//===- lld/ReaderWriter/ELF/Hexagon/HexagonRelocationHandler.h -----------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_HEXAGON_RELOCATION_HANDLER_H
#define LLD_READER_WRITER_ELF_HEXAGON_RELOCATION_HANDLER_H

#include "HexagonTargetHandler.h"
#include "lld/ReaderWriter/RelocationHelperFunctions.h"

namespace lld {
namespace elf {
typedef llvm::object::ELFType<llvm::support::little, 4, false> HexagonELFType;

class HexagonTargetInfo;

class HexagonTargetRelocationHandler LLVM_FINAL
    : public TargetRelocationHandler<HexagonELFType> {
public:
  HexagonTargetRelocationHandler(const HexagonTargetInfo &ti) : _targetInfo(ti) 
  {}

  virtual ErrorOr<void> applyRelocation(ELFWriter &, llvm::FileOutputBuffer &,
                                        const AtomLayout &,
                                        const Reference &)const;

private:
  const HexagonTargetInfo &_targetInfo;
};
} // elf
} // lld 
#endif
