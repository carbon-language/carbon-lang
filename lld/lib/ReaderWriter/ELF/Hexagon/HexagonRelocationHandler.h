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

#include "HexagonSectionChunks.h"
#include "HexagonTargetHandler.h"
#include "lld/ReaderWriter/RelocationHelperFunctions.h"

namespace lld {
namespace elf {

class HexagonTargetHandler;

class HexagonTargetRelocationHandler final :
    public TargetRelocationHandler<HexagonELFType> {
public:
  HexagonTargetRelocationHandler(HexagonTargetLayout<HexagonELFType> &layout)
      : _hexagonTargetLayout(layout) {}

  std::error_code applyRelocation(ELFWriter &, llvm::FileOutputBuffer &,
                                  const lld::AtomLayout &,
                                  const Reference &) const override;

private:
  HexagonTargetLayout<HexagonELFType> &_hexagonTargetLayout;
};
} // elf
} // lld
#endif
