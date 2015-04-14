//===- lld/ReaderWriter/ELF/Hexagon/HexagonRelocationHandler.h -----------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_HEXAGON_HEXAGON_RELOCATION_HANDLER_H
#define LLD_READER_WRITER_ELF_HEXAGON_HEXAGON_RELOCATION_HANDLER_H

#include "lld/ReaderWriter/ELFLinkingContext.h"

namespace lld {
namespace elf {
class HexagonTargetHandler;
class HexagonTargetLayout;

class HexagonTargetRelocationHandler final : public TargetRelocationHandler {
public:
  HexagonTargetRelocationHandler(HexagonTargetLayout &layout)
      : _targetLayout(layout) {}

  std::error_code applyRelocation(ELFWriter &, llvm::FileOutputBuffer &,
                                  const AtomLayout &,
                                  const Reference &) const override;

private:
  HexagonTargetLayout &_targetLayout;
};
} // elf
} // lld
#endif
