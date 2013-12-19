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

#include "HexagonSectionChunks.h"
#include "HexagonTargetHandler.h"
#include "lld/ReaderWriter/RelocationHelperFunctions.h"

namespace lld {
namespace elf {

class HexagonLinkingContext;
class HexagonTargetHandler;
template <class HexagonELFType> class HexagonTargetLayout;

class HexagonTargetRelocationHandler LLVM_FINAL :
    public TargetRelocationHandler<HexagonELFType> {
public:
  HexagonTargetRelocationHandler(
      const HexagonLinkingContext &context, const HexagonTargetHandler &tH,
      const HexagonTargetLayout<HexagonELFType> &layout)
      : _context(context), _targetHandler(tH), _targetLayout(layout) {}

  virtual error_code
  applyRelocation(ELFWriter &, llvm::FileOutputBuffer &,
                  const lld::AtomLayout &, const Reference &) const;
                  
 private:
  const HexagonLinkingContext &_context;
  const HexagonTargetHandler &_targetHandler;
  const HexagonTargetLayout<HexagonELFType> &_targetLayout;
};
} // elf
} // lld
#endif
