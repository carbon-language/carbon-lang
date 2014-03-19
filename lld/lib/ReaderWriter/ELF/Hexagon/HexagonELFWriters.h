//===- lib/ReaderWriter/ELF/Hexagon/HexagonELFWriters.h -------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_ELF_WRITERS_H
#define HEXAGON_ELF_WRITERS_H

#include "HexagonLinkingContext.h"
#include "OutputELFWriter.h"

namespace lld {
namespace elf {

template <class ELFT> class HexagonTargetLayout;

template <typename ELFT> class HexagonELFWriter {
public:
  HexagonELFWriter(HexagonLinkingContext &context,
                   HexagonTargetLayout<ELFT> &targetLayout)
      : _hexagonLinkingContext(context), _hexagonTargetLayout(targetLayout) {}

protected:
  bool setELFHeader(ELFHeader<ELFT> &elfHeader) {
    elfHeader.e_ident(llvm::ELF::EI_VERSION, 1);
    elfHeader.e_ident(llvm::ELF::EI_OSABI, 0);
    elfHeader.e_version(1);
    elfHeader.e_flags(0x3);
    return true;
  }

  void finalizeHexagonRuntimeAtomValues() {
    if (_hexagonLinkingContext.isDynamic()) {
      auto gotAtomIter =
          _hexagonTargetLayout.findAbsoluteAtom("_GLOBAL_OFFSET_TABLE_");
      auto gotpltSection = _hexagonTargetLayout.findOutputSection(".got.plt");
      if (gotpltSection)
        (*gotAtomIter)->_virtualAddr = gotpltSection->virtualAddr();
      else
        (*gotAtomIter)->_virtualAddr = 0;
      auto dynamicAtomIter = _hexagonTargetLayout.findAbsoluteAtom("_DYNAMIC");
      auto dynamicSection = _hexagonTargetLayout.findOutputSection(".dynamic");
      if (dynamicSection)
        (*dynamicAtomIter)->_virtualAddr = dynamicSection->virtualAddr();
      else
        (*dynamicAtomIter)->_virtualAddr = 0;
    }
  }

private:
  HexagonLinkingContext &_hexagonLinkingContext;
  HexagonTargetLayout<ELFT> &_hexagonTargetLayout;
};

} // elf
} // lld
#endif // HEXAGON_ELF_WRITERS_H
