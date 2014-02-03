//===- lib/ReaderWriter/ELF/Mips/MipsELFWriters.h -------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_MIPS_ELF_WRITERS_H
#define LLD_READER_WRITER_ELF_MIPS_MIPS_ELF_WRITERS_H

#include "MipsLinkingContext.h"
#include "OutputELFWriter.h"

namespace lld {
namespace elf {

template <class ELFT> class MipsTargetLayout;

template <typename ELFT> class MipsELFWriter {
public:
  MipsELFWriter(MipsLinkingContext &context,
                MipsTargetLayout<ELFT> &targetLayout)
      : _mipsLinkingContext(context), _mipsTargetLayout(targetLayout) {}

protected:
  bool setELFHeader(ELFHeader<ELFT> &elfHeader) {
    elfHeader.e_version(1);
    elfHeader.e_ident(llvm::ELF::EI_VERSION, llvm::ELF::EV_CURRENT);
    elfHeader.e_ident(llvm::ELF::EI_OSABI, llvm::ELF::ELFOSABI_NONE);
    if (_mipsTargetLayout.findOutputSection(".got.plt"))
      elfHeader.e_ident(llvm::ELF::EI_ABIVERSION, 1);
    else
      elfHeader.e_ident(llvm::ELF::EI_ABIVERSION, 0);

    // FIXME (simon): Read elf flags from all inputs, check compatibility,
    // merge them and write result here.
    uint32_t flags = llvm::ELF::EF_MIPS_NOREORDER | llvm::ELF::EF_MIPS_ABI_O32 |
                     llvm::ELF::EF_MIPS_CPIC | llvm::ELF::EF_MIPS_ARCH_32R2;
    if (_mipsLinkingContext.getOutputELFType() == llvm::ELF::ET_DYN)
      flags |= EF_MIPS_PIC;
    elfHeader.e_flags(flags);
    return true;
  }

  void finalizeMipsRuntimeAtomValues() {
    if (_mipsLinkingContext.isDynamic()) {
      auto gotSection = _mipsTargetLayout.findOutputSection(".got");
      auto got = gotSection ? gotSection->virtualAddr() : 0;
      auto gp = gotSection ? got + _mipsTargetLayout.getGPOffset() : 0;

      auto gotAtomIter =
          _mipsTargetLayout.findAbsoluteAtom("_GLOBAL_OFFSET_TABLE_");
      assert(gotAtomIter != _mipsTargetLayout.absoluteAtoms().end());
      (*gotAtomIter)->_virtualAddr = got;

      auto gpAtomIter = _mipsTargetLayout.findAbsoluteAtom("_gp");
      assert(gpAtomIter != _mipsTargetLayout.absoluteAtoms().end());
      (*gpAtomIter)->_virtualAddr = gp;

      AtomLayout *gpAtom = _mipsTargetLayout.getGP();
      assert(gpAtom != nullptr);
      gpAtom->_virtualAddr = gp;
    }
  }

private:
  MipsLinkingContext &_mipsLinkingContext LLVM_ATTRIBUTE_UNUSED;
  MipsTargetLayout<ELFT> &_mipsTargetLayout;
};

} // elf
} // lld

#endif
