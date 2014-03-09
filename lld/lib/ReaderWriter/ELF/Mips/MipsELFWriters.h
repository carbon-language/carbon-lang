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
      : _context(context), _targetLayout(targetLayout) {}

  void setELFHeader(ELFHeader<ELFT> &elfHeader) {
    elfHeader.e_version(1);
    elfHeader.e_ident(llvm::ELF::EI_VERSION, llvm::ELF::EV_CURRENT);
    elfHeader.e_ident(llvm::ELF::EI_OSABI, llvm::ELF::ELFOSABI_NONE);
    if (_targetLayout.findOutputSection(".got.plt"))
      elfHeader.e_ident(llvm::ELF::EI_ABIVERSION, 1);
    else
      elfHeader.e_ident(llvm::ELF::EI_ABIVERSION, 0);

    // FIXME (simon): Read elf flags from all inputs, check compatibility,
    // merge them and write result here.
    uint32_t flags = llvm::ELF::EF_MIPS_NOREORDER | llvm::ELF::EF_MIPS_ABI_O32 |
                     llvm::ELF::EF_MIPS_CPIC | llvm::ELF::EF_MIPS_ARCH_32R2;
    if (_context.getOutputELFType() == llvm::ELF::ET_DYN)
      flags |= EF_MIPS_PIC;
    elfHeader.e_flags(flags);
  }

  void finalizeMipsRuntimeAtomValues() {
    if (!_context.isDynamic())
      return;

    auto gotSection = _targetLayout.findOutputSection(".got");
    auto got = gotSection ? gotSection->virtualAddr() : 0;
    auto gp = gotSection ? got + _targetLayout.getGPOffset() : 0;

    auto gotAtomIter = _targetLayout.findAbsoluteAtom("_GLOBAL_OFFSET_TABLE_");
    assert(gotAtomIter != _targetLayout.absoluteAtoms().end());
    (*gotAtomIter)->_virtualAddr = got;

    auto gpAtomIter = _targetLayout.findAbsoluteAtom("_gp");
    assert(gpAtomIter != _targetLayout.absoluteAtoms().end());
    (*gpAtomIter)->_virtualAddr = gp;

    AtomLayout *gpAtom = _targetLayout.getGP();
    assert(gpAtom != nullptr);
    gpAtom->_virtualAddr = gp;
  }

  bool hasGlobalGOTEntry(const Atom *a) const {
    return _targetLayout.getGOTSection().hasGlobalGOTEntry(a);
  }

private:
  MipsLinkingContext &_context;
  MipsTargetLayout<ELFT> &_targetLayout;
};

} // elf
} // lld

#endif
