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

namespace lld {
namespace elf {

template <class ELFT> class MipsTargetLayout;

class MipsDynamicAtom : public DynamicAtom {
public:
  MipsDynamicAtom(const File &f) : DynamicAtom(f) {}

  ContentPermissions permissions() const override { return permR__; }
};

template <typename ELFT> class MipsELFWriter {
public:
  MipsELFWriter(MipsLinkingContext &ctx, MipsTargetLayout<ELFT> &targetLayout)
      : _ctx(ctx), _targetLayout(targetLayout) {}

  void setELFHeader(ELFHeader<ELFT> &elfHeader) {
    elfHeader.e_version(1);
    elfHeader.e_ident(llvm::ELF::EI_VERSION, llvm::ELF::EV_CURRENT);
    elfHeader.e_ident(llvm::ELF::EI_OSABI, llvm::ELF::ELFOSABI_NONE);
    if (_targetLayout.findOutputSection(".got.plt"))
      elfHeader.e_ident(llvm::ELF::EI_ABIVERSION, 1);
    else
      elfHeader.e_ident(llvm::ELF::EI_ABIVERSION, 0);

    elfHeader.e_flags(_ctx.getMergedELFFlags());
  }

  void finalizeMipsRuntimeAtomValues() {
    if (!_ctx.isDynamic())
      return;

    auto gotSection = _targetLayout.findOutputSection(".got");
    auto got = gotSection ? gotSection->virtualAddr() : 0;
    auto gp = gotSection ? got + _targetLayout.getGPOffset() : 0;

    setAtomValue("_gp", gp);
    setAtomValue("_gp_disp", gp);
    setAtomValue("__gnu_local_gp", gp);
  }

  std::unique_ptr<RuntimeFile<ELFT>> createRuntimeFile() {
    auto file = llvm::make_unique<RuntimeFile<ELFT>>(_ctx, "Mips runtime file");
    if (_ctx.isDynamic()) {
      file->addAbsoluteAtom("_gp");
      file->addAbsoluteAtom("_gp_disp");
      file->addAbsoluteAtom("__gnu_local_gp");
      file->addAtom(*new (file->allocator()) MipsDynamicAtom(*file));
    }
    return file;
  }

private:
  MipsLinkingContext &_ctx;
  MipsTargetLayout<ELFT> &_targetLayout;

  void setAtomValue(StringRef name, uint64_t value) {
    AtomLayout *atom = _targetLayout.findAbsoluteAtom(name);
    assert(atom);
    atom->_virtualAddr = value;
  }
};

} // elf
} // lld

#endif
