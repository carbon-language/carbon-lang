//===- lib/ReaderWriter/ELF/Mips/MipsTargetLayout.h -----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_MIPS_TARGET_LAYOUT_H
#define LLD_READER_WRITER_ELF_MIPS_MIPS_TARGET_LAYOUT_H

#include "MipsSectionChunks.h"
#include "TargetLayout.h"

namespace lld {
namespace elf {

class MipsLinkingContext;

/// \brief TargetLayout for Mips
template <class ELFT> class MipsTargetLayout final : public TargetLayout<ELFT> {
public:
  MipsTargetLayout(MipsLinkingContext &ctx)
      : TargetLayout<ELFT>(ctx),
        _gotSection(new (this->_allocator) MipsGOTSection<ELFT>(ctx)),
        _pltSection(new (this->_allocator) MipsPLTSection<ELFT>(ctx)) {}

  const MipsGOTSection<ELFT> &getGOTSection() const { return *_gotSection; }
  const MipsPLTSection<ELFT> &getPLTSection() const { return *_pltSection; }

  AtomSection<ELFT> *
  createSection(StringRef name, int32_t type,
                DefinedAtom::ContentPermissions permissions,
                typename TargetLayout<ELFT>::SectionOrder order) override {
    if (type == DefinedAtom::typeGOT && name == ".got")
      return _gotSection;
    if (type == DefinedAtom::typeStub && name == ".plt")
      return _pltSection;
    return TargetLayout<ELFT>::createSection(name, type, permissions, order);
  }

  /// \brief GP offset relative to .got section.
  uint64_t getGPOffset() const { return 0x7FF0; }

  /// \brief Get '_gp' symbol address.
  uint64_t getGPAddr() {
    std::call_once(_gpOnce, [this]() {
      if (AtomLayout *a = this->findAbsoluteAtom("_gp"))
        _gpAddr = a->_virtualAddr;
    });
    return _gpAddr;
  }

  /// \brief Return the section order for a input section
  typename TargetLayout<ELFT>::SectionOrder
  getSectionOrder(StringRef name, int32_t contentType,
                  int32_t contentPermissions) override {
    if ((contentType == DefinedAtom::typeStub) && (name.startswith(".text")))
      return TargetLayout<ELFT>::ORDER_TEXT;

    return TargetLayout<ELFT>::getSectionOrder(name, contentType,
                                               contentPermissions);
  }

protected:
  unique_bump_ptr<RelocationTable<ELFT>>
  createRelocationTable(StringRef name, int32_t order) override {
    return unique_bump_ptr<RelocationTable<ELFT>>(new (
        this->_allocator) MipsRelocationTable<ELFT>(this->_ctx, name, order));
  }

private:
  MipsGOTSection<ELFT> *_gotSection;
  MipsPLTSection<ELFT> *_pltSection;
  uint64_t _gpAddr = 0;
  std::once_flag _gpOnce;
};

} // end namespace elf
} // end namespace lld

#endif
