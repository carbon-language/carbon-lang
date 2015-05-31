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
  enum MipsSectionOrder {
    ORDER_MIPS_ABI_FLAGS = TargetLayout<ELFT>::ORDER_RO_NOTE + 1,
    ORDER_MIPS_REGINFO,
    ORDER_MIPS_OPTIONS,
  };

  MipsTargetLayout(MipsLinkingContext &ctx);

  const MipsGOTSection<ELFT> &getGOTSection() const { return *_gotSection; }
  const MipsPLTSection<ELFT> &getPLTSection() const { return *_pltSection; }

  AtomSection<ELFT> *
  createSection(StringRef name, int32_t type,
                DefinedAtom::ContentPermissions permissions,
                typename TargetLayout<ELFT>::SectionOrder order) override;

  typename TargetLayout<ELFT>::SegmentType
  getSegmentType(Section<ELFT> *section) const override;

  /// \brief GP offset relative to .got section.
  uint64_t getGPOffset() const { return 0x7FF0; }

  /// \brief Get '_gp' symbol address.
  uint64_t getGPAddr();

  /// \brief Return the section order for a input section
  typename TargetLayout<ELFT>::SectionOrder
  getSectionOrder(StringRef name, int32_t contentType,
                  int32_t contentPermissions) override;

protected:
  unique_bump_ptr<RelocationTable<ELFT>>
  createRelocationTable(StringRef name, int32_t order) override;
  uint64_t getLookupSectionFlags(const OutputSection<ELFT> *os) const override;
  void sortSegments() override;

private:
  MipsGOTSection<ELFT> *_gotSection;
  MipsPLTSection<ELFT> *_pltSection;
  uint64_t _gpAddr = 0;
  std::once_flag _gpOnce;
};

} // end namespace elf
} // end namespace lld

#endif
