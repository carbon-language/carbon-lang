//===- lib/ReaderWriter/ELF/Mips/MipsTargetLayout.cpp ---------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsLinkingContext.h"
#include "MipsTargetLayout.h"

namespace lld {
namespace elf {

template <class ELFT>
MipsTargetLayout<ELFT>::MipsTargetLayout(MipsLinkingContext &ctx,
                                         MipsAbiInfoHandler<ELFT> &abi)
    : TargetLayout<ELFT>(ctx), _abiInfo(abi),
      _gotSection(new (this->_allocator) MipsGOTSection<ELFT>(ctx)),
      _pltSection(new (this->_allocator) MipsPLTSection<ELFT>(ctx)) {}

template <class ELFT>
AtomSection<ELFT> *MipsTargetLayout<ELFT>::createSection(
    StringRef name, int32_t type, DefinedAtom::ContentPermissions permissions,
    typename TargetLayout<ELFT>::SectionOrder order) {
  if (type == DefinedAtom::typeGOT && name == ".got")
    return _gotSection;
  if (type == DefinedAtom::typeStub && name == ".plt")
    return _pltSection;
  return TargetLayout<ELFT>::createSection(name, type, permissions, order);
}

template <class ELFT>
typename TargetLayout<ELFT>::SegmentType
MipsTargetLayout<ELFT>::getSegmentType(const Section<ELFT> *section) const {
  switch (section->order()) {
  case ORDER_MIPS_REGINFO:
    return _abiInfo.hasMipsAbiSection() ? llvm::ELF::PT_LOAD
                                        : llvm::ELF::PT_MIPS_REGINFO;
  case ORDER_MIPS_OPTIONS:
    return llvm::ELF::PT_LOAD;
  case ORDER_MIPS_ABI_FLAGS:
    return llvm::ELF::PT_MIPS_ABIFLAGS;
  default:
    return TargetLayout<ELFT>::getSegmentType(section);
  }
}

template <class ELFT> uint64_t MipsTargetLayout<ELFT>::getGPAddr() {
  std::call_once(_gpOnce, [this]() {
    if (AtomLayout *a = this->findAbsoluteAtom("_gp"))
      _gpAddr = a->_virtualAddr;
  });
  return _gpAddr;
}

template <class ELFT>
typename TargetLayout<ELFT>::SectionOrder
MipsTargetLayout<ELFT>::getSectionOrder(StringRef name, int32_t contentType,
                                        int32_t contentPermissions) {
  if ((contentType == DefinedAtom::typeStub) && (name.startswith(".text")))
    return TargetLayout<ELFT>::ORDER_TEXT;

  return TargetLayout<ELFT>::getSectionOrder(name, contentType,
                                             contentPermissions);
}

template <class ELFT>
unique_bump_ptr<RelocationTable<ELFT>>
MipsTargetLayout<ELFT>::createRelocationTable(StringRef name, int32_t order) {
  return unique_bump_ptr<RelocationTable<ELFT>>(new (
      this->_allocator) MipsRelocationTable<ELFT>(this->_ctx, name, order));
}

template <class ELFT>
uint64_t MipsTargetLayout<ELFT>::getLookupSectionFlags(
    const OutputSection<ELFT> *os) const {
  uint64_t flags = TargetLayout<ELFT>::getLookupSectionFlags(os);
  return flags & ~llvm::ELF::SHF_MIPS_NOSTRIP;
}

template <class ELFT> void MipsTargetLayout<ELFT>::sortSegments() {
  using namespace llvm::ELF;
  TargetLayout<ELFT>::sortSegments();
  // Move PT_MIPS_ABIFLAGS or PT_MIPS_REGINFO right after PT_INTERP.
  auto abiIt =
      std::find_if(this->_segments.begin(), this->_segments.end(),
                   [](const Segment<ELFT> *s) {
                     auto typ = s->segmentType();
                     return typ == PT_MIPS_ABIFLAGS || typ == PT_MIPS_REGINFO;
                   });
  if (abiIt == this->_segments.end())
    return;
  Segment<ELFT> *abiSeg = *abiIt;
  this->_segments.erase(abiIt);
  auto outIt = std::find_if(this->_segments.begin(), this->_segments.end(),
                            [](const Segment<ELFT> *s) {
                              auto typ = s->segmentType();
                              return typ != PT_PHDR && typ != PT_INTERP;
                            });
  this->_segments.insert(outIt, abiSeg);
}

template class MipsTargetLayout<ELF32BE>;
template class MipsTargetLayout<ELF32LE>;
template class MipsTargetLayout<ELF64BE>;
template class MipsTargetLayout<ELF64LE>;

} // end namespace elf
} // end namespace lld
