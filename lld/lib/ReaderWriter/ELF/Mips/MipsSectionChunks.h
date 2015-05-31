//===- lib/ReaderWriter/ELF/Mips/MipsSectionChunks.h ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_MIPS_SECTION_CHUNKS_H
#define LLD_READER_WRITER_ELF_MIPS_MIPS_SECTION_CHUNKS_H

#include "SectionChunks.h"

namespace lld {
namespace elf {

template <typename ELFT> class MipsTargetLayout;
class MipsLinkingContext;

/// \brief Handle Mips .reginfo section
template <class ELFT> class MipsReginfoSection : public Section<ELFT> {
public:
  typedef llvm::object::Elf_Mips_RegInfo<ELFT> Elf_Mips_RegInfo;

  MipsReginfoSection(const ELFLinkingContext &ctx,
                     MipsTargetLayout<ELFT> &targetLayout,
                     const Elf_Mips_RegInfo &reginfo);

  StringRef segmentKindToStr() const override { return "REGINFO"; }
  bool hasOutputSegment() const override { return true; }

  void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
             llvm::FileOutputBuffer &buffer) override;
  void finalize() override;

private:
  Elf_Mips_RegInfo _reginfo;
  MipsTargetLayout<ELFT> &_targetLayout;
};

/// \brief Handle .MIPS.options section
template <class ELFT> class MipsOptionsSection : public Section<ELFT> {
public:
  typedef llvm::object::Elf_Mips_Options<ELFT> Elf_Mips_Options;
  typedef llvm::object::Elf_Mips_RegInfo<ELFT> Elf_Mips_RegInfo;

  MipsOptionsSection(const ELFLinkingContext &ctx,
                     MipsTargetLayout<ELFT> &targetLayout,
                     const Elf_Mips_RegInfo &reginfo);

  bool hasOutputSegment() const override { return true; }

  void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
             llvm::FileOutputBuffer &buffer) override;
  void finalize() override;

private:
  Elf_Mips_Options _header;
  Elf_Mips_RegInfo _reginfo;
  MipsTargetLayout<ELFT> &_targetLayout;
};

/// \brief Handle Mips GOT section
template <class ELFT> class MipsGOTSection : public AtomSection<ELFT> {
public:
  MipsGOTSection(const MipsLinkingContext &ctx);

  /// \brief Number of local GOT entries.
  std::size_t getLocalCount() const { return _localCount; }

  /// \brief Number of global GOT entries.
  std::size_t getGlobalCount() const { return _posMap.size(); }

  /// \brief Does the atom have a global GOT entry?
  bool hasGlobalGOTEntry(const Atom *a) const {
    return _posMap.count(a) || _tlsMap.count(a);
  }

  /// \brief Compare two atoms accordingly theirs positions in the GOT.
  bool compare(const Atom *a, const Atom *b) const;

  const AtomLayout *appendAtom(const Atom *atom) override;

private:
  /// \brief True if the GOT contains non-local entries.
  bool _hasNonLocal;

  /// \brief Number of local GOT entries.
  std::size_t _localCount;

  /// \brief Map TLS Atoms to their GOT entry index.
  llvm::DenseMap<const Atom *, std::size_t> _tlsMap;

  /// \brief Map Atoms to their GOT entry index.
  llvm::DenseMap<const Atom *, std::size_t> _posMap;
};

/// \brief Handle Mips PLT section
template <class ELFT> class MipsPLTSection : public AtomSection<ELFT> {
public:
  MipsPLTSection(const MipsLinkingContext &ctx);

  const AtomLayout *findPLTLayout(const Atom *plt) const;

  const AtomLayout *appendAtom(const Atom *atom) override;

private:
  /// \brief Map PLT Atoms to their layouts.
  std::unordered_map<const Atom *, const AtomLayout *> _pltLayoutMap;
};

template <class ELFT> class MipsRelocationTable : public RelocationTable<ELFT> {
  typedef llvm::object::Elf_Rel_Impl<ELFT, false> Elf_Rel;
  typedef llvm::object::Elf_Rel_Impl<ELFT, true> Elf_Rela;

public:
  MipsRelocationTable(const ELFLinkingContext &ctx, StringRef str,
                      int32_t order);

protected:
  void writeRela(ELFWriter *writer, Elf_Rela &r, const DefinedAtom &atom,
                 const Reference &ref) override;
  void writeRel(ELFWriter *writer, Elf_Rel &r, const DefinedAtom &atom,
                const Reference &ref) override;
};

} // elf
} // lld

#endif
