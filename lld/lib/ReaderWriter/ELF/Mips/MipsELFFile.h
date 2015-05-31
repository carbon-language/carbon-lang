//===- lib/ReaderWriter/ELF/MipsELFFile.h ---------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_MIPS_ELF_FILE_H
#define LLD_READER_WRITER_ELF_MIPS_MIPS_ELF_FILE_H

#include "ELFReader.h"
#include "MipsLinkingContext.h"
#include "MipsRelocationHandler.h"
#include "llvm/ADT/STLExtras.h"

namespace lld {
namespace elf {

template <class ELFT> class MipsELFFile;

template <class ELFT>
class MipsELFDefinedAtom : public ELFDefinedAtom<ELFT> {
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;
  typedef llvm::object::Elf_Shdr_Impl<ELFT> Elf_Shdr;

public:
  MipsELFDefinedAtom(const MipsELFFile<ELFT> &file, StringRef symbolName,
                     StringRef sectionName, const Elf_Sym *symbol,
                     const Elf_Shdr *section, ArrayRef<uint8_t> contentData,
                     unsigned int referenceStart, unsigned int referenceEnd,
                     std::vector<ELFReference<ELFT> *> &referenceList);

  const MipsELFFile<ELFT>& file() const override;
  DefinedAtom::CodeModel codeModel() const override;
};

template <class ELFT> class MipsELFReference : public ELFReference<ELFT> {
  typedef llvm::object::Elf_Rel_Impl<ELFT, false> Elf_Rel;
  typedef llvm::object::Elf_Rel_Impl<ELFT, true> Elf_Rela;

public:
  MipsELFReference(uint64_t symValue, const Elf_Rela &rel);
  MipsELFReference(uint64_t symValue, const Elf_Rel &rel);

  uint32_t tag() const override { return _tag; }
  void setTag(uint32_t tag) { _tag = tag; }

private:
  uint32_t _tag;
};

template <class ELFT> class MipsELFFile : public ELFFile<ELFT> {
public:
  MipsELFFile(std::unique_ptr<MemoryBuffer> mb, ELFLinkingContext &ctx);

  bool isPIC() const;

  /// \brief gp register value stored in the .reginfo section.
  int64_t getGP0() const { return _gp0; }

  /// \brief .tdata section address plus fixed offset.
  uint64_t getTPOffset() const { return _tpOff; }
  uint64_t getDTPOffset() const { return _dtpOff; }

protected:
  std::error_code doParse() override;

private:
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;
  typedef llvm::object::Elf_Shdr_Impl<ELFT> Elf_Shdr;
  typedef llvm::object::Elf_Rel_Impl<ELFT, false> Elf_Rel;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rel_Iter Elf_Rel_Iter;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rela_Iter Elf_Rela_Iter;

  enum { TP_OFFSET = 0x7000, DTP_OFFSET = 0x8000 };

  int64_t _gp0 = 0;
  uint64_t _tpOff = 0;
  uint64_t _dtpOff = 0;

  ELFDefinedAtom<ELFT> *
  createDefinedAtom(StringRef symName, StringRef sectionName,
                    const Elf_Sym *sym, const Elf_Shdr *sectionHdr,
                    ArrayRef<uint8_t> contentData, unsigned int referenceStart,
                    unsigned int referenceEnd,
                    std::vector<ELFReference<ELFT> *> &referenceList) override;

  void createRelocationReferences(const Elf_Sym *symbol,
                                  ArrayRef<uint8_t> content,
                                  range<Elf_Rela_Iter> rels) override;
  void createRelocationReferences(const Elf_Sym *symbol,
                                  ArrayRef<uint8_t> symContent,
                                  ArrayRef<uint8_t> secContent,
                                  range<Elf_Rel_Iter> rels) override;

  const Elf_Shdr *findSectionByType(uint64_t type) const;
  const Elf_Shdr *findSectionByFlags(uint64_t flags) const;

  typedef typename llvm::object::ELFFile<ELFT>::Elf_Ehdr Elf_Ehdr;
  typedef llvm::object::Elf_Mips_RegInfo<ELFT> Elf_Mips_RegInfo;
  typedef llvm::object::Elf_Mips_Options<ELFT> Elf_Mips_Options;
  typedef llvm::object::Elf_Mips_ABIFlags<ELFT> Elf_Mips_ABIFlags;

  ErrorOr<const Elf_Mips_RegInfo *> findRegInfoSec() const;
  ErrorOr<const Elf_Mips_ABIFlags*> findAbiFlagsSec() const;

  std::error_code readAuxData();

  Reference::Addend readAddend(const Elf_Rel &ri,
                               const ArrayRef<uint8_t> content) const;

  uint32_t getPairRelocation(const Elf_Rel &rel) const;

  Elf_Rel_Iter findMatchingRelocation(uint32_t pairRelType, Elf_Rel_Iter rit,
                                      Elf_Rel_Iter eit) const;

  bool isLocalBinding(const Elf_Rel &rel) const;
};

} // elf
} // lld

#endif
