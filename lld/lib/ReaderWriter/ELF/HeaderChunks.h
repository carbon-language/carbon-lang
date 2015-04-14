//===- lib/ReaderWriter/ELF/HeaderChunks.h --------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_HEADER_CHUNKS_H
#define LLD_READER_WRITER_ELF_HEADER_CHUNKS_H

#include "SegmentChunks.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileOutputBuffer.h"

/// \brief An Header represents the Elf[32/64]_Ehdr structure at the
///        start of an ELF executable file.
namespace lld {
namespace elf {

template <class ELFT> class ELFHeader : public Chunk<ELFT> {
public:
  typedef llvm::object::Elf_Ehdr_Impl<ELFT> Elf_Ehdr;

  ELFHeader(const ELFLinkingContext &);

  void e_ident(int I, unsigned char C) { _eh.e_ident[I] = C; }
  void e_type(uint16_t type)           { _eh.e_type = type; }
  void e_machine(uint16_t machine)     { _eh.e_machine = machine; }
  void e_version(uint32_t version)     { _eh.e_version = version; }
  void e_entry(int64_t entry)          { _eh.e_entry = entry; }
  void e_phoff(int64_t phoff)          { _eh.e_phoff = phoff; }
  void e_shoff(int64_t shoff)          { _eh.e_shoff = shoff; }
  void e_flags(uint32_t flags)         { _eh.e_flags = flags; }
  void e_ehsize(uint16_t ehsize)       { _eh.e_ehsize = ehsize; }
  void e_phentsize(uint16_t phentsize) { _eh.e_phentsize = phentsize; }
  void e_phnum(uint16_t phnum)         { _eh.e_phnum = phnum; }
  void e_shentsize(uint16_t shentsize) { _eh.e_shentsize = shentsize; }
  void e_shnum(uint16_t shnum)         { _eh.e_shnum = shnum; }
  void e_shstrndx(uint16_t shstrndx)   { _eh.e_shstrndx = shstrndx; }
  uint64_t fileSize() const override { return sizeof(Elf_Ehdr); }

  static bool classof(const Chunk<ELFT> *c) {
    return c->kind() == Chunk<ELFT>::Kind::ELFHeader;
  }

  int getContentType() const override {
    return Chunk<ELFT>::ContentType::Header;
  }

  void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
             llvm::FileOutputBuffer &buffer) override;

  void finalize() override;

private:
  Elf_Ehdr _eh;
};

/// \brief An ProgramHeader represents the Elf[32/64]_Phdr structure at the
///        start of an ELF executable file.
template<class ELFT>
class ProgramHeader : public Chunk<ELFT> {
public:
  typedef llvm::object::Elf_Phdr_Impl<ELFT> Elf_Phdr;
  typedef typename std::vector<Elf_Phdr *>::iterator PhIterT;
  typedef typename std::reverse_iterator<PhIterT> ReversePhIterT;

  ProgramHeader(const ELFLinkingContext &ctx)
      : Chunk<ELFT>("elfphdr", Chunk<ELFT>::Kind::ProgramHeader, ctx) {
    this->_alignment = ELFT::Is64Bits ? 8 : 4;
    resetProgramHeaders();
  }

  bool addSegment(Segment<ELFT> *segment);
  void resetProgramHeaders() { _phi = _ph.begin(); }
  uint64_t fileSize() const override { return sizeof(Elf_Phdr) * _ph.size(); }

  static bool classof(const Chunk<ELFT> *c) {
    return c->kind() == Chunk<ELFT>::Kind::ProgramHeader;
  }

  void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
             llvm::FileOutputBuffer &buffer) override;

  PhIterT begin() { return _ph.begin(); }
  PhIterT end() { return _ph.end(); }
  ReversePhIterT rbegin() { return _ph.rbegin(); }
  ReversePhIterT rend() { return _ph.rend(); }

  int64_t entsize() { return sizeof(Elf_Phdr); }
  int64_t numHeaders() { return _ph.size();  }

  int getContentType() const override {
    return Chunk<ELFT>::ContentType::Header;
  }

private:
  Elf_Phdr *allocateProgramHeader(bool &allocatedNew);

  std::vector<Elf_Phdr *> _ph;
  PhIterT _phi;
  llvm::BumpPtrAllocator _allocator;
};

/// \brief An SectionHeader represents the Elf[32/64]_Shdr structure
/// at the end of the file
template<class ELFT>
class SectionHeader : public Chunk<ELFT> {
public:
  typedef llvm::object::Elf_Shdr_Impl<ELFT> Elf_Shdr;

  SectionHeader(const ELFLinkingContext &, int32_t order);
  void appendSection(OutputSection<ELFT> *section);
  void updateSection(Section<ELFT> *section);

  static bool classof(const Chunk<ELFT> *c) {
    return c->kind() == Chunk<ELFT>::Kind::SectionHeader;
  }

  void setStringSection(StringTable<ELFT> *s) {
    _stringSection = s;
  }

  void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
             llvm::FileOutputBuffer &buffer) override;

  uint64_t fileSize() const override {
    return sizeof(Elf_Shdr) * _sectionInfo.size();
  }

  uint64_t entsize() { return sizeof(Elf_Shdr); }

  int getContentType() const override {
    return Chunk<ELFT>::ContentType::Header;
  }

  uint64_t numHeaders() { return _sectionInfo.size(); }

private:
  StringTable<ELFT> *_stringSection;
  std::vector<Elf_Shdr *> _sectionInfo;
  llvm::BumpPtrAllocator _sectionAllocate;
};

} // end namespace elf
} // end namespace lld

#endif
