//===- OutputSections.cpp -------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OutputSections.h"
#include "Config.h"
#include "SymbolTable.h"
#include "Target.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;

using namespace lld;
using namespace lld::elf2;

template <bool Is64Bits>
OutputSectionBase<Is64Bits>::OutputSectionBase(StringRef Name, uint32_t sh_type,
                                               uintX_t sh_flags)
    : Name(Name) {
  memset(&Header, 0, sizeof(HeaderT));
  Header.sh_type = sh_type;
  Header.sh_flags = sh_flags;
}

template <class ELFT>
GotSection<ELFT>::GotSection()
    : OutputSectionBase<ELFT::Is64Bits>(".got", llvm::ELF::SHT_PROGBITS,
                                        llvm::ELF::SHF_ALLOC |
                                            llvm::ELF::SHF_WRITE) {
  this->Header.sh_addralign = this->getAddrSize();
}

template <class ELFT> void GotSection<ELFT>::addEntry(SymbolBody *Sym) {
  Sym->setGotIndex(Entries.size());
  Entries.push_back(Sym);
}

template <class ELFT>
typename GotSection<ELFT>::uintX_t
GotSection<ELFT>::getEntryAddr(const SymbolBody &B) const {
  return this->getVA() + B.getGotIndex() * this->getAddrSize();
}

template <class ELFT>
PltSection<ELFT>::PltSection(const GotSection<ELFT> &GotSec)
    : OutputSectionBase<ELFT::Is64Bits>(".plt", llvm::ELF::SHT_PROGBITS,
                                        llvm::ELF::SHF_ALLOC |
                                            llvm::ELF::SHF_EXECINSTR),
      GotSec(GotSec) {
  this->Header.sh_addralign = 16;
}

template <class ELFT> void PltSection<ELFT>::writeTo(uint8_t *Buf) {
  uintptr_t Start = reinterpret_cast<uintptr_t>(Buf);
  for (const SymbolBody *E : Entries) {
    uint64_t GotEntryAddr = GotSec.getEntryAddr(*E);
    uintptr_t InstPos = reinterpret_cast<uintptr_t>(Buf);
    uint64_t PltEntryAddr = (InstPos - Start) + this->getVA();
    Target->writePltEntry(Buf, GotEntryAddr, PltEntryAddr);
    Buf += 8;
  }
}

template <class ELFT> void PltSection<ELFT>::addEntry(SymbolBody *Sym) {
  Sym->setPltIndex(Entries.size());
  Entries.push_back(Sym);
}

template <class ELFT>
typename PltSection<ELFT>::uintX_t
PltSection<ELFT>::getEntryAddr(const SymbolBody &B) const {
  return this->getVA() + B.getPltIndex() * EntrySize;
}

template <class ELFT>
RelocationSection<ELFT>::RelocationSection(SymbolTableSection<ELFT> &DynSymSec,
                                           const GotSection<ELFT> &GotSec,
                                           bool IsRela)
    : OutputSectionBase<ELFT::Is64Bits>(IsRela ? ".rela.dyn" : ".rel.dyn",
                                        IsRela ? llvm::ELF::SHT_RELA
                                               : llvm::ELF::SHT_REL,
                                        llvm::ELF::SHF_ALLOC),
      DynSymSec(DynSymSec), GotSec(GotSec), IsRela(IsRela) {
  this->Header.sh_entsize = IsRela ? sizeof(Elf_Rela) : sizeof(Elf_Rel);
  this->Header.sh_addralign = ELFT::Is64Bits ? 8 : 4;
}

template <class ELFT> void RelocationSection<ELFT>::writeTo(uint8_t *Buf) {
  const unsigned EntrySize = IsRela ? sizeof(Elf_Rela) : sizeof(Elf_Rel);
  bool IsMips64EL = Relocs[0].C.getFile()->getObj().isMips64EL();
  for (const DynamicReloc<ELFT> &Rel : Relocs) {
    auto *P = reinterpret_cast<Elf_Rel *>(Buf);
    Buf += EntrySize;

    const InputSection<ELFT> &C = Rel.C;
    const Elf_Rel &RI = Rel.RI;
    OutputSection<ELFT> *Out = C.getOutputSection();
    uint32_t SymIndex = RI.getSymbol(IsMips64EL);
    const SymbolBody *Body = C.getFile()->getSymbolBody(SymIndex);
    uint32_t Type = RI.getType(IsMips64EL);
    if (Target->relocNeedsGot(Type, *Body)) {
      P->r_offset = GotSec.getEntryAddr(*Body);
      P->setSymbolAndType(Body->getDynamicSymbolTableIndex(),
                          Target->getGotReloc(), IsMips64EL);
    } else {
      P->r_offset = RI.r_offset + C.getOutputSectionOff() + Out->getVA();
      P->setSymbolAndType(Body->getDynamicSymbolTableIndex(), Type, IsMips64EL);
      if (IsRela)
        static_cast<Elf_Rela *>(P)->r_addend =
            static_cast<const Elf_Rela &>(RI).r_addend;
    }
  }
}

template <class ELFT> void RelocationSection<ELFT>::finalize() {
  this->Header.sh_link = DynSymSec.getSectionIndex();
  this->Header.sh_size = Relocs.size() * this->Header.sh_entsize;
}

template <bool Is64Bits>
InterpSection<Is64Bits>::InterpSection()
    : OutputSectionBase<Is64Bits>(".interp", llvm::ELF::SHT_PROGBITS,
                                  llvm::ELF::SHF_ALLOC) {
  this->Header.sh_size = Config->DynamicLinker.size() + 1;
  this->Header.sh_addralign = 1;
}

template <bool Is64Bits>
template <endianness E>
void OutputSectionBase<Is64Bits>::writeHeaderTo(
    typename ELFFile<ELFType<E, Is64Bits>>::Elf_Shdr *SHdr) {
  SHdr->sh_name = Header.sh_name;
  SHdr->sh_type = Header.sh_type;
  SHdr->sh_flags = Header.sh_flags;
  SHdr->sh_addr = Header.sh_addr;
  SHdr->sh_offset = Header.sh_offset;
  SHdr->sh_size = Header.sh_size;
  SHdr->sh_link = Header.sh_link;
  SHdr->sh_info = Header.sh_info;
  SHdr->sh_addralign = Header.sh_addralign;
  SHdr->sh_entsize = Header.sh_entsize;
}

template <bool Is64Bits> void InterpSection<Is64Bits>::writeTo(uint8_t *Buf) {
  memcpy(Buf, Config->DynamicLinker.data(), Config->DynamicLinker.size());
}

template <class ELFT>
HashTableSection<ELFT>::HashTableSection(SymbolTableSection<ELFT> &DynSymSec)
    : OutputSectionBase<ELFT::Is64Bits>(".hash", llvm::ELF::SHT_HASH,
                                        llvm::ELF::SHF_ALLOC),
      DynSymSec(DynSymSec) {
  this->Header.sh_entsize = sizeof(Elf_Word);
  this->Header.sh_addralign = sizeof(Elf_Word);
}

template <class ELFT> void HashTableSection<ELFT>::addSymbol(SymbolBody *S) {
  StringRef Name = S->getName();
  DynSymSec.addSymbol(Name);
  Hashes.push_back(hash(Name));
  S->setDynamicSymbolTableIndex(Hashes.size());
}

template <class ELFT>
DynamicSection<ELFT>::DynamicSection(SymbolTable &SymTab,
                                     HashTableSection<ELFT> &HashSec,
                                     RelocationSection<ELFT> &RelaDynSec)
    : OutputSectionBase<ELFT::Is64Bits>(".dynamic", llvm::ELF::SHT_DYNAMIC,
                                        llvm::ELF::SHF_ALLOC |
                                            llvm::ELF::SHF_WRITE),
      HashSec(HashSec), DynSymSec(HashSec.getDynSymSec()),
      DynStrSec(DynSymSec.getStrTabSec()), RelaDynSec(RelaDynSec),
      SymTab(SymTab) {
  typename Base::HeaderT &Header = this->Header;
  Header.sh_addralign = ELFT::Is64Bits ? 8 : 4;
  Header.sh_entsize = ELFT::Is64Bits ? 16 : 8;
}

template <class ELFT> void DynamicSection<ELFT>::finalize() {
  if (this->Header.sh_size)
    return; // Already finalized.

  typename Base::HeaderT &Header = this->Header;
  Header.sh_link = DynStrSec.getSectionIndex();

  unsigned NumEntries = 0;
  if (RelaDynSec.hasRelocs()) {
    ++NumEntries; // DT_RELA / DT_REL
    ++NumEntries; // DT_RELASZ / DT_RELSZ
    ++NumEntries; // DT_RELAENT / DT_RELENT
  }
  ++NumEntries; // DT_SYMTAB
  ++NumEntries; // DT_SYMENT
  ++NumEntries; // DT_STRTAB
  ++NumEntries; // DT_STRSZ
  ++NumEntries; // DT_HASH

  if (!Config->RPath.empty()) {
    ++NumEntries; // DT_RUNPATH
    DynStrSec.add(Config->RPath);
  }

  if (!Config->SoName.empty()) {
    ++NumEntries; // DT_SONAME
    DynStrSec.add(Config->SoName);
  }

  if (PreInitArraySec)
    NumEntries += 2;
  if (InitArraySec)
    NumEntries += 2;
  if (FiniArraySec)
    NumEntries += 2;

  const std::vector<std::unique_ptr<SharedFileBase>> &SharedFiles =
      SymTab.getSharedFiles();
  for (const std::unique_ptr<SharedFileBase> &File : SharedFiles)
    DynStrSec.add(File->getSoName());
  NumEntries += SharedFiles.size();

  ++NumEntries; // DT_NULL

  Header.sh_size = NumEntries * Header.sh_entsize;
}

template <class ELFT> void DynamicSection<ELFT>::writeTo(uint8_t *Buf) {
  auto *P = reinterpret_cast<Elf_Dyn *>(Buf);

  auto WritePtr = [&](int32_t Tag, uint64_t Val) {
    P->d_tag = Tag;
    P->d_un.d_ptr = Val;
    ++P;
  };

  auto WriteVal = [&](int32_t Tag, uint32_t Val) {
    P->d_tag = Tag;
    P->d_un.d_val = Val;
    ++P;
  };

  if (RelaDynSec.hasRelocs()) {
    bool IsRela = RelaDynSec.isRela();
    WritePtr(IsRela ? DT_RELA : DT_REL, RelaDynSec.getVA());
    WriteVal(IsRela ? DT_RELASZ : DT_RELSZ, RelaDynSec.getSize());
    WriteVal(IsRela ? DT_RELAENT : DT_RELENT,
             IsRela ? sizeof(Elf_Rela) : sizeof(Elf_Rel));
  }

  WritePtr(DT_SYMTAB, DynSymSec.getVA());
  WritePtr(DT_SYMENT, sizeof(Elf_Sym));
  WritePtr(DT_STRTAB, DynStrSec.getVA());
  WriteVal(DT_STRSZ, DynStrSec.data().size());
  WritePtr(DT_HASH, HashSec.getVA());

  if (!Config->RPath.empty())
    WriteVal(DT_RUNPATH, DynStrSec.getFileOff(Config->RPath));

  if (!Config->SoName.empty())
    WriteVal(DT_SONAME, DynStrSec.getFileOff(Config->SoName));

  auto WriteArray = [&](int32_t T1, int32_t T2,
                        const OutputSection<ELFT> *Sec) {
    if (!Sec)
      return;
    WritePtr(T1, Sec->getVA());
    WriteVal(T2, Sec->getSize());
  };
  WriteArray(DT_PREINIT_ARRAY, DT_PREINIT_ARRAYSZ, PreInitArraySec);
  WriteArray(DT_INIT_ARRAY, DT_INIT_ARRAYSZ, InitArraySec);
  WriteArray(DT_FINI_ARRAY, DT_FINI_ARRAYSZ, FiniArraySec);

  const std::vector<std::unique_ptr<SharedFileBase>> &SharedFiles =
      SymTab.getSharedFiles();
  for (const std::unique_ptr<SharedFileBase> &File : SharedFiles)
    WriteVal(DT_NEEDED, DynStrSec.getFileOff(File->getSoName()));

  WriteVal(DT_NULL, 0);
}

template <class ELFT>
OutputSection<ELFT>::OutputSection(const PltSection<ELFT> &PltSec,
                                   const GotSection<ELFT> &GotSec,
                                   const OutputSection<ELFT> &BssSec,
                                   StringRef Name, uint32_t sh_type,
                                   uintX_t sh_flags)
    : OutputSectionBase<ELFT::Is64Bits>(Name, sh_type, sh_flags),
      PltSec(PltSec), GotSec(GotSec), BssSec(BssSec) {}

template <class ELFT>
void OutputSection<ELFT>::addSection(InputSection<ELFT> *C) {
  Sections.push_back(C);
  C->setOutputSection(this);
  uint32_t Align = C->getAlign();
  if (Align > this->Header.sh_addralign)
    this->Header.sh_addralign = Align;

  uintX_t Off = this->Header.sh_size;
  Off = RoundUpToAlignment(Off, Align);
  C->setOutputSectionOff(Off);
  Off += C->getSize();
  this->Header.sh_size = Off;
}

template <class ELFT>
typename ELFFile<ELFT>::uintX_t
lld::elf2::getSymVA(const ELFSymbolBody<ELFT> &S,
                    const OutputSection<ELFT> &BssSec) {
  switch (S.kind()) {
  case SymbolBody::DefinedSyntheticKind:
    return cast<DefinedSynthetic<ELFT>>(S).Section.getVA() + S.Sym.st_value;
  case SymbolBody::DefinedAbsoluteKind:
    return S.Sym.st_value;
  case SymbolBody::DefinedRegularKind: {
    const auto &DR = cast<DefinedRegular<ELFT>>(S);
    const InputSection<ELFT> *SC = &DR.Section;
    OutputSection<ELFT> *OS = SC->getOutputSection();
    return OS->getVA() + SC->getOutputSectionOff() + DR.Sym.st_value;
  }
  case SymbolBody::DefinedCommonKind:
    return BssSec.getVA() + cast<DefinedCommon<ELFT>>(S).OffsetInBSS;
  case SymbolBody::SharedKind:
  case SymbolBody::UndefinedKind:
    return 0;
  case SymbolBody::LazyKind:
    break;
  }
  llvm_unreachable("Lazy symbol reached writer");
}

template <class ELFT>
typename ELFFile<ELFT>::uintX_t
lld::elf2::getLocalSymVA(const typename ELFFile<ELFT>::Elf_Sym *Sym,
                         const ObjectFile<ELFT> &File) {
  uint32_t SecIndex = Sym->st_shndx;

  if (SecIndex == SHN_XINDEX)
    SecIndex = File.getObj().getExtendedSymbolTableIndex(
        Sym, File.getSymbolTable(), File.getSymbolTableShndx());
  ArrayRef<InputSection<ELFT> *> Sections = File.getSections();
  InputSection<ELFT> *Section = Sections[SecIndex];
  OutputSection<ELFT> *Out = Section->getOutputSection();
  return Out->getVA() + Section->getOutputSectionOff() + Sym->st_value;
}

template <class ELFT> void OutputSection<ELFT>::writeTo(uint8_t *Buf) {
  for (InputSection<ELFT> *C : Sections)
    C->writeTo(Buf, BssSec, PltSec, GotSec);
}

template <bool Is64Bits>
StringTableSection<Is64Bits>::StringTableSection(bool Dynamic)
    : OutputSectionBase<Is64Bits>(Dynamic ? ".dynstr" : ".strtab",
                                  llvm::ELF::SHT_STRTAB,
                                  Dynamic ? (uintX_t)llvm::ELF::SHF_ALLOC : 0),
      Dynamic(Dynamic) {
  this->Header.sh_addralign = 1;
}

template <bool Is64Bits>
void StringTableSection<Is64Bits>::writeTo(uint8_t *Buf) {
  StringRef Data = StrTabBuilder.data();
  memcpy(Buf, Data.data(), Data.size());
}

bool lld::elf2::includeInSymtab(const SymbolBody &B) {
  if (B.isLazy())
    return false;
  if (!B.isUsedInRegularObj())
    return false;
  uint8_t V = B.getMostConstrainingVisibility();
  if (V != STV_DEFAULT && V != STV_PROTECTED)
    return false;
  return true;
}

bool lld::elf2::includeInDynamicSymtab(const SymbolBody &B) {
  if (Config->ExportDynamic || Config->Shared)
    return true;
  return B.isUsedInDynamicReloc();
}

bool lld::elf2::shouldKeepInSymtab(StringRef SymName) {
  if (Config->DiscardNone)
    return true;

  // ELF defines dynamic locals as symbols which name starts with ".L".
  return !(Config->DiscardLocals && SymName.startswith(".L"));
}

template <class ELFT>
SymbolTableSection<ELFT>::SymbolTableSection(
    SymbolTable &Table, StringTableSection<ELFT::Is64Bits> &StrTabSec,
    const OutputSection<ELFT> &BssSec)
    : OutputSectionBase<ELFT::Is64Bits>(
          StrTabSec.isDynamic() ? ".dynsym" : ".symtab",
          StrTabSec.isDynamic() ? llvm::ELF::SHT_DYNSYM : llvm::ELF::SHT_SYMTAB,
          StrTabSec.isDynamic() ? (uintX_t)llvm::ELF::SHF_ALLOC : 0),
      Table(Table), StrTabSec(StrTabSec), BssSec(BssSec) {
  typedef OutputSectionBase<ELFT::Is64Bits> Base;
  typename Base::HeaderT &Header = this->Header;

  Header.sh_entsize = sizeof(Elf_Sym);
  Header.sh_addralign = ELFT::Is64Bits ? 8 : 4;
}

template <class ELFT> void SymbolTableSection<ELFT>::writeTo(uint8_t *Buf) {
  Buf += sizeof(Elf_Sym);

  // All symbols with STB_LOCAL binding precede the weak and global symbols.
  // .dynsym only contains global symbols.
  if (!Config->DiscardAll && !StrTabSec.isDynamic())
    writeLocalSymbols(Buf);

  writeGlobalSymbols(Buf);
}

template <class ELFT>
void SymbolTableSection<ELFT>::writeLocalSymbols(uint8_t *&Buf) {
  // Iterate over all input object files to copy their local symbols
  // to the output symbol table pointed by Buf.
  for (const std::unique_ptr<ObjectFileBase> &FileB : Table.getObjectFiles()) {
    auto &File = cast<ObjectFile<ELFT>>(*FileB);
    Elf_Sym_Range Syms = File.getLocalSymbols();
    for (const Elf_Sym &Sym : Syms) {
      ErrorOr<StringRef> SymName = Sym.getName(File.getStringTable());
      if (SymName && !shouldKeepInSymtab(*SymName))
        continue;
      auto *ESym = reinterpret_cast<Elf_Sym *>(Buf);
      Buf += sizeof(*ESym);
      ESym->st_name = (SymName) ? StrTabSec.getFileOff(*SymName) : 0;
      ESym->st_size = Sym.st_size;
      ESym->setBindingAndType(Sym.getBinding(), Sym.getType());
      uint32_t SecIndex = Sym.st_shndx;
      uintX_t VA = Sym.st_value;
      if (SecIndex == SHN_ABS) {
        ESym->st_shndx = SHN_ABS;
      } else {
        if (SecIndex == SHN_XINDEX)
          SecIndex = File.getObj().getExtendedSymbolTableIndex(
              &Sym, File.getSymbolTable(), File.getSymbolTableShndx());
        ArrayRef<InputSection<ELFT> *> Sections = File.getSections();
        const InputSection<ELFT> *Section = Sections[SecIndex];
        const OutputSection<ELFT> *Out = Section->getOutputSection();
        ESym->st_shndx = Out->getSectionIndex();
        VA += Out->getVA() + Section->getOutputSectionOff();
      }
      ESym->st_value = VA;
    }
  }
}

template <class ELFT>
void SymbolTableSection<ELFT>::writeGlobalSymbols(uint8_t *&Buf) {
  // Write the internal symbol table contents to the output symbol table
  // pointed by Buf.
  for (const std::pair<StringRef, Symbol *> &P : Table.getSymbols()) {
    StringRef Name = P.first;
    Symbol *Sym = P.second;
    SymbolBody *Body = Sym->Body;
    if (!includeInSymtab(*Body))
      continue;
    if (StrTabSec.isDynamic() && !includeInDynamicSymtab(*Body))
      continue;

    const auto &EBody = *cast<ELFSymbolBody<ELFT>>(Body);
    const Elf_Sym &InputSym = EBody.Sym;
    auto *ESym = reinterpret_cast<Elf_Sym *>(Buf);
    Buf += sizeof(*ESym);
    ESym->st_name = StrTabSec.getFileOff(Name);

    const OutputSection<ELFT> *Out = nullptr;
    const InputSection<ELFT> *Section = nullptr;

    switch (EBody.kind()) {
    case SymbolBody::DefinedSyntheticKind:
      Out = &cast<DefinedSynthetic<ELFT>>(Body)->Section;
      break;
    case SymbolBody::DefinedRegularKind:
      Section = &cast<DefinedRegular<ELFT>>(EBody).Section;
      break;
    case SymbolBody::DefinedCommonKind:
      Out = &BssSec;
      break;
    case SymbolBody::UndefinedKind:
    case SymbolBody::DefinedAbsoluteKind:
    case SymbolBody::SharedKind:
      break;
    case SymbolBody::LazyKind:
      llvm_unreachable("Lazy symbol got to output symbol table!");
    }

    ESym->setBindingAndType(InputSym.getBinding(), InputSym.getType());
    ESym->st_size = InputSym.st_size;
    ESym->setVisibility(EBody.getMostConstrainingVisibility());
    if (InputSym.isAbsolute()) {
      ESym->st_shndx = SHN_ABS;
      ESym->st_value = InputSym.st_value;
    }

    if (Section)
      Out = Section->getOutputSection();

    ESym->st_value = getSymVA(EBody, BssSec);

    if (Out)
      ESym->st_shndx = Out->getSectionIndex();
  }
}

namespace lld {
namespace elf2 {
template class OutputSectionBase<false>;
template class OutputSectionBase<true>;

template void OutputSectionBase<false>::writeHeaderTo<support::little>(
    ELFFile<ELFType<support::little, false>>::Elf_Shdr *SHdr);
template void OutputSectionBase<true>::writeHeaderTo<support::little>(
    ELFFile<ELFType<support::little, true>>::Elf_Shdr *SHdr);
template void OutputSectionBase<false>::writeHeaderTo<support::big>(
    ELFFile<ELFType<support::big, false>>::Elf_Shdr *SHdr);
template void OutputSectionBase<true>::writeHeaderTo<support::big>(
    ELFFile<ELFType<support::big, true>>::Elf_Shdr *SHdr);

template class GotSection<ELF32LE>;
template class GotSection<ELF32BE>;
template class GotSection<ELF64LE>;
template class GotSection<ELF64BE>;

template class PltSection<ELF32LE>;
template class PltSection<ELF32BE>;
template class PltSection<ELF64LE>;
template class PltSection<ELF64BE>;

template class RelocationSection<ELF32LE>;
template class RelocationSection<ELF32BE>;
template class RelocationSection<ELF64LE>;
template class RelocationSection<ELF64BE>;

template class InterpSection<false>;
template class InterpSection<true>;

template class HashTableSection<ELF32LE>;
template class HashTableSection<ELF32BE>;
template class HashTableSection<ELF64LE>;
template class HashTableSection<ELF64BE>;

template class DynamicSection<ELF32LE>;
template class DynamicSection<ELF32BE>;
template class DynamicSection<ELF64LE>;
template class DynamicSection<ELF64BE>;

template class OutputSection<ELF32LE>;
template class OutputSection<ELF32BE>;
template class OutputSection<ELF64LE>;
template class OutputSection<ELF64BE>;

template class StringTableSection<false>;
template class StringTableSection<true>;

template class SymbolTableSection<ELF32LE>;
template class SymbolTableSection<ELF32BE>;
template class SymbolTableSection<ELF64LE>;
template class SymbolTableSection<ELF64BE>;

template ELFFile<ELF32LE>::uintX_t
getSymVA(const ELFSymbolBody<ELF32LE> &, const OutputSection<ELF32LE> &);

template ELFFile<ELF32BE>::uintX_t
getSymVA(const ELFSymbolBody<ELF32BE> &, const OutputSection<ELF32BE> &);

template ELFFile<ELF64LE>::uintX_t
getSymVA(const ELFSymbolBody<ELF64LE> &, const OutputSection<ELF64LE> &);

template ELFFile<ELF64BE>::uintX_t
getSymVA(const ELFSymbolBody<ELF64BE> &, const OutputSection<ELF64BE> &);

template ELFFile<ELF32LE>::uintX_t
getLocalSymVA(const ELFFile<ELF32LE>::Elf_Sym *, const ObjectFile<ELF32LE> &);

template ELFFile<ELF32BE>::uintX_t
getLocalSymVA(const ELFFile<ELF32BE>::Elf_Sym *, const ObjectFile<ELF32BE> &);

template ELFFile<ELF64LE>::uintX_t
getLocalSymVA(const ELFFile<ELF64LE>::Elf_Sym *, const ObjectFile<ELF64LE> &);

template ELFFile<ELF64BE>::uintX_t
getLocalSymVA(const ELFFile<ELF64BE>::Elf_Sym *, const ObjectFile<ELF64BE> &);
}
}
