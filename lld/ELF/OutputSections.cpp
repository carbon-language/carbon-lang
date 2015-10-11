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
using namespace llvm::support::endian;
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
  Sym->GotIndex = Entries.size();
  Entries.push_back(Sym);
}

template <class ELFT>
typename GotSection<ELFT>::uintX_t
GotSection<ELFT>::getEntryAddr(const SymbolBody &B) const {
  return this->getVA() + B.GotIndex * this->getAddrSize();
}

template <class ELFT> void GotSection<ELFT>::writeTo(uint8_t *Buf) {
  for (const SymbolBody *B : Entries) {
    uint8_t *Entry = Buf;
    Buf += sizeof(uintX_t);
    if (canBePreempted(B))
      continue; // The dynamic linker will take care of it.
    uintX_t VA = getSymVA<ELFT>(*B);
    write<uintX_t, ELFT::TargetEndianness, sizeof(uintX_t)>(Entry, VA);
  }
}

template <class ELFT>
PltSection<ELFT>::PltSection()
    : OutputSectionBase<ELFT::Is64Bits>(".plt", llvm::ELF::SHT_PROGBITS,
                                        llvm::ELF::SHF_ALLOC |
                                            llvm::ELF::SHF_EXECINSTR) {
  this->Header.sh_addralign = 16;
}

template <class ELFT> void PltSection<ELFT>::writeTo(uint8_t *Buf) {
  uint8_t *Start = Buf;
  for (const SymbolBody *E : Entries) {
    uint64_t Got = Out<ELFT>::Got->getEntryAddr(*E);
    uint64_t Plt = Buf - Start + this->getVA();
    Target->writePltEntry(Buf, Got, Plt);
    Buf += Target->getPltEntrySize();
  }
}

template <class ELFT> void PltSection<ELFT>::addEntry(SymbolBody *Sym) {
  Sym->PltIndex = Entries.size();
  Entries.push_back(Sym);
}

template <class ELFT>
typename PltSection<ELFT>::uintX_t
PltSection<ELFT>::getEntryAddr(const SymbolBody &B) const {
  return this->getVA() + B.PltIndex * Target->getPltEntrySize();
}

template <class ELFT>
void PltSection<ELFT>::finalize() {
  this->Header.sh_size = Entries.size() * Target->getPltEntrySize();
}

template <class ELFT>
RelocationSection<ELFT>::RelocationSection(bool IsRela)
    : OutputSectionBase<ELFT::Is64Bits>(IsRela ? ".rela.dyn" : ".rel.dyn",
                                        IsRela ? llvm::ELF::SHT_RELA
                                               : llvm::ELF::SHT_REL,
                                        llvm::ELF::SHF_ALLOC),
      IsRela(IsRela) {
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
    OutputSection<ELFT> *OutSec = C.getOutputSection();
    uint32_t SymIndex = RI.getSymbol(IsMips64EL);
    const ObjectFile<ELFT> &File = *C.getFile();
    SymbolBody *Body = File.getSymbolBody(SymIndex);
    const ELFFile<ELFT> &Obj = File.getObj();
    if (Body)
      Body = Body->repl();

    uint32_t Type = RI.getType(IsMips64EL);

    bool CanBePreempted = canBePreempted(Body);
    uintX_t Addend = 0;
    if (!CanBePreempted) {
      if (IsRela) {
        if (Body)
          Addend += getSymVA<ELFT>(cast<ELFSymbolBody<ELFT>>(*Body));
        else
          Addend += getLocalSymVA(
              Obj.getRelocationSymbol(&RI, File.getSymbolTable()), File);
      }
      P->setSymbolAndType(0, Target->getRelativeReloc(), IsMips64EL);
    }

    if (Body && Target->relocNeedsGot(Type, *Body)) {
      P->r_offset = Out<ELFT>::Got->getEntryAddr(*Body);
      if (CanBePreempted)
        P->setSymbolAndType(Body->getDynamicSymbolTableIndex(),
                            Target->getGotReloc(), IsMips64EL);
    } else {
      if (IsRela)
        Addend += static_cast<const Elf_Rela &>(RI).r_addend;
      P->r_offset = RI.r_offset + C.getOutputSectionOff() + OutSec->getVA();
      if (CanBePreempted)
        P->setSymbolAndType(Body->getDynamicSymbolTableIndex(), Type,
                            IsMips64EL);
    }

    if (IsRela)
      static_cast<Elf_Rela *>(P)->r_addend = Addend;
  }
}

template <class ELFT> void RelocationSection<ELFT>::finalize() {
  this->Header.sh_link = Out<ELFT>::DynSymTab->getSectionIndex();
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
HashTableSection<ELFT>::HashTableSection()
    : OutputSectionBase<ELFT::Is64Bits>(".hash", llvm::ELF::SHT_HASH,
                                        llvm::ELF::SHF_ALLOC) {
  this->Header.sh_entsize = sizeof(Elf_Word);
  this->Header.sh_addralign = sizeof(Elf_Word);
}

template <class ELFT> void HashTableSection<ELFT>::addSymbol(SymbolBody *S) {
  StringRef Name = S->getName();
  Out<ELFT>::DynSymTab->addSymbol(Name);
  Hashes.push_back(hash(Name));
  S->setDynamicSymbolTableIndex(Hashes.size());
}

template <class ELFT> void HashTableSection<ELFT>::finalize() {
  this->Header.sh_link = Out<ELFT>::DynSymTab->getSectionIndex();

  assert(Out<ELFT>::DynSymTab->getNumSymbols() == Hashes.size() + 1);
  unsigned NumEntries = 2;                 // nbucket and nchain.
  NumEntries += Out<ELFT>::DynSymTab->getNumSymbols(); // The chain entries.

  // Create as many buckets as there are symbols.
  // FIXME: This is simplistic. We can try to optimize it, but implementing
  // support for SHT_GNU_HASH is probably even more profitable.
  NumEntries += Out<ELFT>::DynSymTab->getNumSymbols();
  this->Header.sh_size = NumEntries * sizeof(Elf_Word);
}

template <class ELFT> void HashTableSection<ELFT>::writeTo(uint8_t *Buf) {
  unsigned NumSymbols = Out<ELFT>::DynSymTab->getNumSymbols();
  auto *P = reinterpret_cast<Elf_Word *>(Buf);
  *P++ = NumSymbols; // nbucket
  *P++ = NumSymbols; // nchain

  Elf_Word *Buckets = P;
  Elf_Word *Chains = P + NumSymbols;

  for (unsigned I = 1; I < NumSymbols; ++I) {
    uint32_t Hash = Hashes[I - 1] % NumSymbols;
    Chains[I] = Buckets[Hash];
    Buckets[Hash] = I;
  }
}

template <class ELFT>
DynamicSection<ELFT>::DynamicSection(SymbolTable<ELFT> &SymTab)
    : OutputSectionBase<ELFT::Is64Bits>(".dynamic", llvm::ELF::SHT_DYNAMIC,
                                        llvm::ELF::SHF_ALLOC |
                                            llvm::ELF::SHF_WRITE),
      SymTab(SymTab) {
  typename Base::HeaderT &Header = this->Header;
  Header.sh_addralign = ELFT::Is64Bits ? 8 : 4;
  Header.sh_entsize = ELFT::Is64Bits ? 16 : 8;
}

template <class ELFT> void DynamicSection<ELFT>::finalize() {
  if (this->Header.sh_size)
    return; // Already finalized.

  typename Base::HeaderT &Header = this->Header;
  Header.sh_link = Out<ELFT>::DynStrTab->getSectionIndex();

  unsigned NumEntries = 0;
  if (Out<ELFT>::RelaDyn->hasRelocs()) {
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
    ++NumEntries; // DT_RUNPATH / DT_RPATH
    Out<ELFT>::DynStrTab->add(Config->RPath);
  }

  if (!Config->SoName.empty()) {
    ++NumEntries; // DT_SONAME
    Out<ELFT>::DynStrTab->add(Config->SoName);
  }

  if (PreInitArraySec)
    NumEntries += 2;
  if (InitArraySec)
    NumEntries += 2;
  if (FiniArraySec)
    NumEntries += 2;

  for (const std::unique_ptr<SharedFile<ELFT>> &F : SymTab.getSharedFiles()) {
    if (!F->isNeeded())
      continue;
    Out<ELFT>::DynStrTab->add(F->getSoName());
    ++NumEntries;
  }

  if (Symbol *S = SymTab.getSymbols().lookup(Config->Init))
    InitSym = dyn_cast<ELFSymbolBody<ELFT>>(S->Body);
  if (Symbol *S = SymTab.getSymbols().lookup(Config->Fini))
    FiniSym = dyn_cast<ELFSymbolBody<ELFT>>(S->Body);
  if (InitSym)
    ++NumEntries; // DT_INIT
  if (FiniSym)
    ++NumEntries; // DT_FINI
  if (Config->ZNow)
    ++NumEntries; // DT_FLAGS_1

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

  if (Out<ELFT>::RelaDyn->hasRelocs()) {
    bool IsRela = Out<ELFT>::RelaDyn->isRela();
    WritePtr(IsRela ? DT_RELA : DT_REL, Out<ELFT>::RelaDyn->getVA());
    WriteVal(IsRela ? DT_RELASZ : DT_RELSZ, Out<ELFT>::RelaDyn->getSize());
    WriteVal(IsRela ? DT_RELAENT : DT_RELENT,
             IsRela ? sizeof(Elf_Rela) : sizeof(Elf_Rel));
  }

  WritePtr(DT_SYMTAB, Out<ELFT>::DynSymTab->getVA());
  WritePtr(DT_SYMENT, sizeof(Elf_Sym));
  WritePtr(DT_STRTAB, Out<ELFT>::DynStrTab->getVA());
  WriteVal(DT_STRSZ, Out<ELFT>::DynStrTab->data().size());
  WritePtr(DT_HASH, Out<ELFT>::HashTab->getVA());

  if (!Config->RPath.empty())

    // If --enable-new-dtags is set lld emits DT_RUNPATH
    // instead of DT_RPATH. The two tags are functionally
    // equivalent except for the following:
    // - DT_RUNPATH is searched after LD_LIBRARY_PATH, while
    // DT_RPATH is searched before.
    // - DT_RUNPATH is used only to search for direct
    // dependencies of the object it's contained in, while
    // DT_RPATH is used for indirect dependencies as well.
    WriteVal(Config->EnableNewDtags ? DT_RUNPATH : DT_RPATH,
             Out<ELFT>::DynStrTab->getFileOff(Config->RPath));

  if (!Config->SoName.empty())
    WriteVal(DT_SONAME, Out<ELFT>::DynStrTab->getFileOff(Config->SoName));

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

  for (const std::unique_ptr<SharedFile<ELFT>> &F : SymTab.getSharedFiles())
    if (F->isNeeded())
      WriteVal(DT_NEEDED, Out<ELFT>::DynStrTab->getFileOff(F->getSoName()));

  if (InitSym)
    WritePtr(DT_INIT, getSymVA<ELFT>(*InitSym));
  if (FiniSym)
    WritePtr(DT_FINI, getSymVA<ELFT>(*FiniSym));

  if (Config->ZNow)
    WriteVal(DT_FLAGS_1, DF_1_NOW);

  WriteVal(DT_NULL, 0);
}

template <class ELFT>
OutputSection<ELFT>::OutputSection(StringRef Name, uint32_t sh_type,
                                   uintX_t sh_flags)
    : OutputSectionBase<ELFT::Is64Bits>(Name, sh_type, sh_flags) {}

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
typename ELFFile<ELFT>::uintX_t lld::elf2::getSymVA(const SymbolBody &S) {
  switch (S.kind()) {
  case SymbolBody::DefinedSyntheticKind: {
    auto &D = cast<DefinedSynthetic<ELFT>>(S);
    return D.Section.getVA() + D.Sym.st_value;
  }
  case SymbolBody::DefinedAbsoluteKind:
    return cast<DefinedAbsolute<ELFT>>(S).Sym.st_value;
  case SymbolBody::DefinedRegularKind: {
    const auto &DR = cast<DefinedRegular<ELFT>>(S);
    const InputSection<ELFT> *SC = &DR.Section;
    OutputSection<ELFT> *OS = SC->getOutputSection();
    return OS->getVA() + SC->getOutputSectionOff() + DR.Sym.st_value;
  }
  case SymbolBody::DefinedCommonKind:
    return Out<ELFT>::Bss->getVA() + cast<DefinedCommon<ELFT>>(S).OffsetInBSS;
  case SymbolBody::SharedKind:
  case SymbolBody::UndefinedKind:
    return 0;
  case SymbolBody::LazyKind:
    assert(S.isUsedInRegularObj() && "Lazy symbol reached writer");
    return 0;
  }
  llvm_unreachable("Invalid symbol kind");
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

  // According to the ELF spec reference to a local symbol from outside
  // the group are not allowed. Unfortunately .eh_frame breaks that rule
  // and must be treated specially. For now we just replace the symbol with
  // 0.
  if (Section == &InputSection<ELFT>::Discarded)
    return 0;

  OutputSection<ELFT> *OutSec = Section->getOutputSection();
  return OutSec->getVA() + Section->getOutputSectionOff() + Sym->st_value;
}

bool lld::elf2::canBePreempted(const SymbolBody *Body) {
  if (!Body)
    return false;
  if (Body->isShared())
    return true;
  if (Body->isUndefined() && !Body->isWeak())
    return true;
  if (!Config->Shared)
    return false;
  return Body->getMostConstrainingVisibility() == STV_DEFAULT;
}

template <class ELFT> void OutputSection<ELFT>::writeTo(uint8_t *Buf) {
  for (InputSection<ELFT> *C : Sections)
    C->writeTo(Buf);
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

template <class ELFT> bool lld::elf2::includeInSymtab(const SymbolBody &B) {
  if (!B.isUsedInRegularObj())
    return false;

  // Don't include synthetic symbols like __init_array_start in every output.
  if (auto *U = dyn_cast<DefinedAbsolute<ELFT>>(&B))
    if (&U->Sym == &DefinedAbsolute<ELFT>::IgnoreUndef)
      return false;

  return true;
}

bool lld::elf2::includeInDynamicSymtab(const SymbolBody &B) {
  uint8_t V = B.getMostConstrainingVisibility();
  if (V != STV_DEFAULT && V != STV_PROTECTED)
    return false;

  if (Config->ExportDynamic || Config->Shared)
    return true;
  return B.isUsedInDynamicReloc();
}

template <class ELFT>
bool lld::elf2::shouldKeepInSymtab(const ObjectFile<ELFT> &File,
                                   StringRef SymName,
                                   const typename ELFFile<ELFT>::Elf_Sym &Sym) {
  if (Sym.getType() == STT_SECTION)
    return false;

  // If sym references a section in a discarded group, don't keep it.
  uint32_t SecIndex = Sym.st_shndx;
  if (SecIndex != SHN_ABS) {
    if (SecIndex == SHN_XINDEX)
      SecIndex = File.getObj().getExtendedSymbolTableIndex(
          &Sym, File.getSymbolTable(), File.getSymbolTableShndx());
    ArrayRef<InputSection<ELFT> *> Sections = File.getSections();
    const InputSection<ELFT> *Section = Sections[SecIndex];
    if (Section == &InputSection<ELFT>::Discarded)
      return false;
  }

  if (Config->DiscardNone)
    return true;

  // ELF defines dynamic locals as symbols which name starts with ".L".
  return !(Config->DiscardLocals && SymName.startswith(".L"));
}

template <class ELFT>
SymbolTableSection<ELFT>::SymbolTableSection(
    SymbolTable<ELFT> &Table, StringTableSection<ELFT::Is64Bits> &StrTabSec)
    : OutputSectionBase<ELFT::Is64Bits>(
          StrTabSec.isDynamic() ? ".dynsym" : ".symtab",
          StrTabSec.isDynamic() ? llvm::ELF::SHT_DYNSYM : llvm::ELF::SHT_SYMTAB,
          StrTabSec.isDynamic() ? (uintX_t)llvm::ELF::SHF_ALLOC : 0),
      Table(Table), StrTabSec(StrTabSec) {
  typedef OutputSectionBase<ELFT::Is64Bits> Base;
  typename Base::HeaderT &Header = this->Header;

  Header.sh_entsize = sizeof(Elf_Sym);
  Header.sh_addralign = ELFT::Is64Bits ? 8 : 4;
}

template <class ELFT> void SymbolTableSection<ELFT>::finalize() {
  this->Header.sh_size = getNumSymbols() * sizeof(Elf_Sym);
  this->Header.sh_link = StrTabSec.getSectionIndex();
  this->Header.sh_info = NumLocals + 1;
}

template <class ELFT>
void SymbolTableSection<ELFT>::addSymbol(StringRef Name, bool isLocal) {
  StrTabSec.add(Name);
  ++NumVisible;
  if (isLocal)
    ++NumLocals;
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
  for (const std::unique_ptr<ObjectFile<ELFT>> &File : Table.getObjectFiles()) {
    Elf_Sym_Range Syms = File->getLocalSymbols();
    for (const Elf_Sym &Sym : Syms) {
      ErrorOr<StringRef> SymNameOrErr = Sym.getName(File->getStringTable());
      error(SymNameOrErr);
      StringRef SymName = *SymNameOrErr;
      if (!shouldKeepInSymtab<ELFT>(*File, SymName, Sym))
        continue;

      auto *ESym = reinterpret_cast<Elf_Sym *>(Buf);
      Buf += sizeof(*ESym);
      ESym->st_name = StrTabSec.getFileOff(SymName);
      ESym->st_size = Sym.st_size;
      ESym->setBindingAndType(Sym.getBinding(), Sym.getType());
      uint32_t SecIndex = Sym.st_shndx;
      uintX_t VA = Sym.st_value;
      if (SecIndex == SHN_ABS) {
        ESym->st_shndx = SHN_ABS;
      } else {
        if (SecIndex == SHN_XINDEX)
          SecIndex = File->getObj().getExtendedSymbolTableIndex(
              &Sym, File->getSymbolTable(), File->getSymbolTableShndx());
        ArrayRef<InputSection<ELFT> *> Sections = File->getSections();
        const InputSection<ELFT> *Section = Sections[SecIndex];
        const OutputSection<ELFT> *OutSec = Section->getOutputSection();
        ESym->st_shndx = OutSec->getSectionIndex();
        VA += OutSec->getVA() + Section->getOutputSectionOff();
      }
      ESym->st_value = VA;
    }
  }
}

template <class ELFT>
void SymbolTableSection<ELFT>::writeGlobalSymbols(uint8_t *&Buf) {
  // Write the internal symbol table contents to the output symbol table
  // pointed by Buf.
  uint8_t *Start = Buf;
  for (const std::pair<StringRef, Symbol *> &P : Table.getSymbols()) {
    StringRef Name = P.first;
    Symbol *Sym = P.second;
    SymbolBody *Body = Sym->Body;
    if (!includeInSymtab<ELFT>(*Body))
      continue;
    if (StrTabSec.isDynamic() && !includeInDynamicSymtab(*Body))
      continue;

    auto *ESym = reinterpret_cast<Elf_Sym *>(Buf);
    Buf += sizeof(*ESym);

    ESym->st_name = StrTabSec.getFileOff(Name);

    const OutputSection<ELFT> *OutSec = nullptr;
    const InputSection<ELFT> *Section = nullptr;

    switch (Body->kind()) {
    case SymbolBody::DefinedSyntheticKind:
      OutSec = &cast<DefinedSynthetic<ELFT>>(Body)->Section;
      break;
    case SymbolBody::DefinedRegularKind:
      Section = &cast<DefinedRegular<ELFT>>(Body)->Section;
      break;
    case SymbolBody::DefinedCommonKind:
      OutSec = Out<ELFT>::Bss;
      break;
    case SymbolBody::UndefinedKind:
    case SymbolBody::DefinedAbsoluteKind:
    case SymbolBody::SharedKind:
    case SymbolBody::LazyKind:
      break;
    }

    unsigned char Binding = Body->isWeak() ? STB_WEAK : STB_GLOBAL;
    unsigned char Type = STT_NOTYPE;
    uintX_t Size = 0;
    if (const auto *EBody = dyn_cast<ELFSymbolBody<ELFT>>(Body)) {
      const Elf_Sym &InputSym = EBody->Sym;
      Binding = InputSym.getBinding();
      Type = InputSym.getType();
      Size = InputSym.st_size;
    }

    unsigned char Visibility = Body->getMostConstrainingVisibility();
    if (Visibility != STV_DEFAULT && Visibility != STV_PROTECTED)
      Binding = STB_LOCAL;

    ESym->setBindingAndType(Binding, Type);
    ESym->st_size = Size;
    ESym->setVisibility(Visibility);
    ESym->st_value = getSymVA<ELFT>(*Body);

    if (Section)
      OutSec = Section->getOutputSection();

    if (isa<DefinedAbsolute<ELFT>>(Body))
      ESym->st_shndx = SHN_ABS;
    else if (OutSec)
      ESym->st_shndx = OutSec->getSectionIndex();
  }
  if (!StrTabSec.isDynamic())
    std::stable_sort(
        reinterpret_cast<Elf_Sym *>(Start), reinterpret_cast<Elf_Sym *>(Buf),
        [](const Elf_Sym &A, const Elf_Sym &B) -> bool {
          return A.getBinding() == STB_LOCAL && B.getBinding() != STB_LOCAL;
        });
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

template ELFFile<ELF32LE>::uintX_t getSymVA<ELF32LE>(const SymbolBody &);
template ELFFile<ELF32BE>::uintX_t getSymVA<ELF32BE>(const SymbolBody &);
template ELFFile<ELF64LE>::uintX_t getSymVA<ELF64LE>(const SymbolBody &);
template ELFFile<ELF64BE>::uintX_t getSymVA<ELF64BE>(const SymbolBody &);

template ELFFile<ELF32LE>::uintX_t
getLocalSymVA(const ELFFile<ELF32LE>::Elf_Sym *, const ObjectFile<ELF32LE> &);

template ELFFile<ELF32BE>::uintX_t
getLocalSymVA(const ELFFile<ELF32BE>::Elf_Sym *, const ObjectFile<ELF32BE> &);

template ELFFile<ELF64LE>::uintX_t
getLocalSymVA(const ELFFile<ELF64LE>::Elf_Sym *, const ObjectFile<ELF64LE> &);

template ELFFile<ELF64BE>::uintX_t
getLocalSymVA(const ELFFile<ELF64BE>::Elf_Sym *, const ObjectFile<ELF64BE> &);

template bool includeInSymtab<ELF32LE>(const SymbolBody &);
template bool includeInSymtab<ELF32BE>(const SymbolBody &);
template bool includeInSymtab<ELF64LE>(const SymbolBody &);
template bool includeInSymtab<ELF64BE>(const SymbolBody &);

template bool shouldKeepInSymtab<ELF32LE>(const ObjectFile<ELF32LE> &,
                                          StringRef,
                                          const ELFFile<ELF32LE>::Elf_Sym &);
template bool shouldKeepInSymtab<ELF32BE>(const ObjectFile<ELF32BE> &,
                                          StringRef,
                                          const ELFFile<ELF32BE>::Elf_Sym &);
template bool shouldKeepInSymtab<ELF64LE>(const ObjectFile<ELF64LE> &,
                                          StringRef,
                                          const ELFFile<ELF64LE>::Elf_Sym &);
template bool shouldKeepInSymtab<ELF64BE>(const ObjectFile<ELF64BE> &,
                                          StringRef,
                                          const ELFFile<ELF64BE>::Elf_Sym &);
}
}
