//===- InputFiles.cpp -----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "InputSection.h"
#include "Error.h"
#include "Symbols.h"
#include "llvm/ADT/STLExtras.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::sys::fs;

using namespace lld;
using namespace lld::elf2;

namespace {
class ECRAII {
  std::error_code EC;

public:
  std::error_code &getEC() { return EC; }
  ~ECRAII() { error(EC); }
};
}

template <class ELFT>
ELFFileBase<ELFT>::ELFFileBase(Kind K, ELFKind EKind, MemoryBufferRef M)
    : InputFile(K, M), EKind(EKind), ELFObj(MB.getBuffer(), ECRAII().getEC()) {}

template <class ELFT>
typename ELFFileBase<ELFT>::Elf_Sym_Range
ELFFileBase<ELFT>::getSymbolsHelper(bool Local) {
  if (!Symtab)
    return Elf_Sym_Range(nullptr, nullptr);
  Elf_Sym_Range Syms = ELFObj.symbols(Symtab);
  uint32_t NumSymbols = std::distance(Syms.begin(), Syms.end());
  uint32_t FirstNonLocal = Symtab->sh_info;
  if (FirstNonLocal > NumSymbols)
    error("Invalid sh_info in symbol table");
  if (!Local)
    return make_range(Syms.begin() + FirstNonLocal, Syms.end());
  // +1 to skip over dummy symbol.
  return make_range(Syms.begin() + 1, Syms.begin() + FirstNonLocal);
}

template <class ELFT> void ELFFileBase<ELFT>::initStringTable() {
  if (!Symtab)
    return;
  ErrorOr<StringRef> StringTableOrErr = ELFObj.getStringTableForSymtab(*Symtab);
  error(StringTableOrErr.getError());
  StringTable = *StringTableOrErr;
}

template <class ELFT>
typename ELFFileBase<ELFT>::Elf_Sym_Range
ELFFileBase<ELFT>::getNonLocalSymbols() {
  return getSymbolsHelper(false);
}

template <class ELFT>
ObjectFile<ELFT>::ObjectFile(MemoryBufferRef M)
    : ELFFileBase<ELFT>(Base::ObjectKind, getStaticELFKind<ELFT>(), M) {}

template <class ELFT>
typename ObjectFile<ELFT>::Elf_Sym_Range ObjectFile<ELFT>::getLocalSymbols() {
  return this->getSymbolsHelper(true);
}

template <class ELFT>
void elf2::ObjectFile<ELFT>::parse(DenseSet<StringRef> &Comdats) {
  // Read section and symbol tables.
  initializeSections(Comdats);
  initializeSymbols();
}

template <class ELFT>
StringRef ObjectFile<ELFT>::getShtGroupSignature(const Elf_Shdr &Sec) {
  const ELFFile<ELFT> &Obj = this->ELFObj;
  uint32_t SymtabdSectionIndex = Sec.sh_link;
  ErrorOr<const Elf_Shdr *> SecOrErr = Obj.getSection(SymtabdSectionIndex);
  error(SecOrErr);
  const Elf_Shdr *SymtabSec = *SecOrErr;
  uint32_t SymIndex = Sec.sh_info;
  const Elf_Sym *Sym = Obj.getSymbol(SymtabSec, SymIndex);
  ErrorOr<StringRef> StringTableOrErr = Obj.getStringTableForSymtab(*SymtabSec);
  error(StringTableOrErr);
  ErrorOr<StringRef> SignatureOrErr = Sym->getName(*StringTableOrErr);
  error(SignatureOrErr);
  return *SignatureOrErr;
}

template <class ELFT>
ArrayRef<typename ObjectFile<ELFT>::GroupEntryType>
ObjectFile<ELFT>::getShtGroupEntries(const Elf_Shdr &Sec) {
  const ELFFile<ELFT> &Obj = this->ELFObj;
  ErrorOr<ArrayRef<GroupEntryType>> EntriesOrErr =
      Obj.template getSectionContentsAsArray<GroupEntryType>(&Sec);
  error(EntriesOrErr.getError());
  ArrayRef<GroupEntryType> Entries = *EntriesOrErr;
  if (Entries.empty() || Entries[0] != GRP_COMDAT)
    error("Unsupported SHT_GROUP format");
  return Entries.slice(1);
}

template <class ELFT>
void elf2::ObjectFile<ELFT>::initializeSections(DenseSet<StringRef> &Comdats) {
  uint64_t Size = this->ELFObj.getNumSections();
  Sections.resize(Size);
  unsigned I = -1;
  const ELFFile<ELFT> &Obj = this->ELFObj;
  for (const Elf_Shdr &Sec : Obj.sections()) {
    ++I;
    if (Sections[I] == &InputSection<ELFT>::Discarded)
      continue;

    switch (Sec.sh_type) {
    case SHT_GROUP:
      Sections[I] = &InputSection<ELFT>::Discarded;
      if (Comdats.insert(getShtGroupSignature(Sec)).second)
        continue;
      for (GroupEntryType E : getShtGroupEntries(Sec)) {
        uint32_t SecIndex = E;
        if (SecIndex >= Size)
          error("Invalid section index in group");
        Sections[SecIndex] = &InputSection<ELFT>::Discarded;
      }
      break;
    case SHT_SYMTAB:
      this->Symtab = &Sec;
      break;
    case SHT_SYMTAB_SHNDX: {
      ErrorOr<ArrayRef<Elf_Word>> ErrorOrTable = Obj.getSHNDXTable(Sec);
      error(ErrorOrTable);
      SymtabSHNDX = *ErrorOrTable;
      break;
    }
    case SHT_STRTAB:
    case SHT_NULL:
      break;
    case SHT_RELA:
    case SHT_REL: {
      uint32_t RelocatedSectionIndex = Sec.sh_info;
      if (RelocatedSectionIndex >= Size)
        error("Invalid relocated section index");
      InputSection<ELFT> *RelocatedSection = Sections[RelocatedSectionIndex];
      if (!RelocatedSection)
        error("Unsupported relocation reference");
      RelocatedSection->RelocSections.push_back(&Sec);
      break;
    }
    default:
      Sections[I] = new (this->Alloc) InputSection<ELFT>(this, &Sec);
      break;
    }
  }
}

template <class ELFT> void elf2::ObjectFile<ELFT>::initializeSymbols() {
  this->initStringTable();
  Elf_Sym_Range Syms = this->getNonLocalSymbols();
  uint32_t NumSymbols = std::distance(Syms.begin(), Syms.end());
  this->SymbolBodies.reserve(NumSymbols);
  for (const Elf_Sym &Sym : Syms)
    this->SymbolBodies.push_back(createSymbolBody(this->StringTable, &Sym));
}

template <class ELFT>
SymbolBody *elf2::ObjectFile<ELFT>::createSymbolBody(StringRef StringTable,
                                                     const Elf_Sym *Sym) {
  ErrorOr<StringRef> NameOrErr = Sym->getName(StringTable);
  error(NameOrErr.getError());
  StringRef Name = *NameOrErr;

  uint32_t SecIndex = Sym->st_shndx;
  switch (SecIndex) {
  case SHN_ABS:
    return new (this->Alloc) DefinedAbsolute<ELFT>(Name, *Sym);
  case SHN_UNDEF:
    return new (this->Alloc) Undefined<ELFT>(Name, *Sym);
  case SHN_COMMON:
    return new (this->Alloc) DefinedCommon<ELFT>(Name, *Sym);
  case SHN_XINDEX:
    SecIndex = this->ELFObj.getExtendedSymbolTableIndex(Sym, this->Symtab,
                                                        SymtabSHNDX);
    break;
  }

  if (SecIndex >= Sections.size() || !SecIndex || !Sections[SecIndex])
    error("Invalid section index");

  switch (Sym->getBinding()) {
  default:
    error("unexpected binding");
  case STB_GLOBAL:
  case STB_WEAK:
  case STB_GNU_UNIQUE: {
    InputSection<ELFT> *Sec = Sections[SecIndex];
    if (Sec == &InputSection<ELFT>::Discarded)
      return new (this->Alloc) Undefined<ELFT>(Name, *Sym);
    return new (this->Alloc) DefinedRegular<ELFT>(Name, *Sym, *Sec);
  }
  }
}

static std::unique_ptr<Archive> openArchive(MemoryBufferRef MB) {
  ErrorOr<std::unique_ptr<Archive>> ArchiveOrErr = Archive::create(MB);
  error(ArchiveOrErr, "Failed to parse archive");
  return std::move(*ArchiveOrErr);
}

void ArchiveFile::parse() {
  File = openArchive(MB);

  // Allocate a buffer for Lazy objects.
  size_t NumSyms = File->getNumberOfSymbols();
  LazySymbols.reserve(NumSyms);

  // Read the symbol table to construct Lazy objects.
  for (const Archive::Symbol &Sym : File->symbols())
    LazySymbols.emplace_back(this, Sym);
}

// Returns a buffer pointing to a member file containing a given symbol.
MemoryBufferRef ArchiveFile::getMember(const Archive::Symbol *Sym) {
  ErrorOr<Archive::child_iterator> ItOrErr = Sym->getMember();
  error(ItOrErr,
        Twine("Could not get the member for symbol ") + Sym->getName());
  Archive::child_iterator It = *ItOrErr;

  if (!Seen.insert(It->getChildOffset()).second)
    return MemoryBufferRef();

  ErrorOr<MemoryBufferRef> Ret = It->getMemoryBufferRef();
  error(Ret, Twine("Could not get the buffer for the member defining symbol ") +
                 Sym->getName());
  return *Ret;
}

std::vector<MemoryBufferRef> ArchiveFile::getMembers() {
  File = openArchive(MB);

  std::vector<MemoryBufferRef> Result;
  for (const Archive::Child &Child : File->children()) {
    ErrorOr<MemoryBufferRef> MbOrErr = Child.getMemoryBufferRef();
    error(MbOrErr,
          Twine("Could not get the buffer for a child of the archive ") +
              File->getFileName());
    Result.push_back(MbOrErr.get());
  }
  return Result;
}

template <class ELFT>
SharedFile<ELFT>::SharedFile(MemoryBufferRef M)
    : ELFFileBase<ELFT>(Base::SharedKind, getStaticELFKind<ELFT>(), M) {
  AsNeeded = Config->AsNeeded;
}

template <class ELFT> void SharedFile<ELFT>::parseSoName() {
  typedef typename ELFFile<ELFT>::Elf_Dyn Elf_Dyn;
  typedef typename ELFFile<ELFT>::uintX_t uintX_t;
  const Elf_Shdr *DynamicSec = nullptr;

  const ELFFile<ELFT> Obj = this->ELFObj;
  for (const Elf_Shdr &Sec : Obj.sections()) {
    uint32_t Type = Sec.sh_type;
    if (Type == SHT_DYNSYM)
      this->Symtab = &Sec;
    else if (Type == SHT_DYNAMIC)
      DynamicSec = &Sec;
  }

  this->initStringTable();
  this->SoName = this->getName();

  if (DynamicSec) {
    auto *Begin =
        reinterpret_cast<const Elf_Dyn *>(Obj.base() + DynamicSec->sh_offset);
    const Elf_Dyn *End = Begin + DynamicSec->sh_size / sizeof(Elf_Dyn);

    for (const Elf_Dyn &Dyn : make_range(Begin, End)) {
      if (Dyn.d_tag == DT_SONAME) {
        uintX_t Val = Dyn.getVal();
        if (Val >= this->StringTable.size())
          error("Invalid DT_SONAME entry");
        this->SoName = StringRef(this->StringTable.data() + Val);
        break;
      }
    }
  }
}

template <class ELFT> void SharedFile<ELFT>::parse() {
  Elf_Sym_Range Syms = this->getNonLocalSymbols();
  uint32_t NumSymbols = std::distance(Syms.begin(), Syms.end());
  SymbolBodies.reserve(NumSymbols);
  for (const Elf_Sym &Sym : Syms) {
    if (Sym.isUndefined())
      continue;

    ErrorOr<StringRef> NameOrErr = Sym.getName(this->StringTable);
    error(NameOrErr.getError());
    StringRef Name = *NameOrErr;

    SymbolBodies.emplace_back(this, Name, Sym);
  }
}

namespace lld {
namespace elf2 {
template class ELFFileBase<llvm::object::ELF32LE>;
template class ELFFileBase<llvm::object::ELF32BE>;
template class ELFFileBase<llvm::object::ELF64LE>;
template class ELFFileBase<llvm::object::ELF64BE>;

template class elf2::ObjectFile<llvm::object::ELF32LE>;
template class elf2::ObjectFile<llvm::object::ELF32BE>;
template class elf2::ObjectFile<llvm::object::ELF64LE>;
template class elf2::ObjectFile<llvm::object::ELF64BE>;

template class elf2::SharedFile<llvm::object::ELF32LE>;
template class elf2::SharedFile<llvm::object::ELF32BE>;
template class elf2::SharedFile<llvm::object::ELF64LE>;
template class elf2::SharedFile<llvm::object::ELF64BE>;
}
}
