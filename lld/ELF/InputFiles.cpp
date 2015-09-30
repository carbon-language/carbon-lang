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

using namespace lld;
using namespace lld::elf2;

template <class ELFT> static uint16_t getEMachine(const ELFFileBase &B) {
  bool IsShared = isa<SharedFileBase>(B);
  if (IsShared)
    return cast<SharedFile<ELFT>>(B).getEMachine();
  return cast<ObjectFile<ELFT>>(B).getEMachine();
}

uint16_t ELFFileBase::getEMachine() const {
  switch (EKind) {
  case ELF32BEKind:
    return ::getEMachine<ELF32BE>(*this);
  case ELF32LEKind:
    return ::getEMachine<ELF32LE>(*this);
  case ELF64BEKind:
    return ::getEMachine<ELF64BE>(*this);
  case ELF64LEKind:
    return ::getEMachine<ELF64LE>(*this);
  }
  llvm_unreachable("Invalid kind");
}

bool ELFFileBase::isCompatibleWith(const ELFFileBase &Other) const {
  return getELFKind() == Other.getELFKind() &&
         getEMachine() == Other.getEMachine();
}

namespace {
class ECRAII {
  std::error_code EC;

public:
  std::error_code &getEC() { return EC; }
  ~ECRAII() { error(EC); }
};
}

template <class ELFT>
ELFData<ELFT>::ELFData(MemoryBufferRef MB)
    : ELFObj(MB.getBuffer(), ECRAII().getEC()) {}

template <class ELFT>
typename ELFData<ELFT>::Elf_Sym_Range
ELFData<ELFT>::getSymbolsHelper(bool Local) {
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

template <class ELFT>
typename ELFData<ELFT>::Elf_Sym_Range ELFData<ELFT>::getNonLocalSymbols() {
  if (!Symtab)
    return Elf_Sym_Range(nullptr, nullptr);
  ErrorOr<StringRef> StringTableOrErr = ELFObj.getStringTableForSymtab(*Symtab);
  error(StringTableOrErr.getError());
  StringTable = *StringTableOrErr;
  return getSymbolsHelper(false);
}

template <class ELFT>
ObjectFile<ELFT>::ObjectFile(MemoryBufferRef M)
    : ObjectFileBase(getStaticELFKind<ELFT>(), M), ELFData<ELFT>(M) {}

template <class ELFT>
typename ObjectFile<ELFT>::Elf_Sym_Range ObjectFile<ELFT>::getLocalSymbols() {
  return this->getSymbolsHelper(true);
}

template <class ELFT> void elf2::ObjectFile<ELFT>::parse() {
  // Read section and symbol tables.
  initializeSections();
  initializeSymbols();
}

template <class ELFT> void elf2::ObjectFile<ELFT>::initializeSections() {
  uint64_t Size = this->ELFObj.getNumSections();
  Sections.resize(Size);
  unsigned I = 0;
  for (const Elf_Shdr &Sec : this->ELFObj.sections()) {
    switch (Sec.sh_type) {
    case SHT_SYMTAB:
      this->Symtab = &Sec;
      break;
    case SHT_SYMTAB_SHNDX: {
      ErrorOr<ArrayRef<Elf_Word>> ErrorOrTable =
          this->ELFObj.getSHNDXTable(Sec);
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
      Sections[I] = new (Alloc) InputSection<ELFT>(this, &Sec);
      break;
    }
    ++I;
  }
}

template <class ELFT> void elf2::ObjectFile<ELFT>::initializeSymbols() {
  Elf_Sym_Range Syms = this->getNonLocalSymbols();
  uint32_t NumSymbols = std::distance(Syms.begin(), Syms.end());
  SymbolBodies.reserve(NumSymbols);
  for (const Elf_Sym &Sym : Syms)
    SymbolBodies.push_back(createSymbolBody(this->StringTable, &Sym));
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
    return new (Alloc) DefinedAbsolute<ELFT>(Name, *Sym);
  case SHN_UNDEF:
    return new (Alloc) Undefined<ELFT>(Name, *Sym);
  case SHN_COMMON:
    return new (Alloc) DefinedCommon<ELFT>(Name, *Sym);
  case SHN_XINDEX:
    SecIndex = this->ELFObj.getExtendedSymbolTableIndex(Sym, this->Symtab,
                                                        SymtabSHNDX);
    break;
  }

  if (SecIndex >= Sections.size() || (SecIndex != 0 && !Sections[SecIndex]))
    error("Invalid section index");

  switch (Sym->getBinding()) {
  default:
    error("unexpected binding");
  case STB_GLOBAL:
  case STB_WEAK:
    return new (Alloc) DefinedRegular<ELFT>(Name, *Sym, *Sections[SecIndex]);
  }
}

void ArchiveFile::parse() {
  ErrorOr<std::unique_ptr<Archive>> ArchiveOrErr = Archive::create(MB);
  error(ArchiveOrErr, "Failed to parse archive");
  File = std::move(*ArchiveOrErr);

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

template <class ELFT>
SharedFile<ELFT>::SharedFile(MemoryBufferRef M)
    : SharedFileBase(getStaticELFKind<ELFT>(), M), ELFData<ELFT>(M) {}

template <class ELFT> void SharedFile<ELFT>::parse() {
  for (const Elf_Shdr &Sec : this->ELFObj.sections()) {
    if (Sec.sh_type == SHT_DYNSYM) {
      this->Symtab = &Sec;
      break;
    }
  }

  Elf_Sym_Range Syms = this->getNonLocalSymbols();
  uint32_t NumSymbols = std::distance(Syms.begin(), Syms.end());
  SymbolBodies.reserve(NumSymbols);
  for (const Elf_Sym &Sym : Syms) {
    if (Sym.isUndefined())
      continue;

    ErrorOr<StringRef> NameOrErr = Sym.getName(this->StringTable);
    error(NameOrErr.getError());
    StringRef Name = *NameOrErr;

    SymbolBodies.emplace_back(Name, Sym);
  }
}

namespace lld {
namespace elf2 {

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
