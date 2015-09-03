//===- InputFiles.cpp -----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "Chunks.h"
#include "Error.h"
#include "Symbols.h"
#include "llvm/ADT/STLExtras.h"

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

static uint16_t getEMachine(const ELFFileBase &B) {
  ELFKind K = B.getELFKind();
  switch (K) {
  case ELF32BEKind:
    return getEMachine<ELF32BE>(B);
  case ELF32LEKind:
    return getEMachine<ELF32LE>(B);
  case ELF64BEKind:
    return getEMachine<ELF64BE>(B);
  case ELF64LEKind:
    return getEMachine<ELF64LE>(B);
  }
  llvm_unreachable("Invalid kind");
}

bool ELFFileBase::isCompatibleWith(const ELFFileBase &Other) const {
  return getELFKind() == Other.getELFKind() &&
         getEMachine(*this) == getEMachine(Other);
}

template <class ELFT> void ELFData<ELFT>::openELF(MemoryBufferRef MB) {
  // Parse a memory buffer as a ELF file.
  std::error_code EC;
  ELFObj = llvm::make_unique<ELFFile<ELFT>>(MB.getBuffer(), EC);
  error(EC);
}

template <class ELFT> void elf2::ObjectFile<ELFT>::parse() {
  this->openELF(MB);

  // Read section and symbol tables.
  initializeChunks();
  initializeSymbols();
}

template <class ELFT> void elf2::ObjectFile<ELFT>::initializeChunks() {
  uint64_t Size = this->ELFObj->getNumSections();
  Chunks.resize(Size);
  unsigned I = 0;
  for (const Elf_Shdr &Sec : this->ELFObj->sections()) {
    switch (Sec.sh_type) {
    case SHT_SYMTAB:
      Symtab = &Sec;
      break;
    case SHT_SYMTAB_SHNDX: {
      ErrorOr<ArrayRef<Elf_Word>> ErrorOrTable =
          this->ELFObj->getSHNDXTable(Sec);
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
      SectionChunk<ELFT> *RelocatedSection = Chunks[RelocatedSectionIndex];
      if (!RelocatedSection)
        error("Unsupported relocation reference");
      RelocatedSection->RelocSections.push_back(&Sec);
      break;
    }
    default:
      Chunks[I] = new (Alloc) SectionChunk<ELFT>(this, &Sec);
      break;
    }
    ++I;
  }
}

template <class ELFT> void elf2::ObjectFile<ELFT>::initializeSymbols() {
  ErrorOr<StringRef> StringTableOrErr =
      this->ELFObj->getStringTableForSymtab(*Symtab);
  error(StringTableOrErr.getError());
  StringRef StringTable = *StringTableOrErr;

  Elf_Sym_Range Syms = this->ELFObj->symbols(Symtab);
  uint32_t NumSymbols = std::distance(Syms.begin(), Syms.end());
  uint32_t FirstNonLocal = Symtab->sh_info;
  if (FirstNonLocal > NumSymbols)
    error("Invalid sh_info in symbol table");
  Syms = llvm::make_range(Syms.begin() + FirstNonLocal, Syms.end());
  SymbolBodies.reserve(NumSymbols - FirstNonLocal);
  for (const Elf_Sym &Sym : Syms)
    SymbolBodies.push_back(createSymbolBody(StringTable, &Sym));
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
    SecIndex =
        this->ELFObj->getExtendedSymbolTableIndex(Sym, Symtab, SymtabSHNDX);
    break;
  }

  if (SecIndex >= Chunks.size() ||
      (SecIndex != 0 && !Chunks[SecIndex]))
    error("Invalid section index");

  switch (Sym->getBinding()) {
  default:
    error("unexpected binding");
  case STB_GLOBAL:
  case STB_WEAK:
    return new (Alloc) DefinedRegular<ELFT>(Name, *Sym, *Chunks[SecIndex]);
  }
}

template <class ELFT> void SharedFile<ELFT>::parse() { this->openELF(MB); }

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
