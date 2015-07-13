//===- InputFiles.cpp -----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Chunks.h"
#include "Error.h"
#include "InputFiles.h"
#include "Writer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/LTO/LTOModule.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::support::endian;
using llvm::RoundUpToAlignment;
using llvm::sys::fs::identify_magic;
using llvm::sys::fs::file_magic;

using namespace lld;
using namespace lld::elfv2;

// Returns the last element of a path, which is supposed to be a filename.
static StringRef getBasename(StringRef Path) {
  size_t Pos = Path.rfind('\\');
  if (Pos == StringRef::npos)
    return Path;
  return Path.substr(Pos + 1);
}

// Returns a string in the format of "foo.obj" or "foo.obj(bar.lib)".
std::string InputFile::getShortName() {
  if (ParentName == "")
    return getName().lower();
  std::string Res =
      (getBasename(ParentName) + "(" + getBasename(getName()) + ")").str();
  return StringRef(Res).lower();
}

std::error_code ArchiveFile::parse() {
  // Parse a MemoryBufferRef as an archive file.
  auto ArchiveOrErr = Archive::create(MB);
  if (auto EC = ArchiveOrErr.getError())
    return EC;
  File = std::move(ArchiveOrErr.get());

  // Allocate a buffer for Lazy objects.
  size_t BufSize = File->getNumberOfSymbols() * sizeof(Lazy);
  Lazy *Buf = (Lazy *)Alloc.Allocate(BufSize, llvm::alignOf<Lazy>());

  // Read the symbol table to construct Lazy objects.
  uint32_t I = 0;
  for (const Archive::Symbol &Sym : File->symbols()) {
    SymbolBodies.push_back(new (&Buf[I++]) Lazy(this, Sym));
  }
  return std::error_code();
}

// Returns a buffer pointing to a member file containing a given symbol.
ErrorOr<MemoryBufferRef> ArchiveFile::getMember(const Archive::Symbol *Sym) {
  auto ItOrErr = Sym->getMember();
  if (auto EC = ItOrErr.getError())
    return EC;
  Archive::child_iterator It = ItOrErr.get();

  // Return an empty buffer if we have already returned the same buffer.
  const char *StartAddr = It->getBuffer().data();
  auto Pair = Seen.insert(StartAddr);
  if (!Pair.second)
    return MemoryBufferRef();
  return It->getMemoryBufferRef();
}

template <class ELFT> std::error_code elfv2::ObjectFile<ELFT>::parse() {
  // Parse a memory buffer as a ELF file.
  std::error_code EC;
  ELFObj = llvm::make_unique<ELFFile<ELFT>>(MB.getBuffer(), EC);

  if (EC) {
    llvm::errs() << getName() << " is not an ELF file.\n";
    return EC;
  }

  // Read section and symbol tables.
  if (EC = initializeChunks())
    return EC;
  return initializeSymbols();
}

template <class ELFT>
SymbolBody *elfv2::ObjectFile<ELFT>::getSymbolBody(uint32_t SymbolIndex) {
  return SparseSymbolBodies[SymbolIndex]->getReplacement();
}

static bool isIgnoredSectionType(unsigned Type) {
  switch (Type) {
  case SHT_NULL:
  case SHT_SYMTAB:
  case SHT_STRTAB:
  case SHT_RELA:
  case SHT_HASH:
  case SHT_DYNAMIC:
  case SHT_NOTE:
  case SHT_REL:
  case SHT_DYNSYM:
  case SHT_SYMTAB_SHNDX:
    return true;
  }
  return false;
}

template <class ELFT>
std::error_code elfv2::ObjectFile<ELFT>::initializeChunks() {
  auto Size = ELFObj->getNumSections();
  Chunks.reserve(Size);
  SparseChunks.resize(Size);
  int I = 0;
  for (auto &&Sec : ELFObj->sections()) {
    if (isIgnoredSectionType(Sec.sh_type) || Sec.sh_addralign == 0) {
      ++I;
      continue;
    }
    auto *C = new (Alloc) SectionChunk<ELFT>(this, &Sec, I);
    Chunks.push_back(C);
    SparseChunks[I] = C;
    ++I;
  }
  return std::error_code();
}

template <class ELFT>
std::error_code elfv2::ObjectFile<ELFT>::initializeSymbols() {
  auto Syms = ELFObj->symbols();
  Syms = ELFFile<ELFT>::Elf_Sym_Range(Syms.begin() + 1, Syms.end());
  auto NumSymbols = std::distance(Syms.begin(), Syms.end());
  SymbolBodies.reserve(NumSymbols + 1);
  SparseSymbolBodies.resize(NumSymbols + 1);
  int I = 1;
  for (auto &&Sym : Syms) {
    SymbolBody *Body = createSymbolBody(&Sym);
    if (Body) {
      SymbolBodies.push_back(Body);
      SparseSymbolBodies[I] = Body;
    }
    ++I;
  }

  return std::error_code();
}

template <class ELFT>
SymbolBody *elfv2::ObjectFile<ELFT>::createSymbolBody(const Elf_Sym *Sym) {
  StringRef Name;
  if (Sym->isUndefined()) {
    Name = *ELFObj->getStaticSymbolName(Sym);
    return new (Alloc) Undefined(Name);
  }
  if (Sym->isCommon()) {
    Chunk *C = new (Alloc) CommonChunk<ELFT>(Sym);
    Chunks.push_back(C);
    return new (Alloc) DefinedRegular<ELFT>(ELFObj.get(), Sym, C);
  }
  if (Sym->isAbsolute()) {
    Name = *ELFObj->getStaticSymbolName(Sym);
    return new (Alloc) DefinedAbsolute(Name, Sym->getValue());
  }
  if (Chunk *C = SparseChunks[Sym->st_shndx])
    return new (Alloc) DefinedRegular<ELFT>(ELFObj.get(), Sym, C);
  return nullptr;
}

std::error_code BitcodeFile::parse() {
  std::string Err;
  M.reset(LTOModule::createFromBuffer(MB.getBufferStart(), MB.getBufferSize(),
                                      llvm::TargetOptions(), Err));
  if (!Err.empty()) {
    llvm::errs() << Err << '\n';
    return make_error_code(LLDError::BrokenFile);
  }

  for (unsigned I = 0, E = M->getSymbolCount(); I != E; ++I) {
    lto_symbol_attributes Attrs = M->getSymbolAttributes(I);
    if ((Attrs & LTO_SYMBOL_SCOPE_MASK) == LTO_SYMBOL_SCOPE_INTERNAL)
      continue;

    StringRef SymName = M->getSymbolName(I);
    int SymbolDef = Attrs & LTO_SYMBOL_DEFINITION_MASK;
    if (SymbolDef == LTO_SYMBOL_DEFINITION_UNDEFINED) {
      SymbolBodies.push_back(new (Alloc) Undefined(SymName));
    } else {
      bool Replaceable = (SymbolDef == LTO_SYMBOL_DEFINITION_TENTATIVE ||
                          (Attrs & LTO_SYMBOL_COMDAT));
      SymbolBodies.push_back(new (Alloc) DefinedBitcode(SymName, Replaceable));
    }
  }

  return std::error_code();
}

template class elfv2::ObjectFile<llvm::object::ELF32LE>;
template class elfv2::ObjectFile<llvm::object::ELF32BE>;
template class elfv2::ObjectFile<llvm::object::ELF64LE>;
template class elfv2::ObjectFile<llvm::object::ELF64BE>;
