//===- SymbolTable.cpp ----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Config.h"
#include "Driver.h"
#include "Error.h"
#include "SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/LTO/LTOCodeGenerator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

using namespace lld;
using namespace lld::elfv2;

template <class ELFT> SymbolTable<ELFT>::SymbolTable() {
  if (!Config->EntryName.empty())
    resolve(new (Alloc) Undefined(Config->EntryName));
}

template <class ELFT>
std::error_code SymbolTable<ELFT>::addFile(std::unique_ptr<InputFile> File) {
  if (auto EC = File->parse())
    return EC;
  InputFile *FileP = File.release();
  if (auto *P = dyn_cast<ObjectFile<ELFT>>(FileP))
    return addObject(P);
  if (auto *P = dyn_cast<ArchiveFile>(FileP))
    return addArchive(P);
  if (auto *P = dyn_cast<BitcodeFile>(FileP))
    return addBitcode(P);
  llvm_unreachable("Unknown file");
}

template <class ELFT>
std::error_code SymbolTable<ELFT>::addObject(ObjectFile<ELFT> *File) {
  ObjectFiles.emplace_back(File);
  for (SymbolBody *Body : File->getSymbols())
    if (Body->isExternal())
      if (auto EC = resolve(Body))
        return EC;
  return std::error_code();
}

template <class ELFT>
std::error_code SymbolTable<ELFT>::addArchive(ArchiveFile *File) {
  ArchiveFiles.emplace_back(File);
  for (SymbolBody *Body : File->getSymbols())
    if (auto EC = resolve(Body))
      return EC;
  return std::error_code();
}

template <class ELFT>
std::error_code SymbolTable<ELFT>::addBitcode(BitcodeFile *File) {
  BitcodeFiles.emplace_back(File);
  for (SymbolBody *Body : File->getSymbols())
    if (Body->isExternal())
      if (auto EC = resolve(Body))
        return EC;
  return std::error_code();
}

template <class ELFT> bool SymbolTable<ELFT>::reportRemainingUndefines() {
  bool Ret = false;
  for (auto &I : Symtab) {
    Symbol *Sym = I.second;
    auto *Undef = dyn_cast<Undefined>(Sym->Body);
    if (!Undef)
      continue;
    if (SymbolBody *Alias = Undef->getWeakAlias()) {
      Sym->Body = Alias->getReplacement();
      if (!isa<Defined>(Sym->Body)) {
        // Aliases are yet another symbols pointed by other symbols
        // that could also remain undefined.
        llvm::errs() << "undefined symbol: " << Undef->getName() << "\n";
        Ret = true;
      }
      continue;
    }
    llvm::errs() << "undefined symbol: " << Undef->getName() << "\n";
    Ret = true;
  }
  return Ret;
}

// This function resolves conflicts if there's an existing symbol with
// the same name. Decisions are made based on symbol type.
template <class ELFT>
std::error_code SymbolTable<ELFT>::resolve(SymbolBody *New) {
  // Find an existing Symbol or create and insert a new one.
  StringRef Name = New->getName();
  Symbol *&Sym = Symtab[Name];
  if (!Sym) {
    Sym = new (Alloc) Symbol(New);
    New->setBackref(Sym);
    return std::error_code();
  }
  New->setBackref(Sym);

  // compare() returns -1, 0, or 1 if the lhs symbol is less preferable,
  // equivalent (conflicting), or more preferable, respectively.
  SymbolBody *Existing = Sym->Body;
  int comp = Existing->compare(New);
  if (comp < 0)
    Sym->Body = New;
  if (comp == 0) {
    llvm::errs() << "duplicate symbol: " << Name << "\n";
    return make_error_code(LLDError::DuplicateSymbols);
  }

  // If we have an Undefined symbol for a Lazy symbol, we need
  // to read an archive member to replace the Lazy symbol with
  // a Defined symbol.
  if (isa<Undefined>(Existing) || isa<Undefined>(New))
    if (auto *B = dyn_cast<Lazy>(Sym->Body))
      return addMemberFile(B);
  return std::error_code();
}

// Reads an archive member file pointed by a given symbol.
template <class ELFT>
std::error_code SymbolTable<ELFT>::addMemberFile(Lazy *Body) {
  auto FileOrErr = Body->getMember();
  if (auto EC = FileOrErr.getError())
    return EC;
  std::unique_ptr<InputFile> File = std::move(FileOrErr.get());

  // getMember returns an empty buffer if the member was already
  // read from the library.
  if (!File)
    return std::error_code();
  if (Config->Verbose)
    llvm::outs() << "Loaded " << File->getShortName() << " for "
                 << Body->getName() << "\n";
  return addFile(std::move(File));
}

template <class ELFT> std::vector<Chunk *> SymbolTable<ELFT>::getChunks() {
  std::vector<Chunk *> Res;
  for (std::unique_ptr<ObjectFile<ELFT>> &File : ObjectFiles) {
    std::vector<Chunk *> &V = File->getChunks();
    Res.insert(Res.end(), V.begin(), V.end());
  }
  return Res;
}

template <class ELFT> Defined *SymbolTable<ELFT>::find(StringRef Name) {
  auto It = Symtab.find(Name);
  if (It == Symtab.end())
    return nullptr;
  if (auto *Def = dyn_cast<Defined>(It->second->Body))
    return Def;
  return nullptr;
}

template <class ELFT>
std::error_code SymbolTable<ELFT>::addUndefined(StringRef Name) {
  return resolve(new (Alloc) Undefined(Name));
}

// Resolve To, and make From an alias to To.
template <class ELFT>
std::error_code SymbolTable<ELFT>::rename(StringRef From, StringRef To) {
  SymbolBody *Body = new (Alloc) Undefined(To);
  if (auto EC = resolve(Body))
    return EC;
  Symtab[From]->Body = Body->getReplacement();
  return std::error_code();
}

template <class ELFT> void SymbolTable<ELFT>::dump() {
  for (auto &P : Symtab) {
    Symbol *Ref = P.second;
    if (auto *Body = dyn_cast<Defined>(Ref->Body))
      llvm::dbgs() << Twine::utohexstr(Body->getVA()) << " " << Body->getName()
                   << "\n";
  }
}

template <class ELFT>
std::error_code SymbolTable<ELFT>::addCombinedLTOObject() {
  if (BitcodeFiles.empty())
    return std::error_code();

  // Create an object file and add it to the symbol table by replacing any
  // DefinedBitcode symbols with the definitions in the object file.
  LTOCodeGenerator CG;
  auto FileOrErr = createLTOObject(&CG);
  if (auto EC = FileOrErr.getError())
    return EC;
  ObjectFile<ELFT> *Obj = FileOrErr.get();

  for (SymbolBody *Body : Obj->getSymbols()) {
    if (!Body->isExternal())
      continue;
    // Find an existing Symbol. We should not see any new undefined symbols at
    // this point.
    StringRef Name = Body->getName();
    Symbol *&Sym = Symtab[Name];
    if (!Sym) {
      if (!isa<Defined>(Body)) {
        llvm::errs() << "LTO: undefined symbol: " << Name << '\n';
        return make_error_code(LLDError::BrokenFile);
      }
      Sym = new (Alloc) Symbol(Body);
      Body->setBackref(Sym);
      continue;
    }
    Body->setBackref(Sym);

    if (isa<DefinedBitcode>(Sym->Body)) {
      // The symbol should now be defined.
      if (!isa<Defined>(Body)) {
        llvm::errs() << "LTO: undefined symbol: " << Name << '\n';
        return make_error_code(LLDError::BrokenFile);
      }
      Sym->Body = Body;
    } else {
      int comp = Sym->Body->compare(Body);
      if (comp < 0)
        Sym->Body = Body;
      if (comp == 0) {
        llvm::errs() << "LTO: unexpected duplicate symbol: " << Name << "\n";
        return make_error_code(LLDError::BrokenFile);
      }
    }

    // We may see new references to runtime library symbols such as __chkstk
    // here. These symbols must be wholly defined in non-bitcode files.
    if (auto *B = dyn_cast<Lazy>(Sym->Body)) {
      size_t NumBitcodeFiles = BitcodeFiles.size();
      if (auto EC = addMemberFile(B))
        return EC;
      if (BitcodeFiles.size() != NumBitcodeFiles) {
        llvm::errs()
            << "LTO: late loaded symbol created new bitcode reference: " << Name
            << "\n";
        return make_error_code(LLDError::BrokenFile);
      }
    }
  }

  // New runtime library symbol references may have created undefined
  // references.
  if (reportRemainingUndefines())
    return make_error_code(LLDError::BrokenFile);
  return std::error_code();
}

// Combine and compile bitcode files and then return the result
// as a regular ELF object file.
template <class ELFT>
ErrorOr<ObjectFile<ELFT> *>
SymbolTable<ELFT>::createLTOObject(LTOCodeGenerator *CG) {
  // All symbols referenced by non-bitcode objects must be preserved.
  for (std::unique_ptr<ObjectFile<ELFT>> &File : ObjectFiles)
    for (SymbolBody *Body : File->getSymbols())
      if (auto *S = dyn_cast<DefinedBitcode>(Body->getReplacement()))
        CG->addMustPreserveSymbol(S->getName());

  // Likewise for bitcode symbols which we initially resolved to non-bitcode.
  for (std::unique_ptr<BitcodeFile> &File : BitcodeFiles)
    for (SymbolBody *Body : File->getSymbols())
      if (isa<DefinedBitcode>(Body) &&
          !isa<DefinedBitcode>(Body->getReplacement()))
        CG->addMustPreserveSymbol(Body->getName());

  // Likewise for other symbols that must be preserved.
  for (StringRef Name : Config->GCRoots)
    if (isa<DefinedBitcode>(Symtab[Name]->Body))
      CG->addMustPreserveSymbol(Name);

  CG->setModule(BitcodeFiles[0]->releaseModule());
  for (unsigned I = 1, E = BitcodeFiles.size(); I != E; ++I)
    CG->addModule(BitcodeFiles[I]->getModule());

  std::string ErrMsg;
  LTOMB = CG->compile(false, false, false, ErrMsg); // take MB ownership
  if (!LTOMB) {
    llvm::errs() << ErrMsg << '\n';
    return make_error_code(LLDError::BrokenFile);
  }
  auto Obj = new ObjectFile<ELFT>(LTOMB->getMemBufferRef());
  ObjectFiles.emplace_back(Obj);
  if (auto EC = Obj->parse())
    return EC;
  return Obj;
}

template class SymbolTable<llvm::object::ELF32LE>;
template class SymbolTable<llvm::object::ELF32BE>;
template class SymbolTable<llvm::object::ELF64LE>;
template class SymbolTable<llvm::object::ELF64BE>;
