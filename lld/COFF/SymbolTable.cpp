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
#include <utility>

using namespace llvm;

namespace lld {
namespace coff {

SymbolTable::SymbolTable() {
  resolve(new (Alloc) DefinedAbsolute("__ImageBase", Config->ImageBase));
  if (!Config->EntryName.empty())
    resolve(new (Alloc) Undefined(Config->EntryName));
}

void SymbolTable::addFile(std::unique_ptr<InputFile> File) {
  Files.push_back(std::move(File));
}

std::error_code SymbolTable::run() {
  while (FileIdx < Files.size()) {
    InputFile *F = Files[FileIdx++].get();
    if (Config->Verbose)
      llvm::outs() << "Reading " << F->getShortName() << "\n";
    if (auto EC = F->parse())
      return EC;
    if (auto *P = dyn_cast<ObjectFile>(F)) {
      ObjectFiles.push_back(P);
    } else if (auto *P = dyn_cast<ArchiveFile>(F)) {
      ArchiveFiles.push_back(P);
    } else if (auto *P = dyn_cast<BitcodeFile>(F)) {
      BitcodeFiles.push_back(P);
    } else {
      ImportFiles.push_back(cast<ImportFile>(F));
    }

    for (SymbolBody *B : F->getSymbols())
      if (B->isExternal())
        if (auto EC = resolve(B))
          return EC;

    // If a object file contains .drectve section,
    // read that and add files listed there.
    StringRef S = F->getDirectives();
    if (!S.empty())
      if (auto EC = Driver->parseDirectives(S))
        return EC;
  }
  return std::error_code();
}

bool SymbolTable::reportRemainingUndefines() {
  bool Ret = false;
  for (auto &I : Symtab) {
    Symbol *Sym = I.second;
    auto *Undef = dyn_cast<Undefined>(Sym->Body);
    if (!Undef)
      continue;
    StringRef Name = Undef->getName();
    // The weak alias may have been resovled, so check for that.
    if (SymbolBody *Alias = Undef->getWeakAlias()) {
      if (auto *D = dyn_cast<Defined>(Alias->getReplacement())) {
        Sym->Body = D;
        continue;
      }
    }
    // If we can resolve a symbol by removing __imp_ prefix, do that.
    // This odd rule is for compatibility with MSVC linker.
    if (Name.startswith("__imp_")) {
      if (Defined *Imp = find(Name.substr(strlen("__imp_")))) {
        auto *S = new (Alloc) DefinedLocalImport(Name, Imp);
        LocalImportChunks.push_back(S->getChunk());
        Sym->Body = S;
        continue;
      }
    }
    llvm::errs() << "undefined symbol: " << Name << "\n";
    // Remaining undefined symbols are not fatal if /force is specified.
    // They are replaced with dummy defined symbols.
    if (Config->Force) {
      Sym->Body = new (Alloc) DefinedAbsolute(Name, 0);
      continue;
    }
    Ret = true;
  }
  return Ret;
}

// This function resolves conflicts if there's an existing symbol with
// the same name. Decisions are made based on symbol type.
std::error_code SymbolTable::resolve(SymbolBody *New) {
  // Find an existing Symbol or create and insert a new one.
  StringRef Name = New->getName();
  Symbol *&Sym = Symtab[Name];
  if (!Sym) {
    Sym = new (Alloc) Symbol(New);
    New->setBackref(Sym);
    ++Version;
    return std::error_code();
  }
  New->setBackref(Sym);

  // compare() returns -1, 0, or 1 if the lhs symbol is less preferable,
  // equivalent (conflicting), or more preferable, respectively.
  SymbolBody *Existing = Sym->Body;
  int comp = Existing->compare(New);
  if (comp < 0) {
    Sym->Body = New;
    ++Version;
  }
  if (comp == 0) {
    llvm::errs() << "duplicate symbol: " << Existing->getDebugName()
                 << " and " << New->getDebugName() << "\n";
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
std::error_code SymbolTable::addMemberFile(Lazy *Body) {
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
  addFile(std::move(File));
  return std::error_code();
}

std::vector<Chunk *> SymbolTable::getChunks() {
  std::vector<Chunk *> Res;
  for (ObjectFile *File : ObjectFiles) {
    std::vector<Chunk *> &V = File->getChunks();
    Res.insert(Res.end(), V.begin(), V.end());
  }
  return Res;
}

Defined *SymbolTable::find(StringRef Name) {
  auto It = Symtab.find(Name);
  if (It == Symtab.end())
    return nullptr;
  if (auto *Def = dyn_cast<Defined>(It->second->Body))
    return Def;
  return nullptr;
}

// Find a given symbol or its mangled symbol.
std::pair<StringRef, Symbol *> SymbolTable::findMangled(StringRef S) {
  auto It = Symtab.find(S);
  if (It != Symtab.end()) {
    Symbol *Sym = It->second;
    if (isa<Defined>(Sym->Body))
      return std::make_pair(S, Sym);
  }

  // In Microsoft ABI, a non-member function name is mangled this way.
  std::string Prefix = ("?" + S + "@@Y").str();
  for (auto I : Symtab) {
    StringRef Name = I.first;
    Symbol *Sym = I.second;
    if (!Name.startswith(Prefix))
      continue;
    if (isa<Defined>(Sym->Body))
      return std::make_pair(Name, Sym);
  }
  return std::make_pair(S, nullptr);
}

std::error_code SymbolTable::resolveLazy(StringRef Name) {
  auto It = Symtab.find(Name);
  if (It == Symtab.end())
    return std::error_code();
  if (auto *B = dyn_cast<Lazy>(It->second->Body)) {
    if (auto EC = addMemberFile(B))
      return EC;
    return run();
  }
  return std::error_code();
}

// Windows specific -- Link default entry point name.
ErrorOr<StringRef> SymbolTable::findDefaultEntry() {
  // User-defined main functions and their corresponding entry points.
  static const char *Entries[][2] = {
      {"main", "mainCRTStartup"},
      {"wmain", "wmainCRTStartup"},
      {"WinMain", "WinMainCRTStartup"},
      {"wWinMain", "wWinMainCRTStartup"},
  };
  for (auto E : Entries) {
    resolveLazy(E[1]);
    if (find(E[1]))
      return StringRef(E[1]);
    if (!find(E[0]))
      continue;
    if (auto EC = resolve(new (Alloc) Undefined(E[1])))
      return EC;
    return StringRef(E[1]);
  }
  llvm::errs() << "entry point must be defined\n";
  return make_error_code(LLDError::InvalidOption);
}

std::error_code SymbolTable::addUndefined(StringRef Name) {
  return resolve(new (Alloc) Undefined(Name));
}

// Resolve To, and make From an alias to To.
std::error_code SymbolTable::rename(StringRef From, StringRef To) {
  // If From is not undefined, do nothing.
  // Otherwise, rename it to see if To can be resolved instead.
  auto It = Symtab.find(From);
  if (It == Symtab.end())
    return std::error_code();
  Symbol *Sym = It->second;
  if (!isa<Undefined>(Sym->Body))
    return std::error_code();
  SymbolBody *Body = new (Alloc) Undefined(To);
  if (auto EC = resolve(Body))
    return EC;
  Sym->Body = Body->getReplacement();
  Body->setBackref(Sym);
  ++Version;
  return std::error_code();
}

void SymbolTable::printMap(llvm::raw_ostream &OS) {
  for (ObjectFile *File : ObjectFiles) {
    OS << File->getShortName() << ":\n";
    for (SymbolBody *Body : File->getSymbols())
      if (auto *R = dyn_cast<DefinedRegular>(Body))
        if (R->isLive())
          OS << Twine::utohexstr(Config->ImageBase + R->getRVA())
             << " " << R->getName() << "\n";
  }
}

std::error_code SymbolTable::addCombinedLTOObject() {
  if (BitcodeFiles.empty())
    return std::error_code();

  // Create an object file and add it to the symbol table by replacing any
  // DefinedBitcode symbols with the definitions in the object file.
  LTOCodeGenerator CG;
  auto FileOrErr = createLTOObject(&CG);
  if (auto EC = FileOrErr.getError())
    return EC;
  ObjectFile *Obj = FileOrErr.get();

  // Skip the combined object file as the file is processed below
  // rather than by run().
  ++FileIdx;

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
    if (auto *B = dyn_cast<Lazy>(Sym->Body))
      if (auto EC = addMemberFile(B))
        return EC;
  }

  size_t NumBitcodeFiles = BitcodeFiles.size();
  if (auto EC = run())
    return EC;
  if (BitcodeFiles.size() != NumBitcodeFiles) {
    llvm::errs() << "LTO: late loaded symbol created new bitcode reference\n";
    return make_error_code(LLDError::BrokenFile);
  }

  // New runtime library symbol references may have created undefined references.
  if (reportRemainingUndefines())
    return make_error_code(LLDError::BrokenFile);
  return std::error_code();
}

// Combine and compile bitcode files and then return the result
// as a regular COFF object file.
ErrorOr<ObjectFile *> SymbolTable::createLTOObject(LTOCodeGenerator *CG) {
  // All symbols referenced by non-bitcode objects must be preserved.
  for (ObjectFile *File : ObjectFiles)
    for (SymbolBody *Body : File->getSymbols())
      if (auto *S = dyn_cast<DefinedBitcode>(Body->getReplacement()))
        CG->addMustPreserveSymbol(S->getName());

  // Likewise for bitcode symbols which we initially resolved to non-bitcode.
  for (BitcodeFile *File : BitcodeFiles)
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
  auto *Obj = new ObjectFile(LTOMB->getMemBufferRef());
  Files.emplace_back(Obj);
  ObjectFiles.push_back(Obj);
  if (auto EC = Obj->parse())
    return EC;
  return Obj;
}

} // namespace coff
} // namespace lld
