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
#include "Symbols.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/LTO/LTOCodeGenerator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

using namespace llvm;

namespace lld {
namespace coff {

void SymbolTable::addFile(std::unique_ptr<InputFile> FileP) {
  InputFile *File = FileP.get();
  Files.push_back(std::move(FileP));
  if (auto *F = dyn_cast<ArchiveFile>(File)) {
    ArchiveQueue.push_back(F);
    return;
  }
  ObjectQueue.push_back(File);
  if (auto *F = dyn_cast<ObjectFile>(File)) {
    ObjectFiles.push_back(F);
  } else if (auto *F = dyn_cast<BitcodeFile>(File)) {
    BitcodeFiles.push_back(F);
  } else {
    ImportFiles.push_back(cast<ImportFile>(File));
  }
}

void SymbolTable::step() {
  if (queueEmpty())
    return;
  readObjects();
  readArchives();
}

void SymbolTable::run() {
  while (!queueEmpty())
    step();
}

void SymbolTable::readArchives() {
  if (ArchiveQueue.empty())
    return;

  // Add lazy symbols to the symbol table. Lazy symbols that conflict
  // with existing undefined symbols are accumulated in LazySyms.
  std::vector<Symbol *> LazySyms;
  for (ArchiveFile *File : ArchiveQueue) {
    if (Config->Verbose)
      llvm::outs() << "Reading " << File->getShortName() << "\n";
    File->parse();
    for (Lazy *Sym : File->getLazySymbols())
      addLazy(Sym, &LazySyms);
  }
  ArchiveQueue.clear();

  // Add archive member files to ObjectQueue that should resolve
  // existing undefined symbols.
  for (Symbol *Sym : LazySyms)
    addMemberFile(cast<Lazy>(Sym->Body));
}

void SymbolTable::readObjects() {
  if (ObjectQueue.empty())
    return;

  // Add defined and undefined symbols to the symbol table.
  std::vector<StringRef> Directives;
  for (size_t I = 0; I < ObjectQueue.size(); ++I) {
    InputFile *File = ObjectQueue[I];
    if (Config->Verbose)
      llvm::outs() << "Reading " << File->getShortName() << "\n";
    File->parse();
    // Adding symbols may add more files to ObjectQueue
    // (but not to ArchiveQueue).
    for (SymbolBody *Sym : File->getSymbols())
      if (Sym->isExternal())
        addSymbol(Sym);
    StringRef S = File->getDirectives();
    if (!S.empty()) {
      Directives.push_back(S);
      if (Config->Verbose)
        llvm::outs() << "Directives: " << File->getShortName()
                     << ": " << S << "\n";
    }
  }
  ObjectQueue.clear();

  // Parse directive sections. This may add files to
  // ArchiveQueue and ObjectQueue.
  for (StringRef S : Directives)
    Driver->parseDirectives(S);
}

bool SymbolTable::queueEmpty() {
  return ArchiveQueue.empty() && ObjectQueue.empty();
}

void SymbolTable::reportRemainingUndefines(bool Resolve) {
  llvm::SmallPtrSet<SymbolBody *, 8> Undefs;
  for (auto &I : Symtab) {
    Symbol *Sym = I.second;
    auto *Undef = dyn_cast<Undefined>(Sym->Body);
    if (!Undef)
      continue;
    StringRef Name = Undef->getName();
    // A weak alias may have been resolved, so check for that.
    if (Defined *D = Undef->getWeakAlias()) {
      if (Resolve)
        Sym->Body = D;
      continue;
    }
    // If we can resolve a symbol by removing __imp_ prefix, do that.
    // This odd rule is for compatibility with MSVC linker.
    if (Name.startswith("__imp_")) {
      Symbol *Imp = find(Name.substr(strlen("__imp_")));
      if (Imp && isa<Defined>(Imp->Body)) {
        if (!Resolve)
          continue;
        auto *D = cast<Defined>(Imp->Body);
        auto *S = new (Alloc) DefinedLocalImport(Name, D);
        LocalImportChunks.push_back(S->getChunk());
        Sym->Body = S;
        continue;
      }
    }
    // Remaining undefined symbols are not fatal if /force is specified.
    // They are replaced with dummy defined symbols.
    if (Config->Force && Resolve)
      Sym->Body = new (Alloc) DefinedAbsolute(Name, 0);
    Undefs.insert(Sym->Body);
  }
  if (Undefs.empty())
    return;
  for (Undefined *U : Config->GCRoot)
    if (Undefs.count(U->repl()))
      llvm::errs() << "<root>: undefined symbol: " << U->getName() << "\n";
  for (std::unique_ptr<InputFile> &File : Files)
    if (!isa<ArchiveFile>(File.get()))
      for (SymbolBody *Sym : File->getSymbols())
        if (Undefs.count(Sym->repl()))
          llvm::errs() << File->getShortName() << ": undefined symbol: "
                       << Sym->getName() << "\n";
  if (!Config->Force)
    error("Link failed");
}

void SymbolTable::addLazy(Lazy *New, std::vector<Symbol *> *Accum) {
  Symbol *Sym = insert(New);
  if (Sym->Body == New)
    return;
  for (;;) {
    SymbolBody *Existing = Sym->Body;
    if (isa<Defined>(Existing))
      return;
    if (Lazy *L = dyn_cast<Lazy>(Existing))
      if (L->getFileIndex() < New->getFileIndex())
        return;
    if (!Sym->Body.compare_exchange_strong(Existing, New))
      continue;
    New->setBackref(Sym);
    if (isa<Undefined>(Existing))
      Accum->push_back(Sym);
    return;
  }
}

void SymbolTable::addSymbol(SymbolBody *New) {
  // Find an existing symbol or create and insert a new one.
  assert(isa<Defined>(New) || isa<Undefined>(New));
  Symbol *Sym = insert(New);
  if (Sym->Body == New)
    return;

  for (;;) {
    SymbolBody *Existing = Sym->Body;

    // If we have an undefined symbol and a lazy symbol,
    // let the lazy symbol to read a member file.
    if (auto *L = dyn_cast<Lazy>(Existing)) {
      // Undefined symbols with weak aliases need not to be resolved,
      // since they would be replaced with weak aliases if they remain
      // undefined.
      if (auto *U = dyn_cast<Undefined>(New))
        if (!U->WeakAlias) {
          addMemberFile(L);
          return;
        }
      if (!Sym->Body.compare_exchange_strong(Existing, New))
        continue;
      return;
    }

    // compare() returns -1, 0, or 1 if the lhs symbol is less preferable,
    // equivalent (conflicting), or more preferable, respectively.
    int Comp = Existing->compare(New);
    if (Comp == 0)
      error(Twine("duplicate symbol: ") + Existing->getDebugName() + " and " +
            New->getDebugName());
    if (Comp < 0)
      if (!Sym->Body.compare_exchange_strong(Existing, New))
        continue;
    return;
  }
}

Symbol *SymbolTable::insert(SymbolBody *New) {
  Symbol *&Sym = Symtab[New->getName()];
  if (Sym) {
    New->setBackref(Sym);
    return Sym;
  }
  Sym = new (Alloc) Symbol(New);
  New->setBackref(Sym);
  return Sym;
}

// Reads an archive member file pointed by a given symbol.
void SymbolTable::addMemberFile(Lazy *Body) {
  std::unique_ptr<InputFile> File = Body->getMember();

  // getMember returns an empty buffer if the member was already
  // read from the library.
  if (!File)
    return;
  if (Config->Verbose)
    llvm::outs() << "Loaded " << File->getShortName() << " for "
                 << Body->getName() << "\n";
  addFile(std::move(File));
}

std::vector<Chunk *> SymbolTable::getChunks() {
  std::vector<Chunk *> Res;
  for (ObjectFile *File : ObjectFiles) {
    std::vector<Chunk *> &V = File->getChunks();
    Res.insert(Res.end(), V.begin(), V.end());
  }
  return Res;
}

Symbol *SymbolTable::find(StringRef Name) {
  auto It = Symtab.find(Name);
  if (It == Symtab.end())
    return nullptr;
  return It->second;
}

Symbol *SymbolTable::findUnderscore(StringRef Name) {
  if (Config->Machine == I386)
    return find(("_" + Name).str());
  return find(Name);
}

StringRef SymbolTable::findByPrefix(StringRef Prefix) {
  for (auto Pair : Symtab) {
    StringRef Name = Pair.first;
    if (Name.startswith(Prefix))
      return Name;
  }
  return "";
}

StringRef SymbolTable::findMangle(StringRef Name) {
  if (Symbol *Sym = find(Name))
    if (!isa<Undefined>(Sym->Body))
      return Name;
  if (Config->Machine != I386)
    return findByPrefix(("?" + Name + "@@Y").str());
  if (!Name.startswith("_"))
    return "";
  // Search for x86 C function.
  StringRef S = findByPrefix((Name + "@").str());
  if (!S.empty())
    return S;
  // Search for x86 C++ non-member function.
  return findByPrefix(("?" + Name.substr(1) + "@@Y").str());
}

void SymbolTable::mangleMaybe(Undefined *U) {
  if (U->WeakAlias)
    return;
  if (!isa<Undefined>(U->repl()))
    return;
  StringRef Alias = findMangle(U->getName());
  if (!Alias.empty())
    U->WeakAlias = addUndefined(Alias);
}

Undefined *SymbolTable::addUndefined(StringRef Name) {
  auto *New = new (Alloc) Undefined(Name);
  addSymbol(New);
  if (auto *U = dyn_cast<Undefined>(New->repl()))
    return U;
  return New;
}

DefinedRelative *SymbolTable::addRelative(StringRef Name, uint64_t VA) {
  auto *New = new (Alloc) DefinedRelative(Name, VA);
  addSymbol(New);
  return New;
}

DefinedAbsolute *SymbolTable::addAbsolute(StringRef Name, uint64_t VA) {
  auto *New = new (Alloc) DefinedAbsolute(Name, VA);
  addSymbol(New);
  return New;
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

void SymbolTable::addCombinedLTOObject() {
  if (BitcodeFiles.empty())
    return;

  // Diagnose any undefined symbols early, but do not resolve weak externals,
  // as resolution breaks the invariant that each Symbol points to a unique
  // SymbolBody, which we rely on to replace DefinedBitcode symbols correctly.
  reportRemainingUndefines(/*Resolve=*/false);

  // Create an object file and add it to the symbol table by replacing any
  // DefinedBitcode symbols with the definitions in the object file.
  LTOCodeGenerator CG;
  ObjectFile *Obj = createLTOObject(&CG);

  for (SymbolBody *Body : Obj->getSymbols()) {
    if (!Body->isExternal())
      continue;
    // We should not see any new undefined symbols at this point, but we'll
    // diagnose them later in reportRemainingUndefines().
    StringRef Name = Body->getName();
    Symbol *Sym = insert(Body);

    if (isa<DefinedBitcode>(Sym->Body)) {
      Sym->Body = Body;
      continue;
    }
    if (auto *L = dyn_cast<Lazy>(Sym->Body)) {
      // We may see new references to runtime library symbols such as __chkstk
      // here. These symbols must be wholly defined in non-bitcode files.
      addMemberFile(L);
      continue;
    }
    SymbolBody *Existing = Sym->Body;
    int Comp = Existing->compare(Body);
    if (Comp == 0)
      error(Twine("LTO: unexpected duplicate symbol: ") + Name);
    if (Comp < 0)
      Sym->Body = Body;
  }

  size_t NumBitcodeFiles = BitcodeFiles.size();
  run();
  if (BitcodeFiles.size() != NumBitcodeFiles)
    error("LTO: late loaded symbol created new bitcode reference");
}

// Combine and compile bitcode files and then return the result
// as a regular COFF object file.
ObjectFile *SymbolTable::createLTOObject(LTOCodeGenerator *CG) {
  // All symbols referenced by non-bitcode objects must be preserved.
  for (ObjectFile *File : ObjectFiles)
    for (SymbolBody *Body : File->getSymbols())
      if (auto *S = dyn_cast<DefinedBitcode>(Body->repl()))
        CG->addMustPreserveSymbol(S->getName());

  // Likewise for bitcode symbols which we initially resolved to non-bitcode.
  for (BitcodeFile *File : BitcodeFiles)
    for (SymbolBody *Body : File->getSymbols())
      if (isa<DefinedBitcode>(Body) && !isa<DefinedBitcode>(Body->repl()))
        CG->addMustPreserveSymbol(Body->getName());

  // Likewise for other symbols that must be preserved.
  for (Undefined *U : Config->GCRoot) {
    if (auto *S = dyn_cast<DefinedBitcode>(U->repl()))
      CG->addMustPreserveSymbol(S->getName());
    else if (auto *S = dyn_cast_or_null<DefinedBitcode>(U->getWeakAlias()))
      CG->addMustPreserveSymbol(S->getName());
  }

  CG->setModule(BitcodeFiles[0]->releaseModule());
  for (unsigned I = 1, E = BitcodeFiles.size(); I != E; ++I)
    CG->addModule(BitcodeFiles[I]->getModule());

  std::string ErrMsg;
  LTOMB = CG->compile(false, false, false, ErrMsg); // take MB ownership
  if (!LTOMB)
    error(ErrMsg);
  auto *Obj = new ObjectFile(LTOMB->getMemBufferRef());
  Files.emplace_back(Obj);
  ObjectFiles.push_back(Obj);
  Obj->parse();
  return Obj;
}

} // namespace coff
} // namespace lld
