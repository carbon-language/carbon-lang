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
#include "SymbolTable.h"
#include "lld/Core/Error.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace lld {
namespace coff {

SymbolTable::SymbolTable() {
  addSymbol(new DefinedAbsolute("__ImageBase", Config->ImageBase));
  if (!Config->EntryName.empty())
    addSymbol(new Undefined(Config->EntryName));
}

std::error_code SymbolTable::addFile(std::unique_ptr<InputFile> File) {
  if (auto EC = File->parse())
    return EC;
  InputFile *FileP = File.release();
  if (auto *P = dyn_cast<ObjectFile>(FileP))
    return addObject(P);
  if (auto *P = dyn_cast<ArchiveFile>(FileP))
    return addArchive(P);
  return addImport(cast<ImportFile>(FileP));
}

std::error_code SymbolTable::addObject(ObjectFile *File) {
  ObjectFiles.emplace_back(File);
  for (SymbolBody *Body : File->getSymbols())
    if (Body->isExternal())
      if (auto EC = resolve(Body))
        return EC;

  // If an object file contains .drectve section, read it and add
  // files listed in the section.
  StringRef Dir = File->getDirectives();
  if (!Dir.empty()) {
    std::vector<std::unique_ptr<InputFile>> Libs;
    if (auto EC = parseDirectives(Dir, &Libs, &StringAlloc))
      return EC;
    for (std::unique_ptr<InputFile> &Lib : Libs)
      addFile(std::move(Lib));
  }
  return std::error_code();
}

std::error_code SymbolTable::addArchive(ArchiveFile *File) {
  ArchiveFiles.emplace_back(File);
  for (SymbolBody *Body : File->getSymbols())
    if (auto EC = resolve(Body))
      return EC;
  return std::error_code();
}

std::error_code SymbolTable::addImport(ImportFile *File) {
  ImportFiles.emplace_back(File);
  for (SymbolBody *Body : File->getSymbols())
    if (auto EC = resolve(Body))
      return EC;
  return std::error_code();
}

bool SymbolTable::reportRemainingUndefines() {
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
std::error_code SymbolTable::resolve(SymbolBody *New) {
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
  if (comp == 0)
    return make_dynamic_error_code(Twine("duplicate symbol: ") + Name);

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
    llvm::dbgs() << "Loaded " << File->getShortName() << " for "
                 << Body->getName() << "\n";
  return addFile(std::move(File));
}

std::vector<Chunk *> SymbolTable::getChunks() {
  std::vector<Chunk *> Res;
  for (std::unique_ptr<ObjectFile> &File : ObjectFiles) {
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

// Link default entry point name.
ErrorOr<StringRef> SymbolTable::findDefaultEntry() {
  static const char *Entries[][2] = {
      {"mainCRTStartup", "mainCRTStartup"},
      {"wmainCRTStartup", "wmainCRTStartup"},
      {"WinMainCRTStartup", "WinMainCRTStartup"},
      {"wWinMainCRTStartup", "wWinMainCRTStartup"},
      {"main", "mainCRTStartup"},
      {"wmain", "wmainCRTStartup"},
      {"WinMain", "WinMainCRTStartup"},
      {"wWinMain", "wWinMainCRTStartup"},
  };
  for (size_t I = 0; I < sizeof(Entries); ++I) {
    if (!find(Entries[I][0]))
      continue;
    if (auto EC = addSymbol(new Undefined(Entries[I][1])))
      return EC;
    return StringRef(Entries[I][1]);
  }
  return make_dynamic_error_code("entry point must be defined");
}

std::error_code SymbolTable::addSymbol(SymbolBody *Body) {
  OwningSymbols.push_back(std::unique_ptr<SymbolBody>(Body));
  return resolve(Body);
}

void SymbolTable::dump() {
  for (auto &P : Symtab) {
    Symbol *Ref = P.second;
    if (auto *Body = dyn_cast<Defined>(Ref->Body))
      llvm::dbgs() << Twine::utohexstr(Config->ImageBase + Body->getRVA())
                   << " " << Body->getName() << "\n";
  }
}

} // namespace coff
} // namespace lld
