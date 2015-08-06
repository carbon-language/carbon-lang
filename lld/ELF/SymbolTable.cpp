//===- SymbolTable.cpp ----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SymbolTable.h"
#include "Error.h"
#include "Symbols.h"

using namespace llvm;

using namespace lld;
using namespace lld::elf2;

SymbolTable::SymbolTable() { resolve(new (Alloc) Undefined("_start")); }

void SymbolTable::addFile(std::unique_ptr<InputFile> File) {
  File->parse();
  InputFile *FileP = File.release();
  auto *P = cast<ObjectFileBase>(FileP);
  addObject(P);
}

void SymbolTable::addObject(ObjectFileBase *File) {
  if (!ObjectFiles.empty()) {
    ObjectFileBase &Old = *ObjectFiles[0];
    if (!Old.isCompatibleWith(*File))
      error(Twine(Old.getName() + " is incompatible with " + File->getName()));
  }

  ObjectFiles.emplace_back(File);
  for (SymbolBody *Body : File->getSymbols())
    resolve(Body);
}

void SymbolTable::reportRemainingUndefines() {
  for (auto &I : Symtab) {
    Symbol *Sym = I.second;
    if (auto *Undef = dyn_cast<Undefined>(Sym->Body))
      error(Twine("undefined symbol: ") + Undef->getName());
  }
}

// This function resolves conflicts if there's an existing symbol with
// the same name. Decisions are made based on symbol type.
void SymbolTable::resolve(SymbolBody *New) {
  // Find an existing Symbol or create and insert a new one.
  StringRef Name = New->getName();
  Symbol *&Sym = Symtab[Name];
  if (!Sym) {
    Sym = new (Alloc) Symbol(New);
    New->setBackref(Sym);
    return;
  }
  New->setBackref(Sym);

  // compare() returns -1, 0, or 1 if the lhs symbol is less preferable,
  // equivalent (conflicting), or more preferable, respectively.
  SymbolBody *Existing = Sym->Body;
  int comp = Existing->compare(New);
  if (comp < 0)
    Sym->Body = New;
  if (comp == 0)
    error(Twine("duplicate symbol: ") + Name);
}
