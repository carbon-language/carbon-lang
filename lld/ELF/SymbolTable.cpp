//===- SymbolTable.cpp ----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SymbolTable.h"
#include "Driver.h"
#include "Symbols.h"

using namespace llvm;

using namespace lld;
using namespace lld::elf2;

template <class ELFT> SymbolTable<ELFT>::SymbolTable() {
  resolve(new (Alloc) Undefined("_start"));
}

template <class ELFT>
void SymbolTable<ELFT>::addFile(std::unique_ptr<InputFile> File) {
  File->parse();
  InputFile *FileP = File.release();
  auto *P = cast<ObjectFile<ELFT>>(FileP);
  addObject(P);
  return;
}

template <class ELFT>
void SymbolTable<ELFT>::addObject(ObjectFile<ELFT> *File) {
  ObjectFiles.emplace_back(File);
  for (SymbolBody *Body : File->getSymbols())
    if (Body->isExternal())
      resolve(Body);
}

template <class ELFT> void SymbolTable<ELFT>::reportRemainingUndefines() {
  for (auto &I : Symtab) {
    Symbol *Sym = I.second;
    if (auto *Undef = dyn_cast<Undefined>(Sym->Body))
      error(Twine("undefined symbol: ") + Undef->getName());
  }
}

// This function resolves conflicts if there's an existing symbol with
// the same name. Decisions are made based on symbol type.
template <class ELFT> void SymbolTable<ELFT>::resolve(SymbolBody *New) {
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

namespace lld {
namespace elf2 {
template class SymbolTable<object::ELF32LE>;
template class SymbolTable<object::ELF32BE>;
template class SymbolTable<object::ELF64LE>;
template class SymbolTable<object::ELF64BE>;
}
}
