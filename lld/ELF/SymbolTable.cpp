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
using namespace llvm::object;

using namespace lld;
using namespace lld::elf2;

SymbolTable::SymbolTable() {
}

void SymbolTable::addFile(std::unique_ptr<InputFile> File) {
  File->parse();
  InputFile *FileP = File.release();
  auto *P = cast<ObjectFileBase>(FileP);
  addObject(P);
}

template <class ELFT> void SymbolTable::init() {
  resolve<ELFT>(new (Alloc)
                    Undefined<ELFT>("_start", Undefined<ELFT>::Synthetic));
}

void SymbolTable::addObject(ObjectFileBase *File) {
  if (const ObjectFileBase *Old = getFirstObject()) {
    if (!Old->isCompatibleWith(*File))
      error(Twine(Old->getName() + " is incompatible with " + File->getName()));
  } else {
    switch (File->getELFKind()) {
    case ELF32LEKind:
      init<ELF32LE>();
      break;
    case ELF32BEKind:
      init<ELF32BE>();
      break;
    case ELF64LEKind:
      init<ELF64LE>();
      break;
    case ELF64BEKind:
      init<ELF64BE>();
      break;
    }
  }

  ObjectFiles.emplace_back(File);
  for (SymbolBody *Body : File->getSymbols()) {
    switch (File->getELFKind()) {
    case ELF32LEKind:
      resolve<ELF32LE>(Body);
      break;
    case ELF32BEKind:
      resolve<ELF32BE>(Body);
      break;
    case ELF64LEKind:
      resolve<ELF64LE>(Body);
      break;
    case ELF64BEKind:
      resolve<ELF64BE>(Body);
      break;
    }
  }
}

void SymbolTable::reportRemainingUndefines() {
  for (auto &I : Symtab) {
    SymbolBody *Body = I.second->Body;
    if (Body->isStrongUndefined())
      error(Twine("undefined symbol: ") + Body->getName());
  }
}

// This function resolves conflicts if there's an existing symbol with
// the same name. Decisions are made based on symbol type.
template <class ELFT> void SymbolTable::resolve(SymbolBody *New) {
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
  int comp = Existing->compare<ELFT>(New);
  if (comp < 0)
    Sym->Body = New;
  if (comp == 0)
    error(Twine("duplicate symbol: ") + Name);
}
