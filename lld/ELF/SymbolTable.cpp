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
  auto *P = cast<ELFFileBase>(FileP);
  addELFFile(P);
}

template <class ELFT> void SymbolTable::init() {
  resolve<ELFT>(new (Alloc)
                    Undefined<ELFT>("_start", Undefined<ELFT>::Synthetic));
}

template <class ELFT> void SymbolTable::addELFFile(ELFFileBase *File) {
  if (const ELFFileBase *Old = getFirstELF()) {
    if (!Old->isCompatibleWith(*File))
      error(Twine(Old->getName() + " is incompatible with " + File->getName()));
  } else {
    init<ELFT>();
  }

  if (auto *O = dyn_cast<ObjectFileBase>(File)) {
    ObjectFiles.emplace_back(O);
    for (SymbolBody *Body : O->getSymbols())
      resolve<ELFT>(Body);
  }

  if (auto *S = dyn_cast<SharedFileBase>(File))
    SharedFiles.emplace_back(S);
}

void SymbolTable::addELFFile(ELFFileBase *File) {
  switch (File->getELFKind()) {
    case ELF32LEKind:
      addELFFile<ELF32LE>(File);
      break;
    case ELF32BEKind:
      addELFFile<ELF32BE>(File);
      break;
    case ELF64LEKind:
      addELFFile<ELF64LE>(File);
      break;
    case ELF64BEKind:
      addELFFile<ELF64BE>(File);
      break;
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
