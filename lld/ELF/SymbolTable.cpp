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

void SymbolTable::addObject(ObjectFileBase *File) {
  if (!ObjectFiles.empty()) {
    ObjectFileBase &Old = *ObjectFiles[0];
    if (!Old.isCompatibleWith(*File))
      error(Twine(Old.getName() + " is incompatible with " + File->getName()));
  } else {
    auto *Start = new (Alloc) SyntheticUndefined("_start");
    switch (File->kind()) {
    case InputFile::Object32LEKind:
      resolve<ELF32LE>(Start);
      break;
    case InputFile::Object32BEKind:
      resolve<ELF32BE>(Start);
      break;
    case InputFile::Object64LEKind:
      resolve<ELF64LE>(Start);
      break;
    case InputFile::Object64BEKind:
      resolve<ELF64BE>(Start);
      break;
    }
  }

  ObjectFiles.emplace_back(File);
  for (SymbolBody *Body : File->getSymbols()) {
    switch (File->kind()) {
    case InputFile::Object32LEKind:
      resolve<ELF32LE>(Body);
      break;
    case InputFile::Object32BEKind:
      resolve<ELF32BE>(Body);
      break;
    case InputFile::Object64LEKind:
      resolve<ELF64LE>(Body);
      break;
    case InputFile::Object64BEKind:
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
  Builder.add(Name);
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
