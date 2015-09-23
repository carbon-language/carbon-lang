//===- SymbolTable.cpp ----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SymbolTable.h"
#include "Config.h"
#include "Error.h"
#include "Symbols.h"
#include "Target.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;

using namespace lld;
using namespace lld::elf2;

SymbolTable::SymbolTable() {}

bool SymbolTable::shouldUseRela() const {
  ELFKind K = getFirstELF()->getELFKind();
  return K == ELF64LEKind || K == ELF64BEKind;
}

void SymbolTable::addFile(std::unique_ptr<InputFile> File) {
  File->parse();
  InputFile *FileP = File.release();
  if (auto *AF = dyn_cast<ArchiveFile>(FileP)) {
    ArchiveFiles.emplace_back(AF);
    for (Lazy &Sym : AF->getLazySymbols())
      addLazy(&Sym);
    return;
  }
  addELFFile(cast<ELFFileBase>(FileP));
}

static TargetInfo *createTarget(uint16_t EMachine) {
  switch (EMachine) {
  case EM_PPC:
    return new PPCTargetInfo();
  case EM_ARM:
    return new ARMTargetInfo();
  case EM_PPC64:
    return new PPC64TargetInfo();
  case EM_X86_64:
    return new X86_64TargetInfo();
  case EM_386:
    return new X86TargetInfo();
  }
  error("Unknown target machine");
}

template <class ELFT> void SymbolTable::init(uint16_t EMachine) {
  Target.reset(createTarget(EMachine));
  if (Config->Shared)
    return;
  EntrySym = new (Alloc) Undefined<ELFT>("_start", Undefined<ELFT>::Synthetic);
  resolve<ELFT>(EntrySym);
}

template <class ELFT> void SymbolTable::addELFFile(ELFFileBase *File) {
  if (const ELFFileBase *Old = getFirstELF()) {
    if (!Old->isCompatibleWith(*File))
      error(Twine(Old->getName() + " is incompatible with " + File->getName()));
  } else {
    init<ELFT>(File->getEMachine());
  }

  if (auto *O = dyn_cast<ObjectFileBase>(File)) {
    ObjectFiles.emplace_back(O);
    for (SymbolBody *Body : O->getSymbols())
      resolve<ELFT>(Body);
  }

  if (auto *S = dyn_cast<SharedFile<ELFT>>(File)) {
    SharedFiles.emplace_back(S);
    for (SharedSymbol<ELFT> &Body : S->getSharedSymbols())
      resolve<ELFT>(&Body);
  }
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

template <class ELFT>
void SymbolTable::dupErorr(const SymbolBody &Old, const SymbolBody &New) {
  typedef typename ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;

  const Elf_Sym &OldE = cast<ELFSymbolBody<ELFT>>(Old).Sym;
  const Elf_Sym &NewE = cast<ELFSymbolBody<ELFT>>(New).Sym;
  ELFFileBase *OldFile = nullptr;
  ELFFileBase *NewFile = nullptr;

  for (const std::unique_ptr<ObjectFileBase> &F : ObjectFiles) {
    const auto &File = cast<ObjectFile<ELFT>>(*F);
    Elf_Sym_Range Syms = File.getObj()->symbols(File.getSymbolTable());
    if (&OldE > Syms.begin() && &OldE < Syms.end())
      OldFile = F.get();
    if (&NewE > Syms.begin() && &NewE < Syms.end())
      NewFile = F.get();
  }

  error(Twine("duplicate symbol: ") + Old.getName() + " in " +
        OldFile->getName() + " and " + NewFile->getName());
}

// This function resolves conflicts if there's an existing symbol with
// the same name. Decisions are made based on symbol type.
template <class ELFT> void SymbolTable::resolve(SymbolBody *New) {
  Symbol *Sym = insert(New);
  if (Sym->Body == New)
    return;

  SymbolBody *Existing = Sym->Body;

  if (Lazy *L = dyn_cast<Lazy>(Existing)) {
    if (New->isUndefined()) {
      addMemberFile(L);
      return;
    }

    // Found a definition for something also in an archive. Ignore the archive
    // definition.
    Sym->Body = New;
    return;
  }

  // compare() returns -1, 0, or 1 if the lhs symbol is less preferable,
  // equivalent (conflicting), or more preferable, respectively.
  int comp = Existing->compare<ELFT>(New);
  if (comp < 0)
    Sym->Body = New;
  else if (comp == 0)
    dupErorr<ELFT>(*Existing, *New);
}

Symbol *SymbolTable::insert(SymbolBody *New) {
  // Find an existing Symbol or create and insert a new one.
  StringRef Name = New->getName();
  Symbol *&Sym = Symtab[Name];
  if (!Sym) {
    Sym = new (Alloc) Symbol(New);
    New->setBackref(Sym);
    return Sym;
  }
  New->setBackref(Sym);
  return Sym;
}

void SymbolTable::addLazy(Lazy *New) {
  Symbol *Sym = insert(New);
  if (Sym->Body == New)
    return;
  SymbolBody *Existing = Sym->Body;
  if (Existing->isDefined() || Existing->isLazy())
    return;
  Sym->Body = New;
  assert(Existing->isUndefined() && "Unexpected symbol kind.");
  addMemberFile(New);
}

void SymbolTable::addMemberFile(Lazy *Body) {
  std::unique_ptr<InputFile> File = Body->getMember();

  // getMember returns nullptr if the member was already read from the library.
  if (!File)
    return;

  addFile(std::move(File));
}
