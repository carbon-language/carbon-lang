//===- Symbols.cpp --------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "InputFiles.h"
#include "Symbols.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm::object;
using llvm::sys::fs::identify_magic;
using llvm::sys::fs::file_magic;

namespace lld {
namespace coff {

// Returns 1, 0 or -1 if this symbol should take precedence
// over the Other, tie or lose, respectively.
int DefinedRegular::compare(SymbolBody *Other) {
  if (Other->kind() < kind())
    return -Other->compare(this);
  if (isa<DefinedRegular>(Other))
    return 0;
  return 1;
}

int DefinedCOMDAT::compare(SymbolBody *Other) {
  if (Other->kind() < kind())
    return -Other->compare(this);
  if (isa<DefinedRegular>(Other))
    return 0;
  return 1;
}

int DefinedCommon::compare(SymbolBody *Other) {
  if (Other->kind() < kind())
    return -Other->compare(this);
  if (auto *D = dyn_cast<DefinedCommon>(Other))
    return getSize() > D->getSize() ? 1 : -1;
  if (isa<Lazy>(Other) || isa<Undefined>(Other))
    return 1;
  return -1;
}

int DefinedBitcode::compare(SymbolBody *Other) {
  assert(Other->kind() >= kind());
  if (!isa<Defined>(Other))
    return 1;

  if (auto *B = dyn_cast<DefinedBitcode>(Other)) {
    if (!Replaceable && !B->Replaceable)
      return 0;
    // Non-replaceable symbols win.
    return Replaceable ? -1 : 1;
  }

  // As an approximation, regular symbols win over bitcode symbols,
  // but we definitely have a conflict if the regular symbol is not
  // replaceable and neither is the bitcode symbol. We do not
  // replicate the rest of the symbol resolution logic here; symbol
  // resolution will be done accurately after lowering bitcode symbols
  // to regular symbols in addCombinedLTOObject().
  if (isa<DefinedRegular>(Other) && Replaceable)
    return -1;
  if (isa<DefinedCommon>(Other) || isa<DefinedCOMDAT>(Other))
    return -1;
  return 0;
}

int Defined::compare(SymbolBody *Other) {
  if (Other->kind() < kind())
    return -Other->compare(this);
  if (isa<Defined>(Other))
    return 0;
  return 1;
}

int Lazy::compare(SymbolBody *Other) {
  if (Other->kind() < kind())
    return -Other->compare(this);

  // Undefined symbols with weak aliases will turn into defined
  // symbols if they remain undefined, so we don't need to resolve
  // such symbols.
  if (auto *U = dyn_cast<Undefined>(Other))
    if (U->getWeakAlias())
      return -1;
  return 1;
}

int Undefined::compare(SymbolBody *Other) {
  if (Other->kind() < kind())
    return -Other->compare(this);
  if (cast<Undefined>(Other)->getWeakAlias())
    return -1;
  return 1;
}

StringRef DefinedRegular::getName() {
  // DefinedSymbol's name is read lazily for a performance reason.
  // Non-external symbol names are never used by the linker
  // except for logging or debugging.
  // Their internal references are resolved not by name but by symbol index.
  // And because they are not external, no one can refer them by name.
  // Object files contain lots of non-external symbols, and creating
  // StringRefs for them (which involves lots of strlen() on the string table)
  // is a waste of time.
  if (Name.empty())
    COFFFile->getSymbolName(Sym, Name);
  return Name;
}

StringRef DefinedCOMDAT::getName() {
  if (Name.empty())
    COFFFile->getSymbolName(Sym, Name);
  return Name;
}

StringRef DefinedCommon::getName() {
  if (Name.empty())
    COFFFile->getSymbolName(Sym, Name);
  return Name;
}

ErrorOr<std::unique_ptr<InputFile>> Lazy::getMember() {
  auto MBRefOrErr = File->getMember(&Sym);
  if (auto EC = MBRefOrErr.getError())
    return EC;
  MemoryBufferRef MBRef = MBRefOrErr.get();

  // getMember returns an empty buffer if the member was already
  // read from the library.
  if (MBRef.getBuffer().empty())
    return std::unique_ptr<InputFile>(nullptr);

  file_magic Magic = identify_magic(MBRef.getBuffer());
  if (Magic == file_magic::bitcode)
    return std::unique_ptr<InputFile>(new BitcodeFile(MBRef));
  if (Magic == file_magic::coff_import_library)
    return std::unique_ptr<InputFile>(new ImportFile(MBRef));

  if (Magic != file_magic::coff_object) {
    llvm::errs() << File->getName() << ": unknown file type\n";
    return make_error_code(LLDError::InvalidFile);
  }

  std::unique_ptr<InputFile> Obj(new ObjectFile(MBRef));
  Obj->setParentName(File->getName());
  return std::move(Obj);
}

} // namespace coff
} // namespace lld
