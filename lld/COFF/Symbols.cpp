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

StringRef SymbolBody::getName() {
  // DefinedCOFF names are read lazily for a performance reason.
  // Non-external symbol names are never used by the linker except for logging
  // or debugging. Their internal references are resolved not by name but by
  // symbol index. And because they are not external, no one can refer them by
  // name. Object files contain lots of non-external symbols, and creating
  // StringRefs for them (which involves lots of strlen() on the string table)
  // is a waste of time.
  if (Name.empty()) {
    auto *D = cast<DefinedCOFF>(this);
    D->File->getCOFFObj()->getSymbolName(D->Sym, Name);
  }
  return Name;
}

// Returns 1, 0 or -1 if this symbol should take precedence
// over the Other, tie or lose, respectively.
int SymbolBody::compare(SymbolBody *Other) {
  Kind LK = kind(), RK = Other->kind();

  // Normalize so that the smaller kind is on the left.
  if (LK > RK)
    return -Other->compare(this);

  // First handle comparisons between two different kinds.
  if (LK != RK) {

    if (RK > LastDefinedKind) {
      if (LK == LazyKind && cast<Undefined>(Other)->getWeakAlias())
        return -1;

      // The LHS is either defined or lazy and so it wins.
      assert((LK <= LastDefinedKind || LK == LazyKind) && "Bad kind!");
      return 1;
    }

    // Bitcode has special complexities.
    if (RK == DefinedBitcodeKind) {
      auto *RHS = cast<DefinedBitcode>(Other);

      switch (LK) {
      case DefinedCommonKind:
        return 1;

      case DefinedRegularKind:
        // As an approximation, regular symbols win over bitcode symbols,
        // but we definitely have a conflict if the regular symbol is not
        // replaceable and neither is the bitcode symbol. We do not
        // replicate the rest of the symbol resolution logic here; symbol
        // resolution will be done accurately after lowering bitcode symbols
        // to regular symbols in addCombinedLTOObject().
        if (cast<DefinedRegular>(this)->isCOMDAT() || RHS->IsReplaceable)
          return 1;

        // Fallthrough to the default of a tie otherwise.
      default:
        return 0;
      }
    }

    // Either of the object file kind will trump a higher kind.
    if (LK <= LastDefinedCOFFKind)
      return 1;

    // The remaining kind pairs are ties amongst defined symbols.
    return 0;
  }

  // Now handle the case where the kinds are the same.
  switch (LK) {
  case DefinedRegularKind: {
    auto *LHS = cast<DefinedRegular>(this);
    auto *RHS = cast<DefinedRegular>(Other);
    return (LHS->isCOMDAT() && RHS->isCOMDAT()) ? 1 : 0;
  }

  case DefinedCommonKind: {
    auto *LHS = cast<DefinedCommon>(this);
    auto *RHS = cast<DefinedCommon>(Other);
    return LHS->getSize() > RHS->getSize() ? 1 : -1;
  }

  case DefinedBitcodeKind: {
    auto *LHS = cast<DefinedBitcode>(this);
    auto *RHS = cast<DefinedBitcode>(Other);
    // If both are non-replaceable, we have a tie.
    if (!LHS->IsReplaceable && !RHS->IsReplaceable)
      return 0;

    // Non-replaceable symbols win, but even two replaceable symboles don't
    // tie.
    return LHS->IsReplaceable ? -1 : 1;
  }

  case LazyKind:
    // Don't tie, just pick the LHS.
    return 1;

  case UndefinedKind:
    // Don't tie, just pick the LHS unless the RHS has a weak alias.
    return cast<Undefined>(Other)->getWeakAlias() ? -1 : 1;

  case DefinedLocalImportKind:
  case DefinedImportThunkKind:
  case DefinedImportDataKind:
  case DefinedAbsoluteKind:
    // These all simply tie.
    return 0;
  }
  llvm_unreachable("unknown symbol kind");
}

std::string SymbolBody::getDebugName() {
  std::string N = getName().str();
  if (auto *D = dyn_cast<DefinedCOFF>(this)) {
    N += " ";
    N += D->File->getShortName();
  } else if (auto *D = dyn_cast<DefinedBitcode>(this)) {
    N += " ";
    N += D->File->getShortName();
  }
  return N;
}

uint64_t Defined::getRVA() {
  switch (kind()) {
  case DefinedAbsoluteKind:
    return cast<DefinedAbsolute>(this)->getRVA();
  case DefinedImportDataKind:
    return cast<DefinedImportData>(this)->getRVA();
  case DefinedImportThunkKind:
    return cast<DefinedImportThunk>(this)->getRVA();
  case DefinedLocalImportKind:
    return cast<DefinedLocalImport>(this)->getRVA();
  case DefinedCommonKind:
    return cast<DefinedCommon>(this)->getRVA();
  case DefinedRegularKind:
    return cast<DefinedRegular>(this)->getRVA();

  case DefinedBitcodeKind:
    llvm_unreachable("There is no address for a bitcode symbol.");
  case LazyKind:
  case UndefinedKind:
    llvm_unreachable("Cannot get the address for an undefined symbol.");
  }
  llvm_unreachable("unknown symbol kind");
}

uint64_t Defined::getFileOff() {
  switch (kind()) {
  case DefinedImportDataKind:
    return cast<DefinedImportData>(this)->getFileOff();
  case DefinedImportThunkKind:
    return cast<DefinedImportThunk>(this)->getFileOff();
  case DefinedLocalImportKind:
    return cast<DefinedLocalImport>(this)->getFileOff();
  case DefinedCommonKind:
    return cast<DefinedCommon>(this)->getFileOff();
  case DefinedRegularKind:
    return cast<DefinedRegular>(this)->getFileOff();

  case DefinedBitcodeKind:
    llvm_unreachable("There is no file offset for a bitcode symbol.");
  case DefinedAbsoluteKind:
    llvm_unreachable("Cannot get a file offset for an absolute symbol.");
  case LazyKind:
  case UndefinedKind:
    llvm_unreachable("Cannot get a file offset for an undefined symbol.");
  }
  llvm_unreachable("unknown symbol kind");
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
