//===- Symbols.h ----------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_SYMBOLS_H
#define LLD_COFF_SYMBOLS_H

#include "Chunks.h"
#include "Config.h"
#include "lld/Core/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/COFF.h"
#include <memory>
#include <vector>

namespace lld {
namespace coff {

using llvm::object::Archive;
using llvm::object::COFFObjectFile;
using llvm::object::COFFSymbolRef;
using llvm::object::coff_import_header;

class ArchiveFile;
class InputFile;
class ObjectFile;
class SymbolBody;

// A real symbol object, SymbolBody, is usually accessed indirectly
// through a Symbol. There's always one Symbol for each symbol name.
// The resolver updates SymbolBody pointers as it resolves symbols.
struct Symbol {
  explicit Symbol(SymbolBody *P) : Body(P) {}
  SymbolBody *Body;
};

// The base class for real symbol classes.
class SymbolBody {
public:
  enum Kind {
    DefinedFirst,
    DefinedBitcodeKind,
    DefinedAbsoluteKind,
    DefinedImportDataKind,
    DefinedImportThunkKind,
    DefinedRegularKind,
    DefinedLast,
    LazyKind,
    UndefinedKind,
  };

  Kind kind() const { return SymbolKind; }
  virtual ~SymbolBody() {}

  // Returns true if this is an external symbol.
  virtual bool isExternal() { return true; }

  // Returns the symbol name.
  virtual StringRef getName() = 0;

  // A SymbolBody has a backreference to a Symbol. Originally they are
  // doubly-linked. A backreference will never change. But the pointer
  // in the Symbol may be mutated by the resolver. If you have a
  // pointer P to a SymbolBody and are not sure whether the resolver
  // has chosen the object among other objects having the same name,
  // you can access P->Backref->Body to get the resolver's result.
  void setBackref(Symbol *P) { Backref = P; }
  SymbolBody *getReplacement() { return Backref ? Backref->Body : this; }

  // Decides which symbol should "win" in the symbol table, this or
  // the Other. Returns 1 if this wins, -1 if the Other wins, or 0 if
  // they are duplicate (conflicting) symbols.
  virtual int compare(SymbolBody *Other) = 0;

protected:
  SymbolBody(Kind K) : SymbolKind(K) {}

private:
  const Kind SymbolKind;
  Symbol *Backref = nullptr;
};

// The base class for any defined symbols, including absolute symbols,
// etc.
class Defined : public SymbolBody {
public:
  Defined(Kind K) : SymbolBody(K) {}

  static bool classof(const SymbolBody *S) {
    Kind K = S->kind();
    return DefinedFirst <= K && K <= DefinedLast;
  }

  // Returns the RVA (relative virtual address) of this symbol. The
  // writer sets and uses RVAs.
  virtual uint64_t getRVA() = 0;

  // Returns the file offset of this symbol in the final executable.
  // The writer uses this information to apply relocations.
  virtual uint64_t getFileOff() = 0;

  // Called by the garbage collector. All Defined subclasses should
  // know how to call depending symbols' markLive functions.
  virtual void markLive() {}

  int compare(SymbolBody *Other) override;
};

// Regular defined symbols read from object file symbol tables.
class DefinedRegular : public Defined {
public:
  DefinedRegular(COFFObjectFile *F, COFFSymbolRef S, Chunk *C)
      : Defined(DefinedRegularKind), COFFFile(F), Sym(S), Data(C) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == DefinedRegularKind;
  }

  StringRef getName() override;
  uint64_t getRVA() override { return Data->getRVA() + Sym.getValue(); }
  bool isExternal() override { return Sym.isExternal(); }
  void markLive() override { Data->markLive(); }
  uint64_t getFileOff() override { return Data->getFileOff() + Sym.getValue(); }
  bool isCOMDAT() const { return Data->isCOMDAT(); }
  int compare(SymbolBody *Other) override;

  // Returns true if this is a common symbol.
  bool isCommon() const { return Sym.isCommon(); }
  uint32_t getCommonSize() const { return Sym.getValue(); }

private:
  StringRef Name;
  COFFObjectFile *COFFFile;
  COFFSymbolRef Sym;
  Chunk *Data;
};

// Absolute symbols.
class DefinedAbsolute : public Defined {
public:
  DefinedAbsolute(StringRef N, uint64_t VA)
      : Defined(DefinedAbsoluteKind), Name(N), RVA(VA - Config->ImageBase) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == DefinedAbsoluteKind;
  }

  StringRef getName() override { return Name; }
  uint64_t getRVA() override { return RVA; }
  uint64_t getFileOff() override { llvm_unreachable("internal error"); }

private:
  StringRef Name;
  uint64_t RVA;
};

// This class represents a symbol defined in an archive file. It is
// created from an archive file header, and it knows how to load an
// object file from an archive to replace itself with a defined
// symbol. If the resolver finds both Undefined and Lazy for
// the same name, it will ask the Lazy to load a file.
class Lazy : public SymbolBody {
public:
  Lazy(ArchiveFile *F, const Archive::Symbol S)
      : SymbolBody(LazyKind), Name(S.getName()), File(F), Sym(S) {}

  static bool classof(const SymbolBody *S) { return S->kind() == LazyKind; }
  StringRef getName() override { return Name; }

  // Returns an object file for this symbol, or a nullptr if the file
  // was already returned.
  ErrorOr<std::unique_ptr<InputFile>> getMember();

  int compare(SymbolBody *Other) override;

private:
  StringRef Name;
  ArchiveFile *File;
  const Archive::Symbol Sym;
};

// Undefined symbols.
class Undefined : public SymbolBody {
public:
  explicit Undefined(StringRef N, SymbolBody **S = nullptr)
      : SymbolBody(UndefinedKind), Name(N), Alias(S) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == UndefinedKind;
  }
  StringRef getName() override { return Name; }

  // An undefined symbol can have a fallback symbol which gives an
  // undefined symbol a second chance if it would remain undefined.
  // If it remains undefined, it'll be replaced with whatever the
  // Alias pointer points to.
  SymbolBody *getWeakAlias() { return Alias ? *Alias : nullptr; }

  int compare(SymbolBody *Other) override;

private:
  StringRef Name;
  SymbolBody **Alias;
};

// Windows-specific classes.

// This class represents a symbol imported from a DLL. This has two
// names for internal use and external use. The former is used for
// name resolution, and the latter is used for the import descriptor
// table in an output. The former has "__imp_" prefix.
class DefinedImportData : public Defined {
public:
  DefinedImportData(StringRef D, StringRef N, StringRef E,
                    const coff_import_header *H)
      : Defined(DefinedImportDataKind), Name(N), DLLName(D), ExternalName(E),
        Hdr(H) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == DefinedImportDataKind;
  }

  StringRef getName() override { return Name; }
  uint64_t getRVA() override { return Location->getRVA(); }
  uint64_t getFileOff() override { return Location->getFileOff(); }
  StringRef getDLLName() { return DLLName; }
  StringRef getExternalName() { return ExternalName; }
  void setLocation(Chunk *AddressTable) { Location = AddressTable; }
  uint16_t getOrdinal() { return Hdr->OrdinalHint; }

private:
  StringRef Name;
  StringRef DLLName;
  StringRef ExternalName;
  const coff_import_header *Hdr;
  Chunk *Location = nullptr;
};

// This class represents a symbol for a jump table entry which jumps
// to a function in a DLL. Linker are supposed to create such symbols
// without "__imp_" prefix for all function symbols exported from
// DLLs, so that you can call DLL functions as regular functions with
// a regular name. A function pointer is given as a DefinedImportData.
class DefinedImportThunk : public Defined {
public:
  DefinedImportThunk(StringRef N, DefinedImportData *S)
      : Defined(DefinedImportThunkKind), Name(N), Data(S) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == DefinedImportThunkKind;
  }

  StringRef getName() override { return Name; }
  uint64_t getRVA() override { return Data.getRVA(); }
  uint64_t getFileOff() override { return Data.getFileOff(); }
  Chunk *getChunk() { return &Data; }

private:
  StringRef Name;
  ImportThunkChunk Data;
};

class DefinedBitcode : public Defined {
public:
  DefinedBitcode(StringRef N, bool R)
      : Defined(DefinedBitcodeKind), Name(N), Replaceable(R) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == DefinedBitcodeKind;
  }

  StringRef getName() override { return Name; }
  uint64_t getRVA() override { llvm_unreachable("bitcode reached writer"); }
  uint64_t getFileOff() override { llvm_unreachable("bitcode reached writer"); }
  int compare(SymbolBody *Other) override;

private:
  StringRef Name;
  bool Replaceable;
};

} // namespace coff
} // namespace lld

#endif
