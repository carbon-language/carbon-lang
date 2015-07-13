//===- Symbols.h ----------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_SYMBOLS_H
#define LLD_ELF_SYMBOLS_H

#include "Chunks.h"
#include "Config.h"
#include "lld/Core/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ELF.h"
#include <memory>
#include <vector>

namespace lld {
namespace elfv2 {

using llvm::object::Archive;
using llvm::object::ELFFile;

class ArchiveFile;
class InputFile;
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

  // Returns the VA (virtual address) of this symbol. The
  // writer sets and uses VAs.
  virtual uint64_t getVA() = 0;

  // Returns the file offset of this symbol in the final executable.
  // The writer uses this information to apply relocations.
  virtual uint64_t getFileOff() = 0;

  // Called by the garbage collector. All Defined subclasses should
  // know how to call depending symbols' markLive functions.
  virtual void markLive() {}

  int compare(SymbolBody *Other) override;
};

// Regular defined symbols read from object file symbol tables.
template <class ELFT> class DefinedRegular : public Defined {
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;

public:
  DefinedRegular(ELFFile<ELFT> *F, const Elf_Sym *S, Chunk *C)
      : Defined(DefinedRegularKind), File(F), Sym(S), Data(C) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == DefinedRegularKind;
  }

  StringRef getName() override;
  uint64_t getVA() override { return Data->getVA() + Sym->getValue(); }
  bool isExternal() override { return Sym->isExternal(); }
  void markLive() override { Data->markLive(); }
  uint64_t getFileOff() override {
    return Data->getFileOff() + Sym->getValue();
  }
  int compare(SymbolBody *Other) override;

  // Returns true if this is a common symbol.
  bool isCommon() const { return Sym->isCommon(); }
  uint32_t getCommonSize() const { return Sym->st_size; }

private:
  StringRef Name;
  ELFFile<ELFT> *File;
  const Elf_Sym *Sym;
  Chunk *Data;
};

// Absolute symbols.
class DefinedAbsolute : public Defined {
public:
  DefinedAbsolute(StringRef N, uint64_t VA)
      : Defined(DefinedAbsoluteKind), Name(N), VA(VA) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == DefinedAbsoluteKind;
  }

  StringRef getName() override { return Name; }
  uint64_t getVA() override { return VA; }
  uint64_t getFileOff() override { llvm_unreachable("internal error"); }

private:
  StringRef Name;
  uint64_t VA;
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

class DefinedBitcode : public Defined {
public:
  DefinedBitcode(StringRef N, bool R)
      : Defined(DefinedBitcodeKind), Name(N), Replaceable(R) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == DefinedBitcodeKind;
  }

  StringRef getName() override { return Name; }
  uint64_t getVA() override { llvm_unreachable("bitcode reached writer"); }
  uint64_t getFileOff() override { llvm_unreachable("bitcode reached writer"); }
  int compare(SymbolBody *Other) override;

private:
  StringRef Name;
  bool Replaceable;
};

} // namespace elfv2
} // namespace lld

#endif
