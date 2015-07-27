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

#include "lld/Core/LLVM.h"
#include "llvm/Object/ELF.h"

namespace lld {
namespace elf2 {

using llvm::object::ELFFile;

class Chunk;
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
    DefinedRegularKind,
    DefinedLast,
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
  int compare(SymbolBody *Other);

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
};

// Regular defined symbols read from object file symbol tables.
template <class ELFT> class DefinedRegular : public Defined {
public:
  DefinedRegular(StringRef Name) : Defined(DefinedRegularKind), Name(Name) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == DefinedRegularKind;
  }

  StringRef getName() override { return Name; }

private:
  StringRef Name;
};

// Undefined symbols.
class Undefined : public SymbolBody {
public:
  explicit Undefined(StringRef N) : SymbolBody(UndefinedKind), Name(N) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == UndefinedKind;
  }
  StringRef getName() override { return Name; }

private:
  StringRef Name;
};

} // namespace elf2
} // namespace lld

#endif
