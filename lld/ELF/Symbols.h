//===- Symbols.h ------------------------------------------------*- C++ -*-===//
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
template <class ELFT> class ObjectFile;

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
    DefinedFirst = 0,
    DefinedRegularKind = 0,
    DefinedWeakKind = 1,
    DefinedLast = 1,
    UndefinedWeakKind = 2,
    UndefinedKind = 3,
    UndefinedSyntheticKind = 4
  };

  Kind kind() const { return static_cast<Kind>(SymbolKind); }

  bool isStrongUndefined() {
    return SymbolKind == UndefinedKind || SymbolKind == UndefinedSyntheticKind;
  }

  // Returns the symbol name.
  StringRef getName() const { return Name; }

  // A SymbolBody has a backreference to a Symbol. Originally they are
  // doubly-linked. A backreference will never change. But the pointer
  // in the Symbol may be mutated by the resolver. If you have a
  // pointer P to a SymbolBody and are not sure whether the resolver
  // has chosen the object among other objects having the same name,
  // you can access P->Backref->Body to get the resolver's result.
  void setBackref(Symbol *P) { Backref = P; }

  // Decides which symbol should "win" in the symbol table, this or
  // the Other. Returns 1 if this wins, -1 if the Other wins, or 0 if
  // they are duplicate (conflicting) symbols.
  int compare(SymbolBody *Other);

protected:
  SymbolBody(Kind K, StringRef Name) : SymbolKind(K), Name(Name) {}

protected:
  const unsigned SymbolKind : 8;
  StringRef Name;
  Symbol *Backref = nullptr;
};

// This is for symbols created from elf files and not from the command line.
// Since they come from object files, they have a Elf_Sym.
//
// FIXME: Another alternative is to give every symbol an Elf_Sym. To do that
// we have to delay creating the symbol table until the output format is
// known and some of its methods will be templated. We should experiment with
// that once we have a bit more code.
template <class ELFT> class ELFSymbolBody : public SymbolBody {
protected:
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  ELFSymbolBody(Kind K, StringRef Name, const Elf_Sym &Sym)
      : SymbolBody(K, Name), Sym(Sym) {}

public:
  const Elf_Sym &Sym;

  static bool classof(const SymbolBody *S) {
    Kind K = S->kind();
    return K >= DefinedFirst && K <= UndefinedKind;
  }
};

// The base class for any defined symbols, including absolute symbols,
// etc.
template <class ELFT> class Defined : public ELFSymbolBody<ELFT> {
  typedef ELFSymbolBody<ELFT> Base;
  typedef typename Base::Kind Kind;

public:
  typedef typename Base::Elf_Sym Elf_Sym;

  explicit Defined(Kind K, StringRef N, const Elf_Sym &Sym)
      : ELFSymbolBody<ELFT>(K, N, Sym) {}

  static bool classof(const SymbolBody *S) {
    Kind K = S->kind();
    return Base::DefinedFirst <= K && K <= Base::DefinedLast;
  }
};

// Regular defined symbols read from object file symbol tables.
template <class ELFT> class DefinedRegular : public Defined<ELFT> {
  typedef Defined<ELFT> Base;
  typedef typename Base::Elf_Sym Elf_Sym;

public:
  explicit DefinedRegular(StringRef N, const Elf_Sym &Sym)
      : Defined<ELFT>(Base::DefinedRegularKind, N, Sym) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == Base::DefinedRegularKind;
  }
};

template <class ELFT> class DefinedWeak : public Defined<ELFT> {
  typedef Defined<ELFT> Base;
  typedef typename Base::Elf_Sym Elf_Sym;

public:
  explicit DefinedWeak(StringRef N, const Elf_Sym &Sym)
      : Defined<ELFT>(Base::DefinedWeakKind, N, Sym) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == Base::DefinedWeakKind;
  }
};

// Undefined symbols.
class SyntheticUndefined : public SymbolBody {
public:
  explicit SyntheticUndefined(StringRef N) : SymbolBody(UndefinedKind, N) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == UndefinedKind;
  }
};

template <class ELFT> class Undefined : public ELFSymbolBody<ELFT> {
  typedef ELFSymbolBody<ELFT> Base;
  typedef typename Base::Elf_Sym Elf_Sym;

public:
  explicit Undefined(StringRef N, const Elf_Sym &Sym)
      : ELFSymbolBody<ELFT>(Base::UndefinedKind, N, Sym) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == Base::UndefinedKind;
  }
};

template <class ELFT> class UndefinedWeak : public ELFSymbolBody<ELFT> {
  typedef ELFSymbolBody<ELFT> Base;
  typedef typename Base::Elf_Sym Elf_Sym;

public:
  explicit UndefinedWeak(StringRef N, const Elf_Sym &Sym)
      : ELFSymbolBody<ELFT>(Base::UndefinedWeakKind, N, Sym) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == Base::UndefinedWeakKind;
  }
};

} // namespace elf2
} // namespace lld

#endif
