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

#include "Chunks.h"

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
    DefinedAbsoluteKind = 1,
    DefinedCommonKind = 2,
    DefinedLast = 2,
    UndefinedKind = 3,
    UndefinedSyntheticKind = 4
  };

  Kind kind() const { return static_cast<Kind>(SymbolKind); }

  bool isWeak() const { return IsWeak; }
  bool isUndefined() const {
    return SymbolKind == UndefinedKind || SymbolKind == UndefinedSyntheticKind;
  }
  bool isDefined() const { return !isUndefined(); }
  bool isStrongUndefined() const { return !IsWeak && isUndefined(); }
  bool isCommon() const { return SymbolKind == DefinedCommonKind; }

  // Returns the symbol name.
  StringRef getName() const { return Name; }

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
  template <class ELFT> int compare(SymbolBody *Other);

protected:
  SymbolBody(Kind K, StringRef Name, bool IsWeak)
      : SymbolKind(K), IsWeak(IsWeak), Name(Name) {}

protected:
  const unsigned SymbolKind : 8;
  const unsigned IsWeak : 1;
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
      : SymbolBody(K, Name, Sym.getBinding() == llvm::ELF::STB_WEAK), Sym(Sym) {
  }

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

protected:
  typedef typename Base::Kind Kind;
  typedef typename Base::Elf_Sym Elf_Sym;

public:
  explicit Defined(Kind K, StringRef N, const Elf_Sym &Sym)
      : ELFSymbolBody<ELFT>(K, N, Sym) {}

  static bool classof(const SymbolBody *S) { return S->isDefined(); }
};

template <class ELFT> class DefinedAbsolute : public Defined<ELFT> {
  typedef ELFSymbolBody<ELFT> Base;
  typedef typename Base::Elf_Sym Elf_Sym;

public:
  explicit DefinedAbsolute(StringRef N, const Elf_Sym &Sym)
      : Defined<ELFT>(Base::DefinedAbsoluteKind, N, Sym) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == Base::DefinedAbsoluteKind;
  }
};

template <class ELFT> class DefinedCommon : public Defined<ELFT> {
  typedef ELFSymbolBody<ELFT> Base;
  typedef typename Base::Elf_Sym Elf_Sym;

public:
  explicit DefinedCommon(StringRef N, const Elf_Sym &Sym)
      : Defined<ELFT>(Base::DefinedCommonKind, N, Sym) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == Base::DefinedCommonKind;
  }
};

// Regular defined symbols read from object file symbol tables.
template <class ELFT> class DefinedRegular : public Defined<ELFT> {
  typedef Defined<ELFT> Base;
  typedef typename Base::Elf_Sym Elf_Sym;

public:
  explicit DefinedRegular(StringRef N, const Elf_Sym &Sym,
                          SectionChunk<ELFT> &Section)
      : Defined<ELFT>(Base::DefinedRegularKind, N, Sym), Section(Section) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == Base::DefinedRegularKind;
  }

  const SectionChunk<ELFT> &Section;
};

// Undefined symbols.
class SyntheticUndefined : public SymbolBody {
public:
  explicit SyntheticUndefined(StringRef N)
      : SymbolBody(UndefinedKind, N, false) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == UndefinedSyntheticKind;
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

} // namespace elf2
} // namespace lld

#endif
