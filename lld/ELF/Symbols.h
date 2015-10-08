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

#include "InputSection.h"

#include "lld/Core/LLVM.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ELF.h"

namespace lld {
namespace elf2 {

class ArchiveFile;
class InputFile;
class SymbolBody;
template <class ELFT> class ObjectFile;
template <class ELFT> class OutputSection;

// Initializes global objects defined in this file.
// Called at the beginning of main().
void initSymbols();

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
    DefinedRegularKind = DefinedFirst,
    DefinedAbsoluteKind,
    DefinedCommonKind,
    DefinedSyntheticKind,
    SharedKind,
    DefinedLast = SharedKind,
    UndefinedKind,
    LazyKind
  };

  Kind kind() const { return static_cast<Kind>(SymbolKind); }

  bool isWeak() const { return IsWeak; }
  bool isUndefined() const { return SymbolKind == UndefinedKind; }
  bool isDefined() const { return SymbolKind <= DefinedLast; }
  bool isCommon() const { return SymbolKind == DefinedCommonKind; }
  bool isLazy() const { return SymbolKind == LazyKind; }
  bool isShared() const { return SymbolKind == SharedKind; }
  bool isUsedInRegularObj() const { return IsUsedInRegularObj; }
  bool isUsedInDynamicReloc() const { return IsUsedInDynamicReloc; }
  void setUsedInDynamicReloc() { IsUsedInDynamicReloc = true; }

  // Returns the symbol name.
  StringRef getName() const { return Name; }

  uint8_t getMostConstrainingVisibility() const {
    return MostConstrainingVisibility;
  }

  unsigned getDynamicSymbolTableIndex() const {
    return DynamicSymbolTableIndex;
  }
  void setDynamicSymbolTableIndex(unsigned V) { DynamicSymbolTableIndex = V; }

  unsigned getGotIndex() const { return GotIndex; }
  bool isInGot() const { return GotIndex != -1U; }
  void setGotIndex(unsigned I) { GotIndex = I; }

  unsigned getPltIndex() const { return PltIndex; }
  bool isInPlt() const { return PltIndex != -1U; }
  void setPltIndex(unsigned I) { PltIndex = I; }

  // A SymbolBody has a backreference to a Symbol. Originally they are
  // doubly-linked. A backreference will never change. But the pointer
  // in the Symbol may be mutated by the resolver. If you have a
  // pointer P to a SymbolBody and are not sure whether the resolver
  // has chosen the object among other objects having the same name,
  // you can access P->Backref->Body to get the resolver's result.
  void setBackref(Symbol *P) { Backref = P; }
  SymbolBody *repl() { return Backref ? Backref->Body : this; }

  // Decides which symbol should "win" in the symbol table, this or
  // the Other. Returns 1 if this wins, -1 if the Other wins, or 0 if
  // they are duplicate (conflicting) symbols.
  template <class ELFT> int compare(SymbolBody *Other);

protected:
  SymbolBody(Kind K, StringRef Name, bool IsWeak, uint8_t Visibility)
      : SymbolKind(K), IsWeak(IsWeak), MostConstrainingVisibility(Visibility),
        Name(Name) {
    IsUsedInRegularObj = K != SharedKind && K != LazyKind;
    IsUsedInDynamicReloc = 0;
  }

  const unsigned SymbolKind : 8;
  unsigned IsWeak : 1;
  unsigned MostConstrainingVisibility : 2;
  unsigned IsUsedInRegularObj : 1;
  unsigned IsUsedInDynamicReloc : 1;
  unsigned DynamicSymbolTableIndex = 0;
  unsigned GotIndex = -1;
  unsigned PltIndex = -1;
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
      : SymbolBody(K, Name, Sym.getBinding() == llvm::ELF::STB_WEAK,
                   Sym.getVisibility()),
        Sym(Sym) {}

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
  Defined(Kind K, StringRef N, const Elf_Sym &Sym)
      : ELFSymbolBody<ELFT>(K, N, Sym) {}

  static bool classof(const SymbolBody *S) { return S->isDefined(); }
};

template <class ELFT> class DefinedAbsolute : public Defined<ELFT> {
  typedef ELFSymbolBody<ELFT> Base;
  typedef typename Base::Elf_Sym Elf_Sym;

public:
  static Elf_Sym IgnoreUndef;

  DefinedAbsolute(StringRef N, const Elf_Sym &Sym)
      : Defined<ELFT>(Base::DefinedAbsoluteKind, N, Sym) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == Base::DefinedAbsoluteKind;
  }
};

template <class ELFT>
typename DefinedAbsolute<ELFT>::Elf_Sym DefinedAbsolute<ELFT>::IgnoreUndef;

template <class ELFT> class DefinedCommon : public Defined<ELFT> {
  typedef ELFSymbolBody<ELFT> Base;
  typedef typename Base::Elf_Sym Elf_Sym;

public:
  typedef typename std::conditional<ELFT::Is64Bits, uint64_t, uint32_t>::type
      uintX_t;
  DefinedCommon(StringRef N, const Elf_Sym &Sym)
      : Defined<ELFT>(Base::DefinedCommonKind, N, Sym) {
    MaxAlignment = Sym.st_value;
  }

  static bool classof(const SymbolBody *S) {
    return S->kind() == Base::DefinedCommonKind;
  }

  // The output offset of this common symbol in the output bss. Computed by the
  // writer.
  uintX_t OffsetInBSS;

  // The maximum alignment we have seen for this symbol.
  uintX_t MaxAlignment;
};

// Regular defined symbols read from object file symbol tables.
template <class ELFT> class DefinedRegular : public Defined<ELFT> {
  typedef Defined<ELFT> Base;
  typedef typename Base::Elf_Sym Elf_Sym;

public:
  DefinedRegular(StringRef N, const Elf_Sym &Sym, InputSection<ELFT> &Section)
      : Defined<ELFT>(Base::DefinedRegularKind, N, Sym), Section(Section) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == Base::DefinedRegularKind;
  }

  const InputSection<ELFT> &Section;
};

template <class ELFT> class DefinedSynthetic : public Defined<ELFT> {
  typedef Defined<ELFT> Base;

public:
  typedef typename Base::Elf_Sym Elf_Sym;
  DefinedSynthetic(StringRef N, const Elf_Sym &Sym,
                   OutputSection<ELFT> &Section)
      : Defined<ELFT>(Base::DefinedSyntheticKind, N, Sym), Section(Section) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == Base::DefinedSyntheticKind;
  }

  const OutputSection<ELFT> &Section;
};

// Undefined symbol.
template <class ELFT> class Undefined : public ELFSymbolBody<ELFT> {
  typedef ELFSymbolBody<ELFT> Base;
  typedef typename Base::Elf_Sym Elf_Sym;

public:
  static Elf_Sym Required;
  static Elf_Sym Optional;

  Undefined(StringRef N, const Elf_Sym &Sym)
      : ELFSymbolBody<ELFT>(Base::UndefinedKind, N, Sym) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == Base::UndefinedKind;
  }

  bool canKeepUndefined() const { return &this->Sym == &Optional; }
};

template <class ELFT>
typename Undefined<ELFT>::Elf_Sym Undefined<ELFT>::Required;
template <class ELFT>
typename Undefined<ELFT>::Elf_Sym Undefined<ELFT>::Optional;

template <class ELFT> class SharedSymbol : public Defined<ELFT> {
  typedef Defined<ELFT> Base;
  typedef typename Base::Elf_Sym Elf_Sym;

public:
  static bool classof(const SymbolBody *S) {
    return S->kind() == Base::SharedKind;
  }

  SharedSymbol(StringRef Name, const Elf_Sym &Sym)
      : Defined<ELFT>(Base::SharedKind, Name, Sym) {}
};

// This class represents a symbol defined in an archive file. It is
// created from an archive file header, and it knows how to load an
// object file from an archive to replace itself with a defined
// symbol. If the resolver finds both Undefined and Lazy for
// the same name, it will ask the Lazy to load a file.
class Lazy : public SymbolBody {
public:
  Lazy(ArchiveFile *F, const llvm::object::Archive::Symbol S)
      : SymbolBody(LazyKind, S.getName(), false, llvm::ELF::STV_DEFAULT),
        File(F), Sym(S) {}

  static bool classof(const SymbolBody *S) { return S->kind() == LazyKind; }

  // Returns an object file for this symbol, or a nullptr if the file
  // was already returned.
  std::unique_ptr<InputFile> getMember();

  void setWeak() { IsWeak = true; }
  void setUsedInRegularObj() { IsUsedInRegularObj = true; }

private:
  ArchiveFile *File;
  const llvm::object::Archive::Symbol Sym;
};

} // namespace elf2
} // namespace lld

#endif
