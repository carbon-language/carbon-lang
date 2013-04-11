//===- lib/ReaderWriter/ELF/Atoms.h ---------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_ATOMS_H
#define LLD_READER_WRITER_ELF_ATOMS_H

#include "TargetHandler.h"

#include "lld/Core/LLVM.h"
#include "lld/ReaderWriter/Simple.h"

#include "llvm/ADT/ArrayRef.h"

#include <memory>
#include <vector>

namespace lld {
namespace elf {
template <class ELFT> class DynamicFile;
template <typename ELFT> class ELFFile;
template <typename ELFT> class TargetAtomHandler;

/// \brief Relocation References: Defined Atoms may contain references that will
/// need to be patched before the executable is written.
template <class ELFT> class ELFReference LLVM_FINAL : public Reference {
  typedef llvm::object::Elf_Rel_Impl<ELFT, false> Elf_Rel;
  typedef llvm::object::Elf_Rel_Impl<ELFT, true> Elf_Rela;
public:

  ELFReference(const Elf_Rela *rela, uint64_t offset, const Atom *target,
               Kind kind, uint32_t symbol)
      : _target(target), _targetSymbolIndex(symbol),
        _offsetInAtom(offset), _addend(rela->r_addend) {
    _kind = kind;
  }

  ELFReference(const Elf_Rel *rel, uint64_t offset, const Atom *target,
               Kind kind, uint32_t symbol)
      : _target(target), _targetSymbolIndex(symbol),
        _offsetInAtom(offset), _addend(0) {
    _kind = kind;
  }

  ELFReference(Kind kind)
      : _target(nullptr), _targetSymbolIndex(0), _offsetInAtom(0), _addend(0) {
    _kind = kind;
  }

  virtual uint64_t offsetInAtom() const { return _offsetInAtom; }

  virtual const Atom *target() const {
    return _target;
  }

  /// \brief The symbol table index that contains the target reference.
  uint64_t targetSymbolIndex() const {
    return _targetSymbolIndex;
  }

  virtual Addend addend() const {
    return _addend;
  }

  virtual void setOffset(uint64_t off) { _offsetInAtom = off; }

  virtual void setAddend(Addend A) { _addend = A; }

  virtual void setTarget(const Atom *newAtom) { _target = newAtom; }

private:
  const Atom *_target;
  uint64_t _targetSymbolIndex;
  uint64_t _offsetInAtom;
  Addend _addend;
};

/// \brief These atoms store symbols that are fixed to a particular address.
/// This atom has no content its address will be used by the writer to fixup
/// references that point to it.
template<class ELFT>
class ELFAbsoluteAtom LLVM_FINAL : public AbsoluteAtom {
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;

public:
  ELFAbsoluteAtom(const ELFFile<ELFT> &file, StringRef name,
                  const Elf_Sym *symbol, uint64_t value)
      : _owningFile(file), _name(name), _symbol(symbol), _value(value) {
  }

  virtual const ELFFile<ELFT> &file() const {
    return _owningFile;
  } virtual Scope scope() const {
    if (_symbol->st_other == llvm::ELF::STV_HIDDEN)
      return scopeLinkageUnit;
    if (_symbol->getBinding() == llvm::ELF::STB_LOCAL)
      return scopeTranslationUnit;
    else
      return scopeGlobal;
  }

  virtual StringRef name() const {
    return _name;
  }

  virtual uint64_t value() const {
    return _value;
  }

private:
  const ELFFile<ELFT> &_owningFile;
  StringRef _name;
  const Elf_Sym *_symbol;
  uint64_t _value;
};

/// \brief ELFUndefinedAtom: These atoms store undefined symbols and are place
/// holders that will be replaced by defined atoms later in the linking process.
template<class ELFT>
class ELFUndefinedAtom LLVM_FINAL : public lld::UndefinedAtom {
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;

public:
  ELFUndefinedAtom(const File &file, StringRef name, const Elf_Sym *symbol)
      : _owningFile(file), _name(name), _symbol(symbol) {}

  virtual const File &file() const { return _owningFile; }

  virtual StringRef name() const { return _name; }

  // FIXME: What distinguishes a symbol in ELF that can help decide if the
  // symbol is undefined only during build and not runtime? This will make us
  // choose canBeNullAtBuildtime and canBeNullAtRuntime.
  virtual CanBeNull canBeNull() const {
    if (_symbol->getBinding() == llvm::ELF::STB_WEAK)
      return CanBeNull::canBeNullAtBuildtime;
    else
      return CanBeNull::canBeNullNever;
  }

private:
  const File &_owningFile;
  StringRef _name;
  const Elf_Sym *_symbol;
};

/// \brief This atom stores defined symbols and will contain either data or
/// code.
template<class ELFT>
class ELFDefinedAtom LLVM_FINAL : public DefinedAtom {
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;
  typedef llvm::object::Elf_Shdr_Impl<ELFT> Elf_Shdr;

public:
  ELFDefinedAtom(const ELFFile<ELFT> &file,
                 StringRef symbolName,
                 StringRef sectionName,
                 const Elf_Sym *symbol,
                 const Elf_Shdr *section,
                 llvm::ArrayRef<uint8_t> contentData,
                 unsigned int referenceStart,
                 unsigned int referenceEnd,
                 std::vector<ELFReference<ELFT>*> &referenceList)

    : _owningFile(file)
    , _symbolName(symbolName)
    , _sectionName(sectionName)
    , _symbol(symbol)
    , _section(section)
    , _contentData(contentData)
    , _referenceStartIndex(referenceStart)
    , _referenceEndIndex(referenceEnd)
    , _referenceList(referenceList)
    , _targetAtomHandler(nullptr)
    , _contentType(typeUnknown)
    , _permissions(permUnknown) {}

  virtual const ELFFile<ELFT> &file() const {
    return _owningFile;
  }

  virtual StringRef name() const {
    return _symbolName;
  }

  virtual uint64_t ordinal() const {
    return _ordinal;
  }

  const Elf_Sym *symbol() const { return _symbol; }

  const Elf_Shdr *section() const { return _section; }

  virtual uint64_t size() const {
    // Common symbols are not allocated in object files,
    // so use st_size to tell how many bytes are required.

    // Treat target defined common symbols
    if ((_symbol->st_shndx > llvm::ELF::SHN_LOPROC &&
         _symbol->st_shndx < llvm::ELF::SHN_HIPROC)) {
      if (!_targetAtomHandler) {
        const ELFTargetInfo &eti = (_owningFile.getTargetInfo());
        TargetHandler<ELFT> &TargetHandler = eti.getTargetHandler<ELFT>();
        _targetAtomHandler = &TargetHandler.targetAtomHandler();
      }
      if (_targetAtomHandler->getType(_symbol) == llvm::ELF::STT_COMMON)
        return (uint64_t) _symbol->st_size;
    }
    if ((_symbol->getType() == llvm::ELF::STT_COMMON) ||
        _symbol->st_shndx == llvm::ELF::SHN_COMMON)
      return (uint64_t) _symbol->st_size;

    return _contentData.size();
  }

  virtual Scope scope() const {
    if (_symbol->st_other == llvm::ELF::STV_HIDDEN)
      return scopeLinkageUnit;
    else if (_symbol->getBinding() != llvm::ELF::STB_LOCAL)
      return scopeGlobal;
    else
      return scopeTranslationUnit;
  }

  // FIXME: Need to revisit this in future.
  virtual Interposable interposable() const {
    return interposeNo;
  }

  // FIXME: What ways can we determine this in ELF?
  virtual Merge merge() const {
    if (_symbol->getBinding() == llvm::ELF::STB_WEAK)
      return mergeAsWeak;

    // If the symbol is a target defined and if the target
    // defines the symbol as a common symbol treat it as
    // mergeTentative
    if ((_symbol->st_shndx > llvm::ELF::SHN_LOPROC &&
         _symbol->st_shndx < llvm::ELF::SHN_HIPROC)) {
      if (!_targetAtomHandler) {
        const ELFTargetInfo &eti = (_owningFile.getTargetInfo());
        TargetHandler<ELFT> &TargetHandler = eti.getTargetHandler<ELFT>();
        _targetAtomHandler = &TargetHandler.targetAtomHandler();
      }
      if (_targetAtomHandler->getType(_symbol) == llvm::ELF::STT_COMMON)
        return mergeAsTentative;
    }

    if ((_symbol->getType() == llvm::ELF::STT_COMMON) ||
        _symbol->st_shndx == llvm::ELF::SHN_COMMON)
      return mergeAsTentative;

    return mergeNo;
  }

  virtual ContentType contentType() const {
    if (_contentType != typeUnknown)
      return _contentType;

    ContentType ret = typeUnknown;
    uint64_t flags = _section->sh_flags;

    // Treat target defined symbols
    if ((_section->sh_flags & llvm::ELF::SHF_MASKPROC) ||
        ((_symbol->st_shndx > llvm::ELF::SHN_LOPROC &&
          _symbol->st_shndx < llvm::ELF::SHN_HIPROC))) {
      if (!_targetAtomHandler) {
        const ELFTargetInfo &eti = (_owningFile.getTargetInfo());
        TargetHandler<ELFT> &TargetHandler = eti.getTargetHandler<ELFT>();
        _targetAtomHandler = &TargetHandler.targetAtomHandler();
      }
      return _contentType = _targetAtomHandler->contentType(this);
    }

    if (_section->sh_flags ==
        (llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_WRITE | llvm::ELF::SHF_TLS)) {
      return _contentType = _section->sh_type == llvm::ELF::SHT_NOBITS ? typeTLVInitialZeroFill
                                                        : typeTLVInitialData;
    }

    if (_symbol->getType() == llvm::ELF::STT_GNU_IFUNC)
      return _contentType = typeResolver;

    if (_symbol->st_shndx == llvm::ELF::SHN_COMMON)
      return _contentType = typeZeroFill;

    switch (_section->sh_type) {
    case llvm::ELF::SHT_PROGBITS:
      flags &= ~llvm::ELF::SHF_ALLOC;
      flags &= ~llvm::ELF::SHF_GROUP;
      switch (flags) {
      case llvm::ELF::SHF_EXECINSTR:
      case (llvm::ELF::SHF_WRITE|llvm::ELF::SHF_EXECINSTR):
        ret = typeCode;
        break;
      case llvm::ELF::SHF_WRITE:
        ret = typeData;
        break;
      case (llvm::ELF::SHF_MERGE|llvm::ELF::SHF_STRINGS):
      case llvm::ELF::SHF_STRINGS:
        ret = typeConstant;
        break;
      default:
        ret = typeCode;
        break;
      }
      break;
    case llvm::ELF::SHT_NOBITS:
      ret = typeZeroFill;
      break;
    case llvm::ELF::SHT_NULL:
      if ((_symbol->getType() == llvm::ELF::STT_COMMON)
          || _symbol->st_shndx == llvm::ELF::SHN_COMMON)
        ret = typeZeroFill;
      break;
    case llvm::ELF::SHT_INIT_ARRAY:
      ret = typeData;
      break;
    }

    return _contentType = ret;
  }

  virtual Alignment alignment() const {
    // Unallocated common symbols specify their alignment constraints in
    // st_value.

    // Treat target defined common symbols
    if ((_symbol->st_shndx > llvm::ELF::SHN_LOPROC &&
         _symbol->st_shndx < llvm::ELF::SHN_HIPROC)) {
      if (!_targetAtomHandler) {
        const ELFTargetInfo &eti = (_owningFile.getTargetInfo());
        TargetHandler<ELFT> &TargetHandler = eti.getTargetHandler<ELFT>();
        _targetAtomHandler = &TargetHandler.targetAtomHandler();
      }
      if (_targetAtomHandler->getType(_symbol) == llvm::ELF::STT_COMMON)
        return Alignment(llvm::Log2_64(_symbol->st_value));
    }

    if ((_symbol->getType() == llvm::ELF::STT_COMMON) ||
        _symbol->st_shndx == llvm::ELF::SHN_COMMON) {
      return Alignment(llvm::Log2_64(_symbol->st_value));
    }
    return Alignment(llvm::Log2_64(_section->sh_addralign),
                     _symbol->st_value % _section->sh_addralign);
  }

  // Do we have a choice for ELF?  All symbols live in explicit sections.
  virtual SectionChoice sectionChoice() const {
    if (_symbol->st_shndx > llvm::ELF::SHN_LORESERVE)
      return sectionBasedOnContent;

    return sectionCustomRequired;
  }

  virtual StringRef customSectionName() const {
    if ((contentType() == typeZeroFill) ||
        (_symbol->st_shndx == llvm::ELF::SHN_COMMON))
      return ".bss";
    return _sectionName;
  }

  virtual SectionPosition sectionPosition() const {
    return sectionPositionAny;
  }

  // It isn't clear that __attribute__((used)) is transmitted to the ELF object
  // file.
  virtual DeadStripKind deadStrip() const {
    return deadStripNormal;
  }

  virtual ContentPermissions permissions() const {
    if (_permissions != permUnknown)
      return _permissions;

    uint64_t flags = _section->sh_flags;
    // Treat target defined symbols
    if ((_symbol->st_shndx > llvm::ELF::SHN_LOPROC &&
         _symbol->st_shndx < llvm::ELF::SHN_HIPROC)) {
      if (!_targetAtomHandler) {
        const ELFTargetInfo &eti = (_owningFile.getTargetInfo());
        TargetHandler<ELFT> &TargetHandler = eti.getTargetHandler<ELFT>();
        _targetAtomHandler = &TargetHandler.targetAtomHandler();
      }
      return _permissions = _targetAtomHandler->contentPermissions(this);
    }
    switch (_section->sh_type) {
    // permRW_L is for sections modified by the runtime
    // loader.
    case llvm::ELF::SHT_REL:
    case llvm::ELF::SHT_RELA:
      return _permissions = permRW_L;

    case llvm::ELF::SHT_DYNAMIC:
    case llvm::ELF::SHT_PROGBITS:
      flags &= ~llvm::ELF::SHF_ALLOC;
      flags &= ~llvm::ELF::SHF_GROUP;
      switch (flags) {
      // Code
      case llvm::ELF::SHF_EXECINSTR:
        return _permissions = permR_X;
      case (llvm::ELF::SHF_WRITE|llvm::ELF::SHF_EXECINSTR):
        return _permissions = permRWX;
      // Data
      case llvm::ELF::SHF_WRITE:
        return _permissions = permRW_;
      // Strings
      case llvm::ELF::SHF_MERGE:
      case llvm::ELF::SHF_STRINGS:
        return _permissions = permR__;

      default:
        if (flags & llvm::ELF::SHF_WRITE)
          return _permissions = permRW_;
        return _permissions = permR__;
      }

    case llvm::ELF::SHT_NOBITS:
      return _permissions = permRW_;

    case llvm::ELF::SHT_INIT_ARRAY:
      return _permissions = permRW_;

    default:
      return _permissions = perm___;
    }
  }

  // Many non ARM architectures use ELF file format This not really a place to
  // put a architecture specific method in an atom. A better approach is needed.
  virtual bool isThumb() const {
    return false;
  }

  // FIXME: Not Sure if ELF supports alias atoms. Find out more.
  virtual bool isAlias() const {
    return false;
  }

  virtual llvm::ArrayRef<uint8_t> rawContent() const {
    return _contentData;
  }

  DefinedAtom::reference_iterator begin() const {
    uintptr_t index = _referenceStartIndex;
    const void *it = reinterpret_cast<const void*>(index);
    return reference_iterator(*this, it);
  }

  DefinedAtom::reference_iterator end() const {
    uintptr_t index = _referenceEndIndex;
    const void *it = reinterpret_cast<const void*>(index);
    return reference_iterator(*this, it);
  }

  const Reference *derefIterator(const void *It) const {
    uintptr_t index = reinterpret_cast<uintptr_t>(It);
    assert(index >= _referenceStartIndex);
    assert(index < _referenceEndIndex);
    return ((_referenceList)[index]);
  }

  void incrementIterator(const void *&It) const {
    uintptr_t index = reinterpret_cast<uintptr_t>(It);
    ++index;
    It = reinterpret_cast<const void *>(index);
  }

  void addReference(ELFReference<ELFT> *reference) {
    _referenceList.push_back(reference);
    _referenceEndIndex = _referenceList.size();
  }

  void setOrdinal(uint64_t ord) { _ordinal = ord; }

private:
  const ELFFile<ELFT> &_owningFile;
  StringRef _symbolName;
  StringRef _sectionName;
  const Elf_Sym *_symbol;
  const Elf_Shdr *_section;
  /// \brief Holds the bits that make up the atom.
  llvm::ArrayRef<uint8_t> _contentData;

  uint64_t _ordinal;
  unsigned int _referenceStartIndex;
  unsigned int _referenceEndIndex;
  std::vector<ELFReference<ELFT> *> &_referenceList;
  // Cached size of the TLS segment.
  mutable TargetAtomHandler<ELFT> *_targetAtomHandler;
  mutable ContentType _contentType;
  mutable ContentPermissions _permissions;
};

/// \brief This atom stores mergeable Strings
template <class ELFT> class ELFMergeAtom LLVM_FINAL : public DefinedAtom {
  typedef llvm::object::Elf_Shdr_Impl<ELFT> Elf_Shdr;

public:
  ELFMergeAtom(const ELFFile<ELFT> &file, StringRef sectionName,
               const Elf_Shdr *section, llvm::ArrayRef<uint8_t> contentData,
               uint64_t offset)
      : _owningFile(file), _sectionName(sectionName), _section(section),
        _contentData(contentData), _offset(offset) {
  }

  virtual const class ELFFile<ELFT> &file() const {
    return _owningFile;
  } virtual StringRef name() const {
    return "";
  }

  virtual uint64_t section() const { return _section->sh_name; }

  virtual uint64_t offset() const { return _offset; }

  virtual uint64_t ordinal() const { return _ordinal; }

  virtual uint64_t size() const { return _contentData.size(); }

  virtual Scope scope() const { return scopeTranslationUnit; }

  virtual Interposable interposable() const { return interposeNo; }

  virtual Merge merge() const { return mergeByContent; }

  virtual ContentType contentType() const { return typeConstant; }

  virtual Alignment alignment() const {
    return Alignment(llvm::Log2_64(_section->sh_addralign));
  }

  virtual SectionChoice sectionChoice() const { return sectionCustomRequired; }

  virtual StringRef customSectionName() const { return _sectionName; }

  virtual SectionPosition sectionPosition() const { return sectionPositionAny; }

  virtual DeadStripKind deadStrip() const { return deadStripNormal; }

  virtual ContentPermissions permissions() const { return permR__; }

  virtual bool isThumb() const { return false; }

  virtual bool isAlias() const { return false; }

  virtual llvm::ArrayRef<uint8_t> rawContent() const { return _contentData; }

  DefinedAtom::reference_iterator begin() const {
    uintptr_t index = 0;
    const void *it = reinterpret_cast<const void *>(index);
    return reference_iterator(*this, it);
  }

  DefinedAtom::reference_iterator end() const {
    uintptr_t index = 0;
    const void *it = reinterpret_cast<const void *>(index);
    return reference_iterator(*this, it);
  }

  const Reference *derefIterator(const void *It) const { return nullptr; }

  void incrementIterator(const void *&It) const {}

private:

  const ELFFile<ELFT> &_owningFile;
  StringRef _sectionName;
  const Elf_Shdr *_section;
  /// \brief Holds the bits that make up the atom.
  llvm::ArrayRef<uint8_t> _contentData;
  uint64_t _ordinal;
  uint64_t _offset;
};

/// \brief An atom from a shared library.
template <class ELFT>
class ELFDynamicAtom LLVM_FINAL : public SharedLibraryAtom {
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;

public:
  ELFDynamicAtom(const DynamicFile<ELFT> &file, StringRef symbolName,
                 StringRef loadName, const Elf_Sym *symbol)
      : _owningFile(file), _symbolName(symbolName), _loadName(loadName),
        _symbol(symbol) {
  }

  virtual const DynamicFile<ELFT> &file() const {
    return _owningFile;
  }

  virtual StringRef name() const {
    return _symbolName;
  }

  virtual Scope scope() const {
    if (_symbol->st_other == llvm::ELF::STV_HIDDEN)
      return scopeLinkageUnit;
    else if (_symbol->getBinding() != llvm::ELF::STB_LOCAL)
      return scopeGlobal;
    else
      return scopeTranslationUnit;
  }

  virtual StringRef loadName() const { return _loadName; }

  virtual bool canBeNullAtRuntime() const {
    return _symbol->getBinding() == llvm::ELF::STB_WEAK;
  }

private:

  const DynamicFile<ELFT> &_owningFile;
  StringRef _symbolName;
  StringRef _loadName;
  const Elf_Sym *_symbol;
};

class GOTAtom : public SimpleDefinedAtom {
  StringRef _section;

public:
  GOTAtom(const File &f, StringRef secName)
      : SimpleDefinedAtom(f), _section(secName) {
  }

  virtual Scope scope() const { return scopeTranslationUnit; }

  virtual SectionChoice sectionChoice() const { return sectionCustomRequired; }

  virtual StringRef customSectionName() const { return _section; }

  virtual ContentType contentType() const { return typeGOT; }

  virtual uint64_t size() const { return rawContent().size(); }

  virtual ContentPermissions permissions() const { return permRW_; }

  virtual ArrayRef<uint8_t> rawContent() const = 0;

  virtual Alignment alignment() const {
    // The alignment should be 8 byte aligned
    return Alignment(3);
  }

#ifndef NDEBUG
  virtual StringRef name() const { return _name; }

  std::string _name;
#else
  virtual StringRef name() const { return ""; }
#endif
};

class PLTAtom : public SimpleDefinedAtom {
  StringRef _section;

public:
  PLTAtom(const File &f, StringRef secName)
      : SimpleDefinedAtom(f), _section(secName) {
  }

  virtual Scope scope() const { return scopeTranslationUnit; }

  virtual SectionChoice sectionChoice() const { return sectionCustomRequired; }

  virtual StringRef customSectionName() const { return _section; }

  virtual ContentType contentType() const { return typeStub; }

  virtual uint64_t size() const { return rawContent().size(); }

  virtual ContentPermissions permissions() const { return permR_X; }

  virtual ArrayRef<uint8_t> rawContent() const = 0;

  virtual Alignment alignment() const {
    return Alignment(4); // 16
  }

#ifndef NDEBUG
  virtual StringRef name() const { return _name; }

  std::string _name;
#else
  virtual StringRef name() const { return ""; }
#endif
};

class PLT0Atom : public PLTAtom {

public:
  PLT0Atom(const File &f) : PLTAtom(f, ".plt") {
#ifndef NDEBUG
    _name = ".PLT0";
#endif
  }
};

class GLOBAL_OFFSET_TABLEAtom : public SimpleDefinedAtom {
public:
  GLOBAL_OFFSET_TABLEAtom(const File &f) : SimpleDefinedAtom(f) {}

  virtual StringRef name() const { return "_GLOBAL_OFFSET_TABLE_"; }

  virtual Scope scope() const { return scopeGlobal; }

  virtual SectionChoice sectionChoice() const { return sectionCustomRequired; }

  virtual StringRef customSectionName() const { return ".got.plt"; }

  virtual ContentType contentType() const { return typeGOT; }

  virtual uint64_t size() const { return 0; }

  virtual ContentPermissions permissions() const { return permRW_; }

  virtual Alignment alignment() const {
    // Needs 8 byte alignment
    return Alignment(3);
  }

  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>();
  }
};

class TLSGETADDRAtom : public SimpleDefinedAtom {
public:
  TLSGETADDRAtom(const File &f) : SimpleDefinedAtom(f) {}

  virtual StringRef name() const { return "__tls_get_addr"; }

  virtual Scope scope() const { return scopeGlobal; }

  virtual Merge merge() const { return mergeAsWeak; }

  virtual SectionChoice sectionChoice() const { return sectionCustomRequired; }

  virtual StringRef customSectionName() const { return ".text"; }

  virtual ContentType contentType() const { return typeCode; }

  virtual uint64_t size() const { return 0; }

  virtual ContentPermissions permissions() const { return permR_X; }

  virtual Alignment alignment() const { return Alignment(0); }

  virtual ArrayRef<uint8_t> rawContent() const { return ArrayRef<uint8_t>(); }
};

} // end namespace elf
} // end namespace lld

#endif
