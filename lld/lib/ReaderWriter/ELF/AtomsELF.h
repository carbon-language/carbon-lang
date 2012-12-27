#ifndef LLD_ELFATOMS_H_
#define LLD_ELFATOMS_H_

#include "lld/Core/LLVM.h"
#include <memory>
#include <vector>

namespace lld {
/// \brief Relocation References: Defined Atoms may contain references that will
/// need to be patched before the executable is written.
template <llvm::support::endianness target_endianness, bool is64Bits>
class ELFReference final : public Reference {
  typedef llvm::object::Elf_Rel_Impl
                        <target_endianness, is64Bits, false> Elf_Rel;
  typedef llvm::object::Elf_Rel_Impl
                        <target_endianness, is64Bits, true> Elf_Rela;
public:

  ELFReference(const Elf_Rela *rela, uint64_t offset, const Atom *target)
    : _target(target)
    , _targetSymbolIndex(rela->getSymbol())
    , _offsetInAtom(offset)
    , _addend(rela->r_addend)
    , _kind(rela->getType()) {}

  ELFReference(const Elf_Rel *rel, uint64_t offset, const Atom *target)
    : _target(target)
    , _targetSymbolIndex(rel->getSymbol())
    , _offsetInAtom(offset)
    , _addend(0)
    , _kind(rel->getType()) {}

  virtual uint64_t offsetInAtom() const {
    return _offsetInAtom;
  }

  virtual Kind kind() const {
    return _kind;
  }

  virtual void setKind(Kind kind) {
    _kind = kind;
  }

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

  virtual void setAddend(Addend A) {
    _addend = A;
  }

  virtual void setTarget(const Atom *newAtom) {
    _target = newAtom;
  }
private:
  const Atom  *_target;
  uint64_t     _targetSymbolIndex;
  uint64_t     _offsetInAtom;
  Addend       _addend;
  Kind         _kind;
};

/// \brief These atoms store symbols that are fixed to a particular address.
/// This atom has no content its address will be used by the writer to fixup
/// references that point to it.
template<llvm::support::endianness target_endianness, bool is64Bits>
class ELFAbsoluteAtom final : public AbsoluteAtom {
  typedef llvm::object::Elf_Sym_Impl<target_endianness, is64Bits> Elf_Sym;

public:
  ELFAbsoluteAtom(const File &file,
                  llvm::StringRef name,
                  const Elf_Sym *symbol,
                  uint64_t value)
    : _owningFile(file)
    , _name(name)
    , _symbol(symbol)
    , _value(value)
  {}

  virtual const class File &file() const {
    return _owningFile;
  }

  virtual Scope scope() const {
    if (_symbol->st_other == llvm::ELF::STV_HIDDEN)
      return scopeLinkageUnit;
    if (_symbol->getBinding() == llvm::ELF::STB_LOCAL)
      return scopeTranslationUnit;
    else
      return scopeGlobal;
  }

  virtual llvm::StringRef name() const {
    return _name;
  }

  virtual uint64_t value() const {
    return _value;
  }

private:
  const File &_owningFile;
  llvm::StringRef _name;
  const Elf_Sym *_symbol;
  uint64_t _value;
};

/// \brief ELFUndefinedAtom: These atoms store undefined symbols and are place
/// holders that will be replaced by defined atoms later in the linking process.
template<llvm::support::endianness target_endianness, bool is64Bits>
class ELFUndefinedAtom final: public UndefinedAtom {
  typedef llvm::object::Elf_Sym_Impl<target_endianness, is64Bits> Elf_Sym;

public:
  ELFUndefinedAtom(const File &file,
                   llvm::StringRef name,
                   const Elf_Sym *symbol)
    : _owningFile(file)
    , _name(name)
    , _symbol(symbol)
  {}

  virtual const class File &file() const {
    return _owningFile;
  }

  virtual llvm::StringRef name() const {
    return _name;
  }

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
  llvm::StringRef _name;
  const Elf_Sym *_symbol;
};

/// \brief This atom stores defined symbols and will contain either data or
/// code.
template<llvm::support::endianness target_endianness, bool is64Bits>
class ELFDefinedAtom final: public DefinedAtom {
  typedef llvm::object::Elf_Sym_Impl<target_endianness, is64Bits> Elf_Sym;
  typedef llvm::object::Elf_Shdr_Impl<target_endianness, is64Bits> Elf_Shdr;

public:
  ELFDefinedAtom(const File &file,
                 llvm::StringRef symbolName,
                 llvm::StringRef sectionName,
                 const Elf_Sym *symbol,
                 const Elf_Shdr *section,
                 llvm::ArrayRef<uint8_t> contentData,
                 unsigned int referenceStart,
                 unsigned int referenceEnd,
                 std::vector<ELFReference
                             <target_endianness, is64Bits> *> &referenceList)

    : _owningFile(file)
    , _symbolName(symbolName)
    , _sectionName(sectionName)
    , _symbol(symbol)
    , _section(section)
    , _contentData(contentData)
    , _referenceStartIndex(referenceStart)
    , _referenceEndIndex(referenceEnd)
    , _referenceList(referenceList) {
    static uint64_t orderNumber = 0;
    _ordinal = ++orderNumber;
  }

  virtual const class File &file() const {
    return _owningFile;
  }

  virtual llvm::StringRef name() const {
    return _symbolName;
  }

  virtual uint64_t ordinal() const {
    return _ordinal;
  }

  virtual uint64_t size() const {
    // Common symbols are not allocated in object files,
    // so use st_size to tell how many bytes are required.
    if ((_symbol->getType() == llvm::ELF::STT_COMMON)
        || _symbol->st_shndx == llvm::ELF::SHN_COMMON)
      return (uint64_t)_symbol->st_size;

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

    if ((_symbol->getType() == llvm::ELF::STT_COMMON)
        || _symbol->st_shndx == llvm::ELF::SHN_COMMON)
      return mergeAsTentative;

    return mergeNo;
  }

  virtual ContentType contentType() const {

    ContentType ret = typeUnknown;
    uint64_t flags = _section->sh_flags;

    if (_symbol->st_shndx == llvm::ELF::SHN_COMMON)
      return typeZeroFill;

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
    }

    return ret;
  }

  virtual Alignment alignment() const {
    // Unallocated common symbols specify their alignment constraints in
    // st_value.
    if ((_symbol->getType() == llvm::ELF::STT_COMMON)
        || _symbol->st_shndx == llvm::ELF::SHN_COMMON) {
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

  virtual llvm::StringRef customSectionName() const {
    if ((contentType() == typeZeroFill) ||
        (_symbol->st_shndx == llvm::ELF::SHN_COMMON)) 
      return ".bss";
    return _sectionName;
  }

  // It isn't clear that __attribute__((used)) is transmitted to the ELF object
  // file.
  virtual DeadStripKind deadStrip() const {
    return deadStripNormal;
  }

  virtual ContentPermissions permissions() const {
    uint64_t flags = _section->sh_flags;
    switch (_section->sh_type) {
    // permRW_L is for sections modified by the runtime
    // loader.
    case llvm::ELF::SHT_REL:
    case llvm::ELF::SHT_RELA:
      return permRW_L;

    case llvm::ELF::SHT_DYNAMIC:
    case llvm::ELF::SHT_PROGBITS:
      flags &= ~llvm::ELF::SHF_ALLOC;
      flags &= ~llvm::ELF::SHF_GROUP;
      switch (flags) {
      // Code
      case llvm::ELF::SHF_EXECINSTR:
        return permR_X;
      case (llvm::ELF::SHF_WRITE|llvm::ELF::SHF_EXECINSTR):
        return permRWX;
      // Data
      case llvm::ELF::SHF_WRITE:
        return permRW_;
      // Strings
      case llvm::ELF::SHF_MERGE:
      case llvm::ELF::SHF_STRINGS:
        return permR__;

      default:
        if (flags & llvm::ELF::SHF_WRITE) 
          return permRW_;
        return permR__;
      }

    case llvm::ELF::SHT_NOBITS:
      return permRW_;

    default:
      return perm___;
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

  void incrementIterator(const void*& It) const {
    uintptr_t index = reinterpret_cast<uintptr_t>(It);
    ++index;
    It = reinterpret_cast<const void*>(index);
  }

private:

  const File &_owningFile;
  llvm::StringRef _symbolName;
  llvm::StringRef _sectionName;
  const Elf_Sym *_symbol;
  const Elf_Shdr *_section;
  /// \brief Holds the bits that make up the atom.
  llvm::ArrayRef<uint8_t> _contentData;

  uint64_t _ordinal;
  unsigned int _referenceStartIndex;
  unsigned int _referenceEndIndex;
  std::vector<ELFReference<target_endianness, is64Bits> *> &_referenceList;
};
} // namespace lld
#endif
