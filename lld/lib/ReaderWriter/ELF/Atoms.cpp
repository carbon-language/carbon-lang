//===- lib/ReaderWriter/ELF/Atoms.cpp -------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Atoms.h"
#include "DynamicFile.h"
#include "ELFFile.h"
#include "TargetHandler.h"

namespace lld {
namespace elf {

template <class ELFT> AbsoluteAtom::Scope ELFAbsoluteAtom<ELFT>::scope() const {
  if (_symbol->getVisibility() == llvm::ELF::STV_HIDDEN)
    return scopeLinkageUnit;
  if (_symbol->getBinding() == llvm::ELF::STB_LOCAL)
    return scopeTranslationUnit;
  return scopeGlobal;
}

template <class ELFT>
UndefinedAtom::CanBeNull ELFUndefinedAtom<ELFT>::canBeNull() const {
  if (_symbol->getBinding() == llvm::ELF::STB_WEAK)
    return CanBeNull::canBeNullAtBuildtime;
  return CanBeNull::canBeNullNever;
}

template <class ELFT> uint64_t ELFDefinedAtom<ELFT>::size() const {
  // Common symbols are not allocated in object files,
  // so use st_size to tell how many bytes are required.
  if (_symbol && (_symbol->getType() == llvm::ELF::STT_COMMON ||
                  _symbol->st_shndx == llvm::ELF::SHN_COMMON))
    return (uint64_t)_symbol->st_size;

  return _contentData.size();
}

template <class ELFT> AbsoluteAtom::Scope ELFDefinedAtom<ELFT>::scope() const {
  if (!_symbol)
    return scopeGlobal;
  if (_symbol->getVisibility() == llvm::ELF::STV_HIDDEN)
    return scopeLinkageUnit;
  if (_symbol->getBinding() != llvm::ELF::STB_LOCAL)
    return scopeGlobal;
  return scopeTranslationUnit;
}

template <class ELFT> DefinedAtom::Merge ELFDefinedAtom<ELFT>::merge() const {
  if (!_symbol)
    return mergeNo;
  if (_symbol->getBinding() == llvm::ELF::STB_WEAK)
    return mergeAsWeak;
  if (_symbol->getType() == llvm::ELF::STT_COMMON ||
      _symbol->st_shndx == llvm::ELF::SHN_COMMON)
    return mergeAsTentative;
  return mergeNo;
}

template <class ELFT>
DefinedAtom::ContentType ELFDefinedAtom<ELFT>::doContentType() const {
  using namespace llvm::ELF;

  if (_section->sh_type == SHT_GROUP)
    return typeGroupComdat;
  if (!_symbol && _sectionName.startswith(".gnu.linkonce"))
    return typeGnuLinkOnce;

  uint64_t flags = _section->sh_flags;
  if (!(flags & SHF_ALLOC))
    return _contentType = typeNoAlloc;
  if (_section->sh_flags == (SHF_ALLOC | SHF_WRITE | SHF_TLS))
    return _section->sh_type == SHT_NOBITS ? typeThreadZeroFill
                                           : typeThreadData;

  if (_section->sh_flags == SHF_ALLOC && _section->sh_type == SHT_PROGBITS)
    return _contentType = typeConstant;
  if (_symbol->getType() == STT_GNU_IFUNC)
    return _contentType = typeResolver;
  if (_symbol->st_shndx == SHN_COMMON)
    return _contentType = typeZeroFill;

  if (_section->sh_type == SHT_ARM_EXIDX)
    return typeARMExidx;
  if (_section->sh_type == SHT_PROGBITS) {
    flags &= ~SHF_ALLOC;
    flags &= ~SHF_GROUP;
    if ((flags & SHF_STRINGS) || (flags & SHF_MERGE))
      return typeConstant;
    if (flags == SHF_WRITE)
      return typeData;
    return typeCode;
  }
  if (_section->sh_type == SHT_NOTE) {
    flags &= ~SHF_ALLOC;
    return (flags == SHF_WRITE) ? typeRWNote : typeRONote;
  }
  if (_section->sh_type == SHT_NOBITS)
    return typeZeroFill;

  if (_section->sh_type == SHT_NULL)
    if (_symbol->getType() == STT_COMMON || _symbol->st_shndx == SHN_COMMON)
      return typeZeroFill;

  if (_section->sh_type == SHT_INIT_ARRAY ||
      _section->sh_type == SHT_FINI_ARRAY)
    return typeData;
  return typeUnknown;
}

template <class ELFT>
DefinedAtom::ContentType ELFDefinedAtom<ELFT>::contentType() const {
  if (_contentType != typeUnknown)
    return _contentType;
  _contentType = doContentType();
  return _contentType;
}

template <class ELFT>
DefinedAtom::Alignment ELFDefinedAtom<ELFT>::alignment() const {
  if (!_symbol)
    return 1;

  // Obtain proper value of st_value field.
  const auto symValue = getSymbolValue();

  // Unallocated common symbols specify their alignment constraints in
  // st_value.
  if ((_symbol->getType() == llvm::ELF::STT_COMMON) ||
      _symbol->st_shndx == llvm::ELF::SHN_COMMON) {
    return symValue;
  }
  if (_section->sh_addralign == 0) {
    // sh_addralign of 0 means no alignment
    return Alignment(1, symValue);
  }
  return Alignment(_section->sh_addralign, symValue % _section->sh_addralign);
}

// Do we have a choice for ELF?  All symbols live in explicit sections.
template <class ELFT>
DefinedAtom::SectionChoice ELFDefinedAtom<ELFT>::sectionChoice() const {
  switch (contentType()) {
  case typeCode:
  case typeData:
  case typeZeroFill:
  case typeThreadZeroFill:
  case typeThreadData:
  case typeConstant:
    if ((_sectionName == ".text") || (_sectionName == ".data") ||
        (_sectionName == ".bss") || (_sectionName == ".rodata") ||
        (_sectionName == ".tdata") || (_sectionName == ".tbss"))
      return sectionBasedOnContent;
  default:
    break;
  }
  return sectionCustomRequired;
}

template <class ELFT>
StringRef ELFDefinedAtom<ELFT>::customSectionName() const {
  if ((contentType() == typeZeroFill) ||
      (_symbol && _symbol->st_shndx == llvm::ELF::SHN_COMMON))
    return ".bss";
  return _sectionName;
}

template <class ELFT>
DefinedAtom::ContentPermissions ELFDefinedAtom<ELFT>::permissions() const {
  if (_permissions != permUnknown)
    return _permissions;

  uint64_t flags = _section->sh_flags;

  if (!(flags & llvm::ELF::SHF_ALLOC))
    return _permissions = perm___;

  switch (_section->sh_type) {
  // permRW_L is for sections modified by the runtime
  // loader.
  case llvm::ELF::SHT_REL:
  case llvm::ELF::SHT_RELA:
    return _permissions = permRW_L;

  case llvm::ELF::SHT_DYNAMIC:
  case llvm::ELF::SHT_PROGBITS:
  case llvm::ELF::SHT_NOTE:
    flags &= ~llvm::ELF::SHF_ALLOC;
    flags &= ~llvm::ELF::SHF_GROUP;
    switch (flags) {
    // Code
    case llvm::ELF::SHF_EXECINSTR:
      return _permissions = permR_X;
    case (llvm::ELF::SHF_WRITE | llvm::ELF::SHF_EXECINSTR):
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
  case llvm::ELF::SHT_FINI_ARRAY:
    return _permissions = permRW_;

  case llvm::ELF::SHT_ARM_EXIDX:
    return _permissions = permR__;

  default:
    return _permissions = perm___;
  }
}

template <class ELFT>
DefinedAtom::reference_iterator ELFDefinedAtom<ELFT>::begin() const {
  uintptr_t index = _referenceStartIndex;
  const void *it = reinterpret_cast<const void *>(index);
  return reference_iterator(*this, it);
}

template <class ELFT>
DefinedAtom::reference_iterator ELFDefinedAtom<ELFT>::end() const {
  uintptr_t index = _referenceEndIndex;
  const void *it = reinterpret_cast<const void *>(index);
  return reference_iterator(*this, it);
}

template <class ELFT>
const Reference *ELFDefinedAtom<ELFT>::derefIterator(const void *It) const {
  uintptr_t index = reinterpret_cast<uintptr_t>(It);
  assert(index >= _referenceStartIndex);
  assert(index < _referenceEndIndex);
  return ((_referenceList)[index]);
}

template <class ELFT>
void ELFDefinedAtom<ELFT>::incrementIterator(const void *&It) const {
  uintptr_t index = reinterpret_cast<uintptr_t>(It);
  ++index;
  It = reinterpret_cast<const void *>(index);
}

template <class ELFT>
void ELFDefinedAtom<ELFT>::addReference(ELFReference<ELFT> *reference) {
  _referenceList.push_back(reference);
  _referenceEndIndex = _referenceList.size();
}

template <class ELFT> AbsoluteAtom::Scope ELFDynamicAtom<ELFT>::scope() const {
  if (_symbol->getVisibility() == llvm::ELF::STV_HIDDEN)
    return scopeLinkageUnit;
  if (_symbol->getBinding() != llvm::ELF::STB_LOCAL)
    return scopeGlobal;
  return scopeTranslationUnit;
}

template <class ELFT>
SharedLibraryAtom::Type ELFDynamicAtom<ELFT>::type() const {
  switch (_symbol->getType()) {
  case llvm::ELF::STT_FUNC:
  case llvm::ELF::STT_GNU_IFUNC:
    return Type::Code;
  case llvm::ELF::STT_OBJECT:
    return Type::Data;
  default:
    return Type::Unknown;
  }
}

#define INSTANTIATE(klass)        \
  template class klass<ELF32LE>;  \
  template class klass<ELF32BE>;  \
  template class klass<ELF64LE>;  \
  template class klass<ELF64BE>

INSTANTIATE(ELFAbsoluteAtom);
INSTANTIATE(ELFDefinedAtom);
INSTANTIATE(ELFDynamicAtom);
INSTANTIATE(ELFUndefinedAtom);

} // end namespace elf
} // end namespace lld
