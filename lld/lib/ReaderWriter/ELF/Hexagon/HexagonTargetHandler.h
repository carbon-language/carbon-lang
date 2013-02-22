//===- lib/ReaderWriter/ELF/Hexagon/HexagonTargetHandler.h ----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_HEXAGON_TARGET_HANDLER_H
#define LLD_READER_WRITER_ELF_HEXAGON_TARGET_HANDLER_H

#include "DefaultTargetHandler.h"
#include "ExecutableAtoms.h"
#include "HexagonRelocationHandler.h"
#include "TargetLayout.h"

namespace lld {
namespace elf {
typedef llvm::object::ELFType<llvm::support::little, 4, false> HexagonELFType;
class HexagonTargetInfo;


/// \brief Handle Hexagon specific Atoms
template <class HexagonELFType>
class HexagonTargetAtomHandler LLVM_FINAL :
    public TargetAtomHandler<HexagonELFType> {
  typedef llvm::object::Elf_Sym_Impl<HexagonELFType> Elf_Sym;
  typedef llvm::object::Elf_Shdr_Impl<HexagonELFType> Elf_Shdr;
public:

  virtual DefinedAtom::ContentType
  contentType(const ELFDefinedAtom<HexagonELFType> *atom) const {
    return contentType(atom->section(), atom->symbol());
  }

  virtual DefinedAtom::ContentType
  contentType(const Elf_Shdr *section, const Elf_Sym *sym) const {
    switch (sym->st_shndx) {
    // Common symbols
    case llvm::ELF::SHN_HEXAGON_SCOMMON:
    case llvm::ELF::SHN_HEXAGON_SCOMMON_1:
    case llvm::ELF::SHN_HEXAGON_SCOMMON_2:
    case llvm::ELF::SHN_HEXAGON_SCOMMON_4:
    case llvm::ELF::SHN_HEXAGON_SCOMMON_8:
      return DefinedAtom::typeZeroFill;

    default:
      if (section->sh_flags & llvm::ELF::SHF_HEX_GPREL)
        return DefinedAtom::typeDataFast;
      else
        llvm_unreachable("unknown symbol type");
    }
  }

  virtual DefinedAtom::ContentPermissions
  contentPermissions(const ELFDefinedAtom<HexagonELFType> *atom) const {
    // All of the hexagon specific symbols belong in the data segment
    return DefinedAtom::permRW_;
  }

  virtual int64_t getType(const Elf_Sym *sym) const {
    switch (sym->st_shndx) {
    // Common symbols
    case llvm::ELF::SHN_HEXAGON_SCOMMON:
    case llvm::ELF::SHN_HEXAGON_SCOMMON_1:
    case llvm::ELF::SHN_HEXAGON_SCOMMON_2:
    case llvm::ELF::SHN_HEXAGON_SCOMMON_4:
    case llvm::ELF::SHN_HEXAGON_SCOMMON_8:
      return llvm::ELF::STT_COMMON;

    default:
      return sym->getType();
    }
  }
};

/// \brief TargetHandler for Hexagon 
class HexagonTargetHandler LLVM_FINAL :
    public DefaultTargetHandler<HexagonELFType> {
public:
  HexagonTargetHandler(HexagonTargetInfo &targetInfo);

  virtual TargetLayout<HexagonELFType> &targetLayout() {
    return _targetLayout;
  }

  virtual TargetAtomHandler<HexagonELFType> &targetAtomHandler() {
    return _targetAtomHandler;
  }

  virtual const HexagonTargetRelocationHandler &getRelocationHandler() const {
    return _relocationHandler;
  }

private:
  HexagonTargetRelocationHandler _relocationHandler;
  TargetLayout<HexagonELFType> _targetLayout;
  HexagonTargetAtomHandler<HexagonELFType> _targetAtomHandler;
};
} // end namespace elf
} // end namespace lld

#endif
