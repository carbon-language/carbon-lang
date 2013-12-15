//===- lib/ReaderWriter/ELF/Mips/MipsTargetHandler.h ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_TARGET_HANDLER_H
#define LLD_READER_WRITER_ELF_MIPS_TARGET_HANDLER_H

#include "DefaultTargetHandler.h"
#include "MipsLinkingContext.h"
#include "MipsRelocationHandler.h"
#include "MipsSectionChunks.h"
#include "TargetLayout.h"

namespace lld {
namespace elf {

/// \brief TargetLayout for Mips
template <class ELFType>
class MipsTargetLayout LLVM_FINAL : public TargetLayout<ELFType> {
public:
  MipsTargetLayout(const MipsLinkingContext &ctx)
      : TargetLayout<ELFType>(ctx),
        _gotSection(new (_alloc) MipsGOTSection<ELFType>(ctx)) {}

  const MipsGOTSection<ELFType> &getGOTSection() const { return *_gotSection; }

  virtual AtomSection<ELFType> *
  createSection(StringRef name, int32_t type,
                DefinedAtom::ContentPermissions permissions,
                Layout::SectionOrder order) {
    if (type == DefinedAtom::typeGOT)
      return _gotSection;
    return DefaultLayout<ELFType>::createSection(name, type, permissions,
                                                 order);
  }

private:
  llvm::BumpPtrAllocator _alloc;
  MipsGOTSection<ELFType> *_gotSection;
};

/// \brief TargetHandler for Mips
class MipsTargetHandler LLVM_FINAL
    : public DefaultTargetHandler<Mips32ElELFType> {
public:
  MipsTargetHandler(MipsLinkingContext &targetInfo);

  uint64_t getGPDispSymAddr() const;

  virtual MipsTargetLayout<Mips32ElELFType> &targetLayout();
  virtual const MipsTargetRelocationHandler &getRelocationHandler() const;
  virtual LLD_UNIQUE_BUMP_PTR(DynamicTable<Mips32ElELFType>)
  createDynamicTable();
  virtual LLD_UNIQUE_BUMP_PTR(DynamicSymbolTable<Mips32ElELFType>)
  createDynamicSymbolTable();
  virtual bool createImplicitFiles(std::vector<std::unique_ptr<File>> &result);
  virtual void finalizeSymbolValues();

private:
  llvm::BumpPtrAllocator _alloc;
  MipsTargetLayout<Mips32ElELFType> _targetLayout;
  MipsTargetRelocationHandler _relocationHandler;
  AtomLayout *_gotSymAtom;
  AtomLayout *_gpDispSymAtom;
};

} // end namespace elf
} // end namespace lld

#endif
