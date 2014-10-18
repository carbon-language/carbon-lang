//===- lib/ReaderWriter/ELF/X86_64/X86_64LinkingContext.h -----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_X86_64_X86_64_LINKING_CONTEXT_H
#define LLD_READER_WRITER_ELF_X86_64_X86_64_LINKING_CONTEXT_H

#include "X86_64TargetHandler.h"
#include "lld/ReaderWriter/ELFLinkingContext.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/ELF.h"

namespace lld {
namespace elf {

/// \brief x86-64 internal references.
enum {
  /// \brief The 32 bit index of the relocation in the got this reference refers
  /// to.
  LLD_R_X86_64_GOTRELINDEX = 1024,
};

class X86_64LinkingContext final : public ELFLinkingContext {
public:
  X86_64LinkingContext(llvm::Triple triple)
      : ELFLinkingContext(triple, std::unique_ptr<TargetHandlerBase>(
                                      new X86_64TargetHandler(*this))) {}

  void addPasses(PassManager &) override;

  uint64_t getBaseAddress() const override {
    if (_baseAddress == 0)
      return 0x400000;
    return _baseAddress;
  }

  bool isDynamicRelocation(const DefinedAtom &,
                           const Reference &r) const override {
    if (r.kindNamespace() != Reference::KindNamespace::ELF)
      return false;
    assert(r.kindArch() == Reference::KindArch::x86_64);
    switch (r.kindValue()) {
    case llvm::ELF::R_X86_64_RELATIVE:
    case llvm::ELF::R_X86_64_GLOB_DAT:
    case llvm::ELF::R_X86_64_COPY:
    case llvm::ELF::R_X86_64_DTPMOD64:
    case llvm::ELF::R_X86_64_DTPOFF64:
      return true;
    default:
      return false;
    }
  }

  bool isCopyRelocation(const Reference &r) const override {
    if (r.kindNamespace() != Reference::KindNamespace::ELF)
      return false;
    assert(r.kindArch() == Reference::KindArch::x86_64);
    if (r.kindValue() == llvm::ELF::R_X86_64_COPY)
      return true;
    return false;
  }

  virtual bool isPLTRelocation(const DefinedAtom &,
                               const Reference &r) const override {
    if (r.kindNamespace() != Reference::KindNamespace::ELF)
      return false;
    assert(r.kindArch() == Reference::KindArch::x86_64);
    switch (r.kindValue()) {
    case llvm::ELF::R_X86_64_JUMP_SLOT:
    case llvm::ELF::R_X86_64_IRELATIVE:
      return true;
    default:
      return false;
    }
  }

  /// \brief X86_64 has two relative relocations
  /// a) for supporting IFUNC - R_X86_64_IRELATIVE
  /// b) for supporting relative relocs - R_X86_64_RELATIVE
  bool isRelativeReloc(const Reference &r) const override {
    if (r.kindNamespace() != Reference::KindNamespace::ELF)
      return false;
    assert(r.kindArch() == Reference::KindArch::x86_64);
    switch (r.kindValue()) {
    case llvm::ELF::R_X86_64_IRELATIVE:
    case llvm::ELF::R_X86_64_RELATIVE:
      return true;
    default:
      return false;
    }
  }

  /// \brief Create Internal files for Init/Fini
  void createInternalFiles(std::vector<std::unique_ptr<File>> &) const override;
};
} // end namespace elf
} // end namespace lld

#endif
