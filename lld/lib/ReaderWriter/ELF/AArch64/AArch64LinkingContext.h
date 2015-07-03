//===- lib/ReaderWriter/ELF/AArch64/AArch64LinkingContext.h ---------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_AARCH64_AARCH64_LINKING_CONTEXT_H
#define LLD_READER_WRITER_ELF_AARCH64_AARCH64_LINKING_CONTEXT_H

#include "lld/ReaderWriter/ELFLinkingContext.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/ELF.h"

namespace lld {
namespace elf {

enum {
  /// \brief The offset to add operation for a R_AARCH64_ADR_GOT_PAGE
  ADD_AARCH64_GOTRELINDEX = 0xE000,
};

class AArch64LinkingContext final : public ELFLinkingContext {
public:
  int getMachineType() const override { return llvm::ELF::EM_AARCH64; }
  AArch64LinkingContext(llvm::Triple);

  void addPasses(PassManager &) override;
  void registerRelocationNames(Registry &r) override;

  uint64_t getBaseAddress() const override {
    if (_baseAddress == 0)
      return 0x400000;
    return _baseAddress;
  }

  bool isDynamicRelocation(const Reference &r) const override {
    if (r.kindNamespace() != Reference::KindNamespace::ELF)
      return false;
    assert(r.kindArch() == Reference::KindArch::AArch64);
    switch (r.kindValue()) {
    case llvm::ELF::R_AARCH64_COPY:
    case llvm::ELF::R_AARCH64_GLOB_DAT:
    case llvm::ELF::R_AARCH64_RELATIVE:
    case llvm::ELF::R_AARCH64_TLS_DTPREL64:
    case llvm::ELF::R_AARCH64_TLS_DTPMOD64:
    case llvm::ELF::R_AARCH64_TLS_TPREL64:
    case llvm::ELF::R_AARCH64_TLSDESC:
    case llvm::ELF::R_AARCH64_IRELATIVE:
      return true;
    default:
      return false;
    }
  }

  bool isCopyRelocation(const Reference &r) const override {
    if (r.kindNamespace() != Reference::KindNamespace::ELF)
      return false;
    assert(r.kindArch() == Reference::KindArch::AArch64);
    if (r.kindValue() == llvm::ELF::R_AARCH64_COPY)
      return true;
    return false;
  }

  bool isPLTRelocation(const Reference &r) const override {
    if (r.kindNamespace() != Reference::KindNamespace::ELF)
      return false;
    assert(r.kindArch() == Reference::KindArch::AArch64);
    switch (r.kindValue()) {
    case llvm::ELF::R_AARCH64_JUMP_SLOT:
    case llvm::ELF::R_AARCH64_IRELATIVE:
      return true;
    default:
      return false;
    }
  }

  bool isRelativeReloc(const Reference &r) const override {
    if (r.kindNamespace() != Reference::KindNamespace::ELF)
      return false;
    assert(r.kindArch() == Reference::KindArch::AArch64);
    switch (r.kindValue()) {
    case llvm::ELF::R_AARCH64_IRELATIVE:
    case llvm::ELF::R_AARCH64_RELATIVE:
      return true;
    default:
      return false;
    }
  }

  /// \brief The path to the dynamic interpreter
  StringRef getDefaultInterpreter() const override {
    return "/lib/ld-linux-aarch64.so.1";
  }
};
} // end namespace elf
} // end namespace lld

#endif
