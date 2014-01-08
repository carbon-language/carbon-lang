//===- lib/ReaderWriter/ELF/Mips/MipsLinkingContext.h ---------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_LINKING_CONTEXT_H
#define LLD_READER_WRITER_ELF_MIPS_LINKING_CONTEXT_H

#include "lld/ReaderWriter/ELFLinkingContext.h"

namespace lld {
namespace elf {

/// \brief Mips internal references.
enum {
  /// \brief Do nothing but mark GOT entry as a global one.
  LLD_R_MIPS_GLOBAL_GOT = 1024,
  /// \brief The same as R_MIPS_GOT16 but for global symbols.
  LLD_R_MIPS_GLOBAL_GOT16 = 1025
};

typedef llvm::object::ELFType<llvm::support::little, 2, false> Mips32ElELFType;

template <class ELFType> class MipsTargetLayout;

class MipsLinkingContext LLVM_FINAL : public ELFLinkingContext {
public:
  MipsLinkingContext(llvm::Triple triple);

  MipsTargetLayout<Mips32ElELFType> &getTargetLayout();
  const MipsTargetLayout<Mips32ElELFType> &getTargetLayout() const;

  // ELFLinkingContext
  virtual bool isLittleEndian() const;
  virtual void addPasses(PassManager &pm);
};

} // elf
} // lld

#endif
