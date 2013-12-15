//===- lld/ReaderWriter/ELF/Mips/MipsRelocationHandler.h ------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_RELOCATION_HANDLER_H
#define LLD_READER_WRITER_ELF_MIPS_RELOCATION_HANDLER_H

#include "MipsLinkingContext.h"

namespace lld {
namespace elf {

class MipsTargetHandler;

class MipsTargetRelocationHandler LLVM_FINAL
    : public TargetRelocationHandler<Mips32ElELFType> {
public:
  MipsTargetRelocationHandler(const MipsLinkingContext &context,
                              const MipsTargetHandler &handler);

  virtual error_code applyRelocation(ELFWriter &, llvm::FileOutputBuffer &,
                                     const lld::AtomLayout &,
                                     const Reference &) const;

private:
  const MipsLinkingContext &_context;
  const MipsTargetHandler &_targetHandler;

  typedef std::vector<const Reference *> PairedRelocationsT;
  typedef std::unordered_map<const lld::AtomLayout *, PairedRelocationsT>
  PairedRelocationMapT;

  mutable PairedRelocationMapT _pairedRelocations;

  void savePairedRelocation(const lld::AtomLayout &atom,
                            const Reference &ref) const;
  void applyPairedRelocations(ELFWriter &writer, llvm::FileOutputBuffer &buf,
                              const lld::AtomLayout &atom,
                              int64_t loAddend) const;
};

} // elf
} // lld

#endif
