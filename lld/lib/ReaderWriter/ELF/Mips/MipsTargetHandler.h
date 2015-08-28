//===- lib/ReaderWriter/ELF/Mips/MipsTargetHandler.h ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_MIPS_TARGET_HANDLER_H
#define LLD_READER_WRITER_ELF_MIPS_MIPS_TARGET_HANDLER_H

#include "MipsAbiInfoHandler.h"
#include "MipsLinkingContext.h"
#include "MipsTargetLayout.h"
#include "TargetHandler.h"

namespace lld {
namespace elf {

class MipsBaseTargetHandler : public TargetHandler {
public:
  virtual MipsAbi getAbi() const = 0;
};

/// \brief TargetHandler for Mips
template <class ELFT>
class MipsTargetHandler final : public MipsBaseTargetHandler {
public:
  MipsTargetHandler(MipsLinkingContext &ctx);

  MipsAbiInfoHandler<ELFT> &getAbiInfoHandler() { return _abiInfoHandler; }

  std::unique_ptr<Reader> getObjReader() override;
  std::unique_ptr<Reader> getDSOReader() override;
  const TargetRelocationHandler &getRelocationHandler() const override;
  std::unique_ptr<Writer> getWriter() override;
  MipsAbi getAbi() const override;

private:
  MipsLinkingContext &_ctx;
  MipsAbiInfoHandler<ELFT> _abiInfoHandler;
  std::unique_ptr<MipsTargetLayout<ELFT>> _targetLayout;
  std::unique_ptr<TargetRelocationHandler> _relocationHandler;
};

template <class ELFT> class MipsSymbolTable : public SymbolTable<ELFT> {
public:
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;

  MipsSymbolTable(const ELFLinkingContext &ctx);

  void addDefinedAtom(Elf_Sym &sym, const DefinedAtom *da,
                      int64_t addr) override;
  void finalize(bool sort) override;
};

template <class ELFT>
class MipsDynamicSymbolTable : public DynamicSymbolTable<ELFT> {
public:
  MipsDynamicSymbolTable(const ELFLinkingContext &ctx,
                         MipsTargetLayout<ELFT> &layout);

  void sortSymbols() override;
  void finalize() override;

private:
  MipsTargetLayout<ELFT> &_targetLayout;
};

} // end namespace elf
} // end namespace lld

#endif
