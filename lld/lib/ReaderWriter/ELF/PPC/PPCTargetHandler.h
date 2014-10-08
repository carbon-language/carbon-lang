//===- lib/ReaderWriter/ELF/PPC/PPCTargetHandler.h ------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_PPC_PPC_TARGET_HANDLER_H
#define LLD_READER_WRITER_ELF_PPC_PPC_TARGET_HANDLER_H

#include "DefaultTargetHandler.h"
#include "TargetLayout.h"

namespace lld {
namespace elf {
typedef llvm::object::ELFType<llvm::support::big, 2, false> PPCELFType;
class PPCLinkingContext;

template <class ELFT> class PPCTargetLayout : public TargetLayout<ELFT> {
public:
  PPCTargetLayout(PPCLinkingContext &context) : TargetLayout<ELFT>(context) {}
};

class PPCTargetRelocationHandler final
    : public TargetRelocationHandler<PPCELFType> {
public:
  PPCTargetRelocationHandler(PPCLinkingContext &context,
                             PPCTargetLayout<PPCELFType> &layout)
      : _ppcContext(context), _ppcTargetLayout(layout) {}

  virtual std::error_code applyRelocation(ELFWriter &, llvm::FileOutputBuffer &,
                                          const lld::AtomLayout &,
                                          const Reference &) const override;

protected:
  PPCLinkingContext &_ppcContext;
  PPCTargetLayout<PPCELFType> &_ppcTargetLayout;
};

class PPCTargetHandler final
    : public DefaultTargetHandler<PPCELFType> {
public:
  PPCTargetHandler(PPCLinkingContext &context);

  PPCTargetLayout<PPCELFType> &getTargetLayout() override {
    return *(_ppcTargetLayout.get());
  }

  void registerRelocationNames(Registry &registry) override;

  const PPCTargetRelocationHandler &getRelocationHandler() const override {
    return *(_ppcRelocationHandler.get());
  }

  std::unique_ptr<Writer> getWriter() override;

private:
  static const Registry::KindStrings kindStrings[];
  PPCLinkingContext &_ppcLinkingContext;
  std::unique_ptr<PPCTargetLayout<PPCELFType>> _ppcTargetLayout;
  std::unique_ptr<PPCTargetRelocationHandler> _ppcRelocationHandler;
};
} // end namespace elf
} // end namespace lld

#endif
