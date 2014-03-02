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

  virtual error_code applyRelocation(ELFWriter &, llvm::FileOutputBuffer &,
                                     const lld::AtomLayout &,
                                     const Reference &) const;

protected:
  PPCLinkingContext &_ppcContext;
  PPCTargetLayout<PPCELFType> &_ppcTargetLayout;
};

class PPCTargetHandler final
    : public DefaultTargetHandler<PPCELFType> {
public:
  PPCTargetHandler(PPCLinkingContext &context);

  virtual PPCTargetLayout<PPCELFType> &getTargetLayout() {
    return *(_ppcTargetLayout.get());
  }

  virtual void registerRelocationNames(Registry &registry);

  virtual const PPCTargetRelocationHandler &getRelocationHandler() const {
    return *(_ppcRelocationHandler.get());
  }

  virtual std::unique_ptr<Writer> getWriter();

private:
  static const Registry::KindStrings kindStrings[];
  PPCLinkingContext &_ppcLinkingContext;
  std::unique_ptr<PPCTargetLayout<PPCELFType>> _ppcTargetLayout;
  std::unique_ptr<PPCTargetRelocationHandler> _ppcRelocationHandler;
};
} // end namespace elf
} // end namespace lld

#endif
