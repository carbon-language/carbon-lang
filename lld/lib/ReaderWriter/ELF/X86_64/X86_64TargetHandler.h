//===- lib/ReaderWriter/ELF/X86_64/X86_64TargetHandler.h ------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_X86_64_X86_64_TARGET_HANDLER_H
#define LLD_READER_WRITER_ELF_X86_64_X86_64_TARGET_HANDLER_H

#include "DefaultTargetHandler.h"
#include "ELFFile.h"
#include "X86_64RelocationHandler.h"
#include "TargetLayout.h"

#include "lld/ReaderWriter/Simple.h"

namespace lld {
namespace elf {
typedef llvm::object::ELFType<llvm::support::little, 2, true> X86_64ELFType;
class X86_64LinkingContext;

template <class ELFT> class X86_64TargetLayout : public TargetLayout<ELFT> {
public:
  X86_64TargetLayout(X86_64LinkingContext &context)
      : TargetLayout<ELFT>(context) {}
};

class X86_64TargetHandler final
    : public DefaultTargetHandler<X86_64ELFType> {
public:
  X86_64TargetHandler(X86_64LinkingContext &context);

  virtual X86_64TargetLayout<X86_64ELFType> &getTargetLayout() {
    return *(_x86_64TargetLayout.get());
  }

  virtual void registerRelocationNames(Registry &registry);

  virtual const X86_64TargetRelocationHandler &getRelocationHandler() const {
    return *(_x86_64RelocationHandler.get());
  }

  virtual std::unique_ptr<Writer> getWriter();

private:
  static const Registry::KindStrings kindStrings[];
  X86_64LinkingContext &_context;
  std::unique_ptr<X86_64TargetLayout<X86_64ELFType>> _x86_64TargetLayout;
  std::unique_ptr<X86_64TargetRelocationHandler> _x86_64RelocationHandler;
};

} // end namespace elf
} // end namespace lld

#endif
