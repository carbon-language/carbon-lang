//===- lib/ReaderWriter/ELF/X86/X86TargetHandler.h ------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_X86_TARGET_HANDLER_H
#define LLD_READER_WRITER_ELF_X86_TARGET_HANDLER_H

#include "DefaultTargetHandler.h"
#include "TargetLayout.h"

#include "lld/ReaderWriter/Reader.h"

namespace lld {
namespace elf {

typedef llvm::object::ELFType<llvm::support::little, 2, false> X86ELFType;
class X86LinkingContext;

template <class ELFT> class X86TargetLayout : public TargetLayout<ELFT> {
public:
  X86TargetLayout(X86LinkingContext &context) : TargetLayout<ELFT>(context) {}
};

class X86TargetRelocationHandler final
    : public TargetRelocationHandler<X86ELFType> {
public:
  X86TargetRelocationHandler(X86LinkingContext &context,
                             X86TargetLayout<X86ELFType> &layout)
      : _x86Context(context), _x86TargetLayout(layout) {}

  error_code applyRelocation(ELFWriter &, llvm::FileOutputBuffer &,
                             const lld::AtomLayout &,
                             const Reference &) const override;

  static const Registry::KindStrings kindStrings[];

protected:
  X86LinkingContext &_x86Context;
  X86TargetLayout<X86ELFType> &_x86TargetLayout;
};

class X86TargetHandler final
    : public DefaultTargetHandler<X86ELFType> {
public:
  X86TargetHandler(X86LinkingContext &context);

  X86TargetLayout<X86ELFType> &getTargetLayout() override {
    return *(_x86TargetLayout.get());
  }

  void registerRelocationNames(Registry &registry) override;

  const X86TargetRelocationHandler &getRelocationHandler() const override {
    return *(_x86RelocationHandler.get());
  }

  std::unique_ptr<Writer> getWriter() override;

protected:
  static const Registry::KindStrings kindStrings[];
  X86LinkingContext &_x86LinkingContext;
  std::unique_ptr<X86TargetLayout<X86ELFType>> _x86TargetLayout;
  std::unique_ptr<X86TargetRelocationHandler> _x86RelocationHandler;
};
} // end namespace elf
} // end namespace lld

#endif
