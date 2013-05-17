//===- lib/ReaderWriter/ELF/X86_64/X86_64TargetHandler.h ------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_X86_64_TARGET_HANDLER_H
#define LLD_READER_WRITER_ELF_X86_64_TARGET_HANDLER_H

#include "DefaultTargetHandler.h"
#include "X86_64RelocationHandler.h"
#include "TargetLayout.h"

#include "lld/ReaderWriter/Simple.h"

namespace lld {
namespace elf {
typedef llvm::object::ELFType<llvm::support::little, 8, true> X86_64ELFType;
class X86_64TargetInfo;

class X86_64TargetHandler LLVM_FINAL
    : public DefaultTargetHandler<X86_64ELFType> {
public:
  X86_64TargetHandler(X86_64TargetInfo &targetInfo);

  virtual TargetLayout<X86_64ELFType> &targetLayout() {
    return _targetLayout;
  }

  virtual const X86_64TargetRelocationHandler &getRelocationHandler() const {
    return _relocationHandler;
  }

  virtual void addFiles(InputFiles &f);

private:
  class GOTFile : public SimpleFile {
  public:
    GOTFile(const ELFTargetInfo &eti) : SimpleFile(eti, "GOTFile") {}
    llvm::BumpPtrAllocator _alloc;
  } _gotFile;

  X86_64TargetRelocationHandler _relocationHandler;
  TargetLayout<X86_64ELFType> _targetLayout;
};
} // end namespace elf
} // end namespace lld

#endif
