//===- lib/ReaderWriter/ELF/X86/X86_64DynamicLibraryWriter.h ---------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef X86_64_DYNAMIC_LIBRARY_WRITER_H
#define X86_64_DYNAMIC_LIBRARY_WRITER_H

#include "DynamicLibraryWriter.h"
#include "X86_64ElfType.h"
#include "X86_64LinkingContext.h"
#include "X86_64TargetHandler.h"

namespace lld {
namespace elf {

class X86_64DynamicLibraryWriter : public DynamicLibraryWriter<X86_64ELFType> {
public:
  X86_64DynamicLibraryWriter(X86_64LinkingContext &context,
                             X86_64TargetLayout &layout);

protected:
  // Add any runtime files and their atoms to the output
  virtual bool createImplicitFiles(std::vector<std::unique_ptr<File>> &);

  virtual void finalizeDefaultAtomValues() {
    return DynamicLibraryWriter::finalizeDefaultAtomValues();
  }

  virtual void addDefaultAtoms() {
    return DynamicLibraryWriter::addDefaultAtoms();
  }

private:
  class GOTFile : public SimpleFile {
  public:
    GOTFile(const ELFLinkingContext &eti) : SimpleFile("GOTFile") {}
    llvm::BumpPtrAllocator _alloc;
  };

  std::unique_ptr<GOTFile> _gotFile;
};

X86_64DynamicLibraryWriter::X86_64DynamicLibraryWriter(
    X86_64LinkingContext &context, X86_64TargetLayout &layout)
    : DynamicLibraryWriter(context, layout), _gotFile(new GOTFile(context)) {}

bool X86_64DynamicLibraryWriter::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  DynamicLibraryWriter::createImplicitFiles(result);
  _gotFile->addAtom(*new (_gotFile->_alloc) GLOBAL_OFFSET_TABLEAtom(*_gotFile));
  _gotFile->addAtom(*new (_gotFile->_alloc) DYNAMICAtom(*_gotFile));
  result.push_back(std::move(_gotFile));
  return true;
}

} // namespace elf
} // namespace lld

#endif
