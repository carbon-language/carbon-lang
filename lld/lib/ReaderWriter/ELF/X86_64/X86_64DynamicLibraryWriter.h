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
#include "X86_64LinkingContext.h"

namespace lld {
namespace elf {

template <class ELFT>
class X86_64DynamicLibraryWriter : public DynamicLibraryWriter<ELFT> {
public:
  X86_64DynamicLibraryWriter(X86_64LinkingContext &context,
                             X86_64TargetLayout<ELFT> &layout);

protected:
  // Add any runtime files and their atoms to the output
  virtual bool createImplicitFiles(std::vector<std::unique_ptr<File>> &);

  virtual void finalizeDefaultAtomValues() {
    return DynamicLibraryWriter<ELFT>::finalizeDefaultAtomValues();
  }

  virtual void addDefaultAtoms() {
    return DynamicLibraryWriter<ELFT>::addDefaultAtoms();
  }

private:
  class GOTFile : public SimpleFile {
  public:
    GOTFile(const ELFLinkingContext &eti) : SimpleFile("GOTFile") {}
    llvm::BumpPtrAllocator _alloc;
  };

  std::unique_ptr<GOTFile> _gotFile;
  X86_64LinkingContext &_context;
  X86_64TargetLayout<ELFT> &_x86_64Layout;
};

template <class ELFT>
X86_64DynamicLibraryWriter<ELFT>::X86_64DynamicLibraryWriter(
    X86_64LinkingContext &context, X86_64TargetLayout<ELFT> &layout)
    : DynamicLibraryWriter<ELFT>(context, layout),
      _gotFile(new GOTFile(context)), _context(context), _x86_64Layout(layout) {
}

template <class ELFT>
bool X86_64DynamicLibraryWriter<ELFT>::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  DynamicLibraryWriter<ELFT>::createImplicitFiles(result);
  _gotFile->addAtom(*new (_gotFile->_alloc) GLOBAL_OFFSET_TABLEAtom(*_gotFile));
  _gotFile->addAtom(*new (_gotFile->_alloc) DYNAMICAtom(*_gotFile));
  result.push_back(std::move(_gotFile));
  return true;
}

} // namespace elf
} // namespace lld

#endif
