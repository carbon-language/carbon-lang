//===- lib/ReaderWriter/ELF/X86/X86DynamicLibraryWriter.h -----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef X86_X86_DYNAMIC_LIBRARY_WRITER_H
#define X86_X86_DYNAMIC_LIBRARY_WRITER_H

#include "DynamicLibraryWriter.h"
#include "X86LinkingContext.h"

namespace lld {
namespace elf {

template <class ELFT>
class X86DynamicLibraryWriter : public DynamicLibraryWriter<ELFT> {
public:
  X86DynamicLibraryWriter(X86LinkingContext &context,
                          X86TargetLayout<ELFT> &layout);

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
  X86LinkingContext &_context;
  X86TargetLayout<ELFT> &_x86Layout;
};

template <class ELFT>
X86DynamicLibraryWriter<ELFT>::X86DynamicLibraryWriter(
    X86LinkingContext &context, X86TargetLayout<ELFT> &layout)
    : DynamicLibraryWriter<ELFT>(context, layout),
      _gotFile(new GOTFile(context)), _context(context), _x86Layout(layout) {}

template <class ELFT>
bool X86DynamicLibraryWriter<ELFT>::createImplicitFiles(
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
