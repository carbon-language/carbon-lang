//===- lib/ReaderWriter/ELF/ExecutableWriter.h ----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_EXECUTABLE_WRITER_H
#define LLD_READER_WRITER_ELF_EXECUTABLE_WRITER_H

#include "OutputELFWriter.h"

namespace lld {
namespace elf {
using namespace llvm;
using namespace llvm::object;

template<class ELFT>
class ExecutableWriter;

//===----------------------------------------------------------------------===//
//  ExecutableWriter Class
//===----------------------------------------------------------------------===//
template<class ELFT>
class ExecutableWriter : public OutputELFWriter<ELFT> {
public:
  ExecutableWriter(const ELFTargetInfo &ti)
    : OutputELFWriter<ELFT>(ti), _runtimeFile(ti)
  {}

private:
  virtual void addDefaultAtoms();
  virtual void addFiles(InputFiles&);
  virtual void finalizeDefaultAtomValues();

  CRuntimeFile<ELFT> _runtimeFile;
};

//===----------------------------------------------------------------------===//
//  ExecutableWriter
//===----------------------------------------------------------------------===//

/// \brief Add absolute symbols by default. These are linker added
/// absolute symbols
template<class ELFT>
void ExecutableWriter<ELFT>::addDefaultAtoms() {
  _runtimeFile.addUndefinedAtom(this->_targetInfo.getEntry());
  _runtimeFile.addAbsoluteAtom("__bss_start");
  _runtimeFile.addAbsoluteAtom("__bss_end");
  _runtimeFile.addAbsoluteAtom("_end");
  _runtimeFile.addAbsoluteAtom("end");
  _runtimeFile.addAbsoluteAtom("__preinit_array_start");
  _runtimeFile.addAbsoluteAtom("__preinit_array_end");
  _runtimeFile.addAbsoluteAtom("__init_array_start");
  _runtimeFile.addAbsoluteAtom("__init_array_end");
  _runtimeFile.addAbsoluteAtom("__rela_iplt_start");
  _runtimeFile.addAbsoluteAtom("__rela_iplt_end");
  _runtimeFile.addAbsoluteAtom("__fini_array_start");
  _runtimeFile.addAbsoluteAtom("__fini_array_end");
}

/// \brief Hook in lld to add CRuntime file
template <class ELFT>
void ExecutableWriter<ELFT>::addFiles(InputFiles &inputFiles) {
  addDefaultAtoms();
  inputFiles.prependFile(_runtimeFile);
  // Give a chance for the target to add atoms
  this->_targetHandler.addFiles(inputFiles);
}

/// Finalize the value of all the absolute symbols that we
/// created
template<class ELFT>
void ExecutableWriter<ELFT>::finalizeDefaultAtomValues() {
  auto bssStartAtomIter = this->_layout->findAbsoluteAtom("__bss_start");
  auto bssEndAtomIter = this->_layout->findAbsoluteAtom("__bss_end");
  auto underScoreEndAtomIter = this->_layout->findAbsoluteAtom("_end");
  auto endAtomIter = this->_layout->findAbsoluteAtom("end");

  auto startEnd = [&](StringRef sym, StringRef sec) -> void {
    // TODO: This looks like a good place to use Twine...
    std::string start("__"), end("__");
    start += sym;
    start += "_start";
    end += sym;
    end += "_end";
    auto s = this->_layout->findAbsoluteAtom(start);
    auto e = this->_layout->findAbsoluteAtom(end);
    auto section = this->_layout->findOutputSection(sec);
    if (section) {
      (*s)->_virtualAddr = section->virtualAddr();
      (*e)->_virtualAddr = section->virtualAddr() + section->memSize();
    } else {
      (*s)->_virtualAddr = 0;
      (*e)->_virtualAddr = 0;
    }
  };

  startEnd("preinit_array", ".preinit_array");
  startEnd("init_array", ".init_array");
  startEnd("rela_iplt", ".rela.plt");
  startEnd("fini_array", ".fini_array");

  assert(!(bssStartAtomIter == this->_layout->absoluteAtoms().end() ||
           bssEndAtomIter == this->_layout->absoluteAtoms().end() ||
           underScoreEndAtomIter == this->_layout->absoluteAtoms().end() ||
           endAtomIter == this->_layout->absoluteAtoms().end()) &&
         "Unable to find the absolute atoms that have been added by lld");

  auto phe = this->_programHeader
      ->findProgramHeader(llvm::ELF::PT_LOAD, llvm::ELF::PF_W, llvm::ELF::PF_X);

  assert(!(phe == this->_programHeader->rend()) &&
         "Can't find a data segment in the program header!");

  (*bssStartAtomIter)->_virtualAddr = (*phe)->p_vaddr + (*phe)->p_filesz;
  (*bssEndAtomIter)->_virtualAddr = (*phe)->p_vaddr + (*phe)->p_memsz;
  (*underScoreEndAtomIter)->_virtualAddr = (*phe)->p_vaddr + (*phe)->p_memsz;
  (*endAtomIter)->_virtualAddr = (*phe)->p_vaddr + (*phe)->p_memsz;

  // Give a chance for the target to finalize its atom values
  this->_targetHandler.finalizeSymbolValues();
}

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_ELF_EXECUTABLE_WRITER_H
