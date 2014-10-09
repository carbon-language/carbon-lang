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
  ExecutableWriter(const ELFLinkingContext &context, TargetLayout<ELFT> &layout)
      : OutputELFWriter<ELFT>(context, layout),
        _runtimeFile(new CRuntimeFile<ELFT>(context)) {}

protected:
  virtual void buildDynamicSymbolTable(const File &file);
  virtual void addDefaultAtoms();
  virtual bool createImplicitFiles(std::vector<std::unique_ptr<File> > &);
  virtual void finalizeDefaultAtomValues();
  virtual void createDefaultSections();

  virtual bool isNeededTagRequired(const SharedLibraryAtom *sla) const {
    return this->_layout.isCopied(sla);
  }

  LLD_UNIQUE_BUMP_PTR(InterpSection<ELFT>) _interpSection;
  std::unique_ptr<CRuntimeFile<ELFT> > _runtimeFile;
};

//===----------------------------------------------------------------------===//
//  ExecutableWriter
//===----------------------------------------------------------------------===//
template<class ELFT>
void ExecutableWriter<ELFT>::buildDynamicSymbolTable(const File &file) {
  for (auto sec : this->_layout.sections())
    if (auto section = dyn_cast<AtomSection<ELFT>>(sec))
      for (const auto &atom : section->atoms()) {
        const DefinedAtom *da = dyn_cast<const DefinedAtom>(atom->_atom);
        if (!da)
          continue;
        if (da->dynamicExport() != DefinedAtom::dynamicExportAlways &&
            !this->_context.isDynamicallyExportedSymbol(da->name()) &&
            !(this->_context.shouldExportDynamic() &&
              da->scope() == Atom::Scope::scopeGlobal))
          continue;
        this->_dynamicSymbolTable->addSymbol(atom->_atom, section->ordinal(),
                                             atom->_virtualAddr, atom);
      }

  // Put weak symbols in the dynamic symbol table.
  if (this->_context.isDynamic()) {
    for (const UndefinedAtom *a : file.undefined()) {
      if (this->_layout.isReferencedByDefinedAtom(a) &&
          a->canBeNull() != UndefinedAtom::canBeNullNever)
        this->_dynamicSymbolTable->addSymbol(a, ELF::SHN_UNDEF);
    }
  }

  OutputELFWriter<ELFT>::buildDynamicSymbolTable(file);
}

/// \brief Add absolute symbols by default. These are linker added
/// absolute symbols
template<class ELFT>
void ExecutableWriter<ELFT>::addDefaultAtoms() {
  _runtimeFile->addUndefinedAtom(this->_context.entrySymbolName());
  _runtimeFile->addAbsoluteAtom("__bss_start");
  _runtimeFile->addAbsoluteAtom("__bss_end");
  _runtimeFile->addAbsoluteAtom("_end");
  _runtimeFile->addAbsoluteAtom("end");
  _runtimeFile->addAbsoluteAtom("__preinit_array_start");
  _runtimeFile->addAbsoluteAtom("__preinit_array_end");
  _runtimeFile->addAbsoluteAtom("__init_array_start");
  _runtimeFile->addAbsoluteAtom("__init_array_end");
  if (this->_context.isRelaOutputFormat()) {
    _runtimeFile->addAbsoluteAtom("__rela_iplt_start");
    _runtimeFile->addAbsoluteAtom("__rela_iplt_end");
  } else {
    _runtimeFile->addAbsoluteAtom("__rel_iplt_start");
    _runtimeFile->addAbsoluteAtom("__rel_iplt_end");
  }
  _runtimeFile->addAbsoluteAtom("__fini_array_start");
  _runtimeFile->addAbsoluteAtom("__fini_array_end");
}

/// \brief Hook in lld to add CRuntime file
template <class ELFT>
bool ExecutableWriter<ELFT>::createImplicitFiles(
    std::vector<std::unique_ptr<File> > &result) {
  // Add the default atoms as defined by executables
  ExecutableWriter<ELFT>::addDefaultAtoms();
  OutputELFWriter<ELFT>::createImplicitFiles(result);
  result.push_back(std::move(_runtimeFile));
  return true;
}

template <class ELFT> void ExecutableWriter<ELFT>::createDefaultSections() {
  OutputELFWriter<ELFT>::createDefaultSections();
  if (this->_context.isDynamic()) {
    _interpSection.reset(new (this->_alloc) InterpSection<ELFT>(
        this->_context, ".interp", DefaultLayout<ELFT>::ORDER_INTERP,
        this->_context.getInterpreter()));
    this->_layout.addSection(_interpSection.get());
  }
}

/// Finalize the value of all the absolute symbols that we
/// created
template <class ELFT> void ExecutableWriter<ELFT>::finalizeDefaultAtomValues() {
  auto bssStartAtomIter = this->_layout.findAbsoluteAtom("__bss_start");
  auto bssEndAtomIter = this->_layout.findAbsoluteAtom("__bss_end");
  auto underScoreEndAtomIter = this->_layout.findAbsoluteAtom("_end");
  auto endAtomIter = this->_layout.findAbsoluteAtom("end");

  auto startEnd = [&](StringRef sym, StringRef sec) -> void {
    // TODO: This looks like a good place to use Twine...
    std::string start("__"), end("__");
    start += sym;
    start += "_start";
    end += sym;
    end += "_end";
    auto s = this->_layout.findAbsoluteAtom(start);
    auto e = this->_layout.findAbsoluteAtom(end);
    auto section = this->_layout.findOutputSection(sec);
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
  if (this->_context.isRelaOutputFormat())
    startEnd("rela_iplt", ".rela.plt");
  else
    startEnd("rel_iplt", ".rel.plt");
  startEnd("fini_array", ".fini_array");

  assert(!(bssStartAtomIter == this->_layout.absoluteAtoms().end() ||
           bssEndAtomIter == this->_layout.absoluteAtoms().end() ||
           underScoreEndAtomIter == this->_layout.absoluteAtoms().end() ||
           endAtomIter == this->_layout.absoluteAtoms().end()) &&
         "Unable to find the absolute atoms that have been added by lld");

  auto bssSection = this->_layout.findOutputSection(".bss");

  // If we don't find a bss section, then don't set these values
  if (bssSection) {
    (*bssStartAtomIter)->_virtualAddr = bssSection->virtualAddr();
    (*bssEndAtomIter)->_virtualAddr =
        bssSection->virtualAddr() + bssSection->memSize();
    (*underScoreEndAtomIter)->_virtualAddr = (*bssEndAtomIter)->_virtualAddr;
    (*endAtomIter)->_virtualAddr = (*bssEndAtomIter)->_virtualAddr;
  } else if (auto dataSection = this->_layout.findOutputSection(".data")) {
    (*underScoreEndAtomIter)->_virtualAddr =
        dataSection->virtualAddr() + dataSection->memSize();
    (*endAtomIter)->_virtualAddr = (*underScoreEndAtomIter)->_virtualAddr;
  }
}

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_ELF_EXECUTABLE_WRITER_H
