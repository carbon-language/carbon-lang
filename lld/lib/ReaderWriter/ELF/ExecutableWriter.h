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

//===----------------------------------------------------------------------===//
//  ExecutableWriter Class
//===----------------------------------------------------------------------===//
template<class ELFT>
class ExecutableWriter : public OutputELFWriter<ELFT> {
public:
  ExecutableWriter(ELFLinkingContext &ctx, TargetLayout<ELFT> &layout)
      : OutputELFWriter<ELFT>(ctx, layout),
        _runtimeFile(new RuntimeFile<ELFT>(ctx, "C runtime")) {}

protected:
  void buildDynamicSymbolTable(const File &file) override;
  void addDefaultAtoms() override;
  bool createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;
  void finalizeDefaultAtomValues() override;
  void createDefaultSections() override;

  bool isNeededTagRequired(const SharedLibraryAtom *sla) const override {
    return this->_layout.isCopied(sla);
  }

  unique_bump_ptr<InterpSection<ELFT>> _interpSection;
  std::unique_ptr<RuntimeFile<ELFT> > _runtimeFile;
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
            !this->_ctx.isDynamicallyExportedSymbol(da->name()) &&
            !(this->_ctx.shouldExportDynamic() &&
              da->scope() == Atom::Scope::scopeGlobal))
          continue;
        this->_dynamicSymbolTable->addSymbol(atom->_atom, section->ordinal(),
                                             atom->_virtualAddr, atom);
      }

  // Put weak symbols in the dynamic symbol table.
  if (this->_ctx.isDynamic()) {
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
  OutputELFWriter<ELFT>::addDefaultAtoms();
  _runtimeFile->addUndefinedAtom(this->_ctx.entrySymbolName());
  _runtimeFile->addAbsoluteAtom("__bss_start");
  _runtimeFile->addAbsoluteAtom("__bss_end");
  _runtimeFile->addAbsoluteAtom("_end");
  _runtimeFile->addAbsoluteAtom("end");
  _runtimeFile->addAbsoluteAtom("__preinit_array_start");
  _runtimeFile->addAbsoluteAtom("__preinit_array_end");
  _runtimeFile->addAbsoluteAtom("__init_array_start");
  _runtimeFile->addAbsoluteAtom("__init_array_end");
  if (this->_ctx.isRelaOutputFormat()) {
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
  if (this->_ctx.isDynamic()) {
    _interpSection.reset(new (this->_alloc) InterpSection<ELFT>(
        this->_ctx, ".interp", TargetLayout<ELFT>::ORDER_INTERP,
        this->_ctx.getInterpreter()));
    this->_layout.addSection(_interpSection.get());
  }
}

/// Finalize the value of all the absolute symbols that we
/// created
template <class ELFT> void ExecutableWriter<ELFT>::finalizeDefaultAtomValues() {
  OutputELFWriter<ELFT>::finalizeDefaultAtomValues();
  AtomLayout *bssStartAtom = this->_layout.findAbsoluteAtom("__bss_start");
  AtomLayout *bssEndAtom = this->_layout.findAbsoluteAtom("__bss_end");
  AtomLayout *underScoreEndAtom = this->_layout.findAbsoluteAtom("_end");
  AtomLayout *endAtom = this->_layout.findAbsoluteAtom("end");

  assert((bssStartAtom || bssEndAtom || underScoreEndAtom || endAtom) &&
         "Unable to find the absolute atoms that have been added by lld");

  auto startEnd = [&](StringRef sym, StringRef sec) -> void {
    std::string start = ("__" + sym + "_start").str();
    std::string end = ("__" + sym + "_end").str();
    AtomLayout *s = this->_layout.findAbsoluteAtom(start);
    AtomLayout *e = this->_layout.findAbsoluteAtom(end);
    OutputSection<ELFT> *section = this->_layout.findOutputSection(sec);
    if (section) {
      s->_virtualAddr = section->virtualAddr();
      e->_virtualAddr = section->virtualAddr() + section->memSize();
    } else {
      s->_virtualAddr = 0;
      e->_virtualAddr = 0;
    }
  };

  startEnd("preinit_array", ".preinit_array");
  startEnd("init_array", ".init_array");
  if (this->_ctx.isRelaOutputFormat())
    startEnd("rela_iplt", ".rela.plt");
  else
    startEnd("rel_iplt", ".rel.plt");
  startEnd("fini_array", ".fini_array");

  auto bssSection = this->_layout.findOutputSection(".bss");

  // If we don't find a bss section, then don't set these values
  if (bssSection) {
    bssStartAtom->_virtualAddr = bssSection->virtualAddr();
    bssEndAtom->_virtualAddr =
        bssSection->virtualAddr() + bssSection->memSize();
    underScoreEndAtom->_virtualAddr = bssEndAtom->_virtualAddr;
    endAtom->_virtualAddr = bssEndAtom->_virtualAddr;
  } else if (auto dataSection = this->_layout.findOutputSection(".data")) {
    underScoreEndAtom->_virtualAddr =
        dataSection->virtualAddr() + dataSection->memSize();
    endAtom->_virtualAddr = underScoreEndAtom->_virtualAddr;
  }
}

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_ELF_EXECUTABLE_WRITER_H
