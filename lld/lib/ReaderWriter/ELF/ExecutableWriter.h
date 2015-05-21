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
      : OutputELFWriter<ELFT>(ctx, layout) {}

protected:
  void buildDynamicSymbolTable(const File &file) override;
  void createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;
  void finalizeDefaultAtomValues() override;
  void createDefaultSections() override;

  bool isNeededTagRequired(const SharedLibraryAtom *sla) const override {
    return this->_layout.isCopied(sla);
  }

  unique_bump_ptr<InterpSection<ELFT>> _interpSection;

private:
  std::unique_ptr<RuntimeFile<ELFT>> createRuntimeFile();
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

template<class ELFT>
std::unique_ptr<RuntimeFile<ELFT>> ExecutableWriter<ELFT>::createRuntimeFile() {
  auto file = llvm::make_unique<RuntimeFile<ELFT>>(this->_ctx, "C runtime");
  file->addUndefinedAtom(this->_ctx.entrySymbolName());
  file->addAbsoluteAtom("__bss_start");
  file->addAbsoluteAtom("__bss_end");
  file->addAbsoluteAtom("_end");
  file->addAbsoluteAtom("end");
  file->addAbsoluteAtom("__preinit_array_start", true);
  file->addAbsoluteAtom("__preinit_array_end", true);
  file->addAbsoluteAtom("__init_array_start", true);
  file->addAbsoluteAtom("__init_array_end", true);
  if (this->_ctx.isRelaOutputFormat()) {
    file->addAbsoluteAtom("__rela_iplt_start");
    file->addAbsoluteAtom("__rela_iplt_end");
  } else {
    file->addAbsoluteAtom("__rel_iplt_start");
    file->addAbsoluteAtom("__rel_iplt_end");
  }
  file->addAbsoluteAtom("__fini_array_start", true);
  file->addAbsoluteAtom("__fini_array_end", true);
  return file;
}

/// \brief Hook in lld to add CRuntime file
template <class ELFT>
void ExecutableWriter<ELFT>::createImplicitFiles(
    std::vector<std::unique_ptr<File> > &result) {
  OutputELFWriter<ELFT>::createImplicitFiles(result);
  result.push_back(createRuntimeFile());
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

  this->updateScopeAtomValues("preinit_array", ".preinit_array");
  this->updateScopeAtomValues("init_array", ".init_array");
  if (this->_ctx.isRelaOutputFormat())
    this->updateScopeAtomValues("rela_iplt", ".rela.plt");
  else
    this->updateScopeAtomValues("rel_iplt", ".rel.plt");
  this->updateScopeAtomValues("fini_array", ".fini_array");

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
