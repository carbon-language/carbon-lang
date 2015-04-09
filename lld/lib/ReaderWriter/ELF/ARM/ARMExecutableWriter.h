//===--------- lib/ReaderWriter/ELF/ARM/ARMExecutableWriter.h -------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_ARM_ARM_EXECUTABLE_WRITER_H
#define LLD_READER_WRITER_ELF_ARM_ARM_EXECUTABLE_WRITER_H

#include "ExecutableWriter.h"
#include "ARMLinkingContext.h"
#include "ARMTargetHandler.h"
#include "ARMSymbolTable.h"
#include "llvm/Support/ELF.h"

namespace {
const char *gotSymbol = "_GLOBAL_OFFSET_TABLE_";
}

namespace lld {
namespace elf {

template <class ELFT>
class ARMExecutableWriter : public ExecutableWriter<ELFT> {
public:
  ARMExecutableWriter(ARMLinkingContext &ctx, ARMTargetLayout<ELFT> &layout);

protected:
  // Add any runtime files and their atoms to the output
  void createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;

  void finalizeDefaultAtomValues() override;

  /// \brief Create symbol table.
  unique_bump_ptr<SymbolTable<ELFT>> createSymbolTable() override;

  void processUndefinedSymbol(StringRef symName,
                              RuntimeFile<ELFT> &file) const override;

  // Setup the ELF header.
  std::error_code setELFHeader() override;

private:
  ARMLinkingContext &_ctx;
  ARMTargetLayout<ELFT> &_armLayout;
};

template <class ELFT>
ARMExecutableWriter<ELFT>::ARMExecutableWriter(ARMLinkingContext &ctx,
                                               ARMTargetLayout<ELFT> &layout)
    : ExecutableWriter<ELFT>(ctx, layout), _ctx(ctx), _armLayout(layout) {}

template <class ELFT>
void ARMExecutableWriter<ELFT>::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  ExecutableWriter<ELFT>::createImplicitFiles(result);
}

template <class ELFT>
void ARMExecutableWriter<ELFT>::finalizeDefaultAtomValues() {
  // Finalize the atom values that are part of the parent.
  ExecutableWriter<ELFT>::finalizeDefaultAtomValues();
  AtomLayout *gotAtom = _armLayout.findAbsoluteAtom(gotSymbol);
  if (gotAtom) {
    if (auto gotpltSection = _armLayout.findOutputSection(".got.plt"))
      gotAtom->_virtualAddr = gotpltSection->virtualAddr();
    else if (auto gotSection = _armLayout.findOutputSection(".got"))
      gotAtom->_virtualAddr = gotSection->virtualAddr();
    else
      gotAtom->_virtualAddr = 0;
  }
  // TODO: resolve addresses of __exidx_start/_end atoms
}

template <class ELFT>
unique_bump_ptr<SymbolTable<ELFT>>
ARMExecutableWriter<ELFT>::createSymbolTable() {
  return unique_bump_ptr<SymbolTable<ELFT>>(
      new (this->_alloc) ARMSymbolTable<ELFT>(this->_ctx));
}

template <class ELFT>
void ARMExecutableWriter<ELFT>::processUndefinedSymbol(
    StringRef symName, RuntimeFile<ELFT> &file) const {
  if (symName == gotSymbol) {
    file.addAbsoluteAtom(gotSymbol);
  } else if (symName.startswith("__exidx")) {
    file.addAbsoluteAtom("__exidx_start");
    file.addAbsoluteAtom("__exidx_end");
  }
}

template <class ELFT>
std::error_code ARMExecutableWriter<ELFT>::setELFHeader() {
  if (std::error_code ec = ExecutableWriter<ELFT>::setELFHeader())
    return ec;

  // Set ARM-specific flags.
  this->_elfHeader->e_flags(llvm::ELF::EF_ARM_EABI_VER5 |
                            llvm::ELF::EF_ARM_VFP_FLOAT);

  StringRef entryName = _ctx.entrySymbolName();
  if (const AtomLayout *al = _armLayout.findAtomLayoutByName(entryName)) {
    if (const auto *ea = dyn_cast<DefinedAtom>(al->_atom)) {
      switch (ea->codeModel()) {
      case DefinedAtom::codeNA:
        if (al->_virtualAddr & 0x3) {
          llvm::report_fatal_error(
              "Two least bits must be zero for ARM entry point");
        }
        break;
      case DefinedAtom::codeARMThumb:
        // Fixup entry point for Thumb code.
        this->_elfHeader->e_entry(al->_virtualAddr | 0x1);
        break;
      default:
        llvm_unreachable("Wrong code model of entry point atom");
      }
    }
  }

  return std::error_code();
}

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_ELF_ARM_ARM_EXECUTABLE_WRITER_H
