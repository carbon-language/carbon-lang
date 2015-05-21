//===- lib/ReaderWriter/ELF/ARM/ARMELFWriters.h ---------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_ARM_ARM_ELF_WRITERS_H
#define LLD_READER_WRITER_ELF_ARM_ARM_ELF_WRITERS_H

#include "ARMLinkingContext.h"
#include "ARMSymbolTable.h"
#include "llvm/Support/ELF.h"

namespace lld {
namespace elf {

template <class WriterT> class ARMELFWriter : public WriterT {
public:
  ARMELFWriter(ARMLinkingContext &ctx, TargetLayout<ELF32LE> &layout);

  void finalizeDefaultAtomValues() override;

  /// \brief Create symbol table.
  unique_bump_ptr<SymbolTable<ELF32LE>> createSymbolTable() override;

  // Setup the ELF header.
  std::error_code setELFHeader() override;

protected:
  static const char *gotSymbol;
  static const char *dynamicSymbol;

private:
  ARMLinkingContext &_ctx;
  TargetLayout<ELF32LE> &_armLayout;
};

template <class WriterT>
const char *ARMELFWriter<WriterT>::gotSymbol = "_GLOBAL_OFFSET_TABLE_";
template <class WriterT>
const char *ARMELFWriter<WriterT>::dynamicSymbol = "_DYNAMIC";

template <class WriterT>
ARMELFWriter<WriterT>::ARMELFWriter(ARMLinkingContext &ctx,
                                    TargetLayout<ELF32LE> &layout)
    : WriterT(ctx, layout), _ctx(ctx), _armLayout(layout) {}

template <class WriterT>
void ARMELFWriter<WriterT>::finalizeDefaultAtomValues() {
  // Finalize the atom values that are part of the parent.
  WriterT::finalizeDefaultAtomValues();

  if (auto *gotAtom = _armLayout.findAbsoluteAtom(gotSymbol)) {
    if (auto gotpltSection = _armLayout.findOutputSection(".got.plt"))
      gotAtom->_virtualAddr = gotpltSection->virtualAddr();
    else if (auto gotSection = _armLayout.findOutputSection(".got"))
      gotAtom->_virtualAddr = gotSection->virtualAddr();
    else
      gotAtom->_virtualAddr = 0;
  }

  if (auto *dynamicAtom = _armLayout.findAbsoluteAtom(dynamicSymbol)) {
    if (auto dynamicSection = _armLayout.findOutputSection(".dynamic"))
      dynamicAtom->_virtualAddr = dynamicSection->virtualAddr();
    else
      dynamicAtom->_virtualAddr = 0;
  }

  // Set required by gcc libc __ehdr_start symbol with pointer to ELF header
  if (auto ehdr = _armLayout.findAbsoluteAtom("__ehdr_start"))
    ehdr->_virtualAddr = this->_elfHeader->virtualAddr();

  // Set required by gcc libc symbols __exidx_start/__exidx_end
  this->updateScopeAtomValues("exidx", ".ARM.exidx");
}

template <class WriterT>
unique_bump_ptr<SymbolTable<ELF32LE>>
ARMELFWriter<WriterT>::createSymbolTable() {
  return unique_bump_ptr<SymbolTable<ELF32LE>>(new (this->_alloc)
                                                   ARMSymbolTable(_ctx));
}

template <class WriterT> std::error_code ARMELFWriter<WriterT>::setELFHeader() {
  if (std::error_code ec = WriterT::setELFHeader())
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

#endif // LLD_READER_WRITER_ELF_ARM_ARM_ELF_WRITERS_H
