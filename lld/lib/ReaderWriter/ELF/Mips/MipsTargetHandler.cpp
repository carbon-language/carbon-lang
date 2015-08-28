//===- lib/ReaderWriter/ELF/Mips/MipsTargetHandler32EL.cpp ----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ELFReader.h"
#include "MipsELFFile.h"
#include "MipsELFWriters.h"
#include "MipsTargetHandler.h"

namespace lld {
namespace elf {

template <class ELFT>
MipsTargetHandler<ELFT>::MipsTargetHandler(MipsLinkingContext &ctx)
    : _ctx(ctx), _targetLayout(new MipsTargetLayout<ELFT>(ctx, _abiInfoHandler)),
      _relocationHandler(
          createMipsRelocationHandler<ELFT>(ctx, *_targetLayout)) {}

template <class ELFT>
std::unique_ptr<Reader> MipsTargetHandler<ELFT>::getObjReader() {
  return llvm::make_unique<ELFReader<MipsELFFile<ELFT>>>(_ctx);
}

template <class ELFT>
std::unique_ptr<Reader> MipsTargetHandler<ELFT>::getDSOReader() {
  return llvm::make_unique<ELFReader<DynamicFile<ELFT>>>(_ctx);
}

template <class ELFT>
const TargetRelocationHandler &
MipsTargetHandler<ELFT>::getRelocationHandler() const {
  return *_relocationHandler;
}

template <class ELFT>
std::unique_ptr<Writer> MipsTargetHandler<ELFT>::getWriter() {
  switch (_ctx.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
    return llvm::make_unique<MipsExecutableWriter<ELFT>>(_ctx, *_targetLayout,
                                                         _abiInfoHandler);
  case llvm::ELF::ET_DYN:
    return llvm::make_unique<MipsDynamicLibraryWriter<ELFT>>(
        _ctx, *_targetLayout, _abiInfoHandler);
  case llvm::ELF::ET_REL:
    llvm_unreachable("TODO: support -r mode");
  default:
    llvm_unreachable("unsupported output type");
  }
}

template <class ELFT> MipsAbi MipsTargetHandler<ELFT>::getAbi() const {
  return _abiInfoHandler.getAbi();
}

template class MipsTargetHandler<ELF32BE>;
template class MipsTargetHandler<ELF32LE>;
template class MipsTargetHandler<ELF64BE>;
template class MipsTargetHandler<ELF64LE>;

template <class ELFT>
MipsSymbolTable<ELFT>::MipsSymbolTable(const ELFLinkingContext &ctx)
    : SymbolTable<ELFT>(ctx, ".symtab",
                        TargetLayout<ELFT>::ORDER_SYMBOL_TABLE) {}

template <class ELFT>
void MipsSymbolTable<ELFT>::addDefinedAtom(Elf_Sym &sym, const DefinedAtom *da,
                                           int64_t addr) {
  SymbolTable<ELFT>::addDefinedAtom(sym, da, addr);

  switch (da->codeModel()) {
  case DefinedAtom::codeMipsMicro:
    sym.st_other |= llvm::ELF::STO_MIPS_MICROMIPS;
    break;
  case DefinedAtom::codeMipsMicroPIC:
    sym.st_other |= llvm::ELF::STO_MIPS_MICROMIPS | llvm::ELF::STO_MIPS_PIC;
    break;
  default:
    break;
  }
}

template <class ELFT> void MipsSymbolTable<ELFT>::finalize(bool sort) {
  SymbolTable<ELFT>::finalize(sort);

  for (auto &ste : this->_symbolTable) {
    if (!ste._atom)
      continue;
    if (const auto *da = dyn_cast<DefinedAtom>(ste._atom)) {
      if (da->codeModel() == DefinedAtom::codeMipsMicro ||
          da->codeModel() == DefinedAtom::codeMipsMicroPIC) {
        // Adjust dynamic microMIPS symbol value. That allows a dynamic
        // linker to recognize and handle this symbol correctly.
        ste._symbol.st_value = ste._symbol.st_value | 1;
      }
    }
  }
}

template class MipsSymbolTable<ELF32BE>;
template class MipsSymbolTable<ELF32LE>;
template class MipsSymbolTable<ELF64BE>;
template class MipsSymbolTable<ELF64LE>;

template <class ELFT>
MipsDynamicSymbolTable<ELFT>::MipsDynamicSymbolTable(
    const ELFLinkingContext &ctx, MipsTargetLayout<ELFT> &layout)
    : DynamicSymbolTable<ELFT>(ctx, layout, ".dynsym",
                               TargetLayout<ELFT>::ORDER_DYNAMIC_SYMBOLS),
      _targetLayout(layout) {}

template <class ELFT> void MipsDynamicSymbolTable<ELFT>::sortSymbols() {
  typedef typename DynamicSymbolTable<ELFT>::SymbolEntry SymbolEntry;
  std::stable_sort(this->_symbolTable.begin(), this->_symbolTable.end(),
                   [this](const SymbolEntry &A, const SymbolEntry &B) {
                     if (A._symbol.getBinding() != STB_GLOBAL &&
                         B._symbol.getBinding() != STB_GLOBAL)
                       return A._symbol.getBinding() < B._symbol.getBinding();

                     return _targetLayout.getGOTSection().compare(A._atom,
                                                                  B._atom);
                   });
}

template <class ELFT> void MipsDynamicSymbolTable<ELFT>::finalize() {
  DynamicSymbolTable<ELFT>::finalize();

  const auto &pltSection = _targetLayout.getPLTSection();

  for (auto &ste : this->_symbolTable) {
    const Atom *a = ste._atom;
    if (!a)
      continue;
    if (auto *layout = pltSection.findPLTLayout(a)) {
      a = layout->_atom;
      // Under some conditions a dynamic symbol table record should hold
      // a symbol value of the corresponding PLT entry. For details look
      // at the PLT entry creation code in the class MipsRelocationPass.
      // Let's update atomLayout fields for such symbols.
      assert(!ste._atomLayout);
      ste._symbol.st_value = layout->_virtualAddr;
      ste._symbol.st_other |= ELF::STO_MIPS_PLT;
    }

    if (const auto *da = dyn_cast<DefinedAtom>(a)) {
      if (da->codeModel() == DefinedAtom::codeMipsMicro ||
          da->codeModel() == DefinedAtom::codeMipsMicroPIC) {
        // Adjust dynamic microMIPS symbol value. That allows a dynamic
        // linker to recognize and handle this symbol correctly.
        ste._symbol.st_value = ste._symbol.st_value | 1;
      }
    }
  }
}

template class MipsDynamicSymbolTable<ELF32BE>;
template class MipsDynamicSymbolTable<ELF32LE>;
template class MipsDynamicSymbolTable<ELF64BE>;
template class MipsDynamicSymbolTable<ELF64LE>;

}
}
