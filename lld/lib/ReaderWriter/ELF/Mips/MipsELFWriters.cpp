//===- lib/ReaderWriter/ELF/Mips/MipsELFWriters.cpp -----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsDynamicTable.h"
#include "MipsELFWriters.h"
#include "MipsLinkingContext.h"
#include "MipsTargetHandler.h"
#include "MipsTargetLayout.h"

namespace {
class MipsDynamicAtom : public lld::elf::DynamicAtom {
public:
  MipsDynamicAtom(const lld::File &f) : DynamicAtom(f) {}

  ContentPermissions permissions() const override { return permR__; }
};
}

namespace lld {
namespace elf {

template <class ELFT>
MipsELFWriter<ELFT>::MipsELFWriter(MipsLinkingContext &ctx,
                                   MipsTargetLayout<ELFT> &targetLayout,
                                   const MipsAbiInfoHandler<ELFT> &abiInfo)
    : _ctx(ctx), _targetLayout(targetLayout), _abiInfo(abiInfo) {}

template <class ELFT>
void MipsELFWriter<ELFT>::setELFHeader(ELFHeader<ELFT> &elfHeader) {
  elfHeader.e_version(1);
  elfHeader.e_ident(llvm::ELF::EI_VERSION, llvm::ELF::EV_CURRENT);
  elfHeader.e_ident(llvm::ELF::EI_OSABI, llvm::ELF::ELFOSABI_NONE);

  unsigned char abiVer = 0;
  if (_ctx.getOutputELFType() == ET_EXEC && _abiInfo.isCPicOnly())
    abiVer = 1;
  if (_abiInfo.isFp64())
    abiVer = 3;

  elfHeader.e_ident(llvm::ELF::EI_ABIVERSION, abiVer);
  elfHeader.e_flags(_abiInfo.getFlags());
}

template <class ELFT>
void MipsELFWriter<ELFT>::finalizeMipsRuntimeAtomValues() {
  auto gotSection = _targetLayout.findOutputSection(".got");
  auto got = gotSection ? gotSection->virtualAddr() : 0;
  auto gp = gotSection ? got + _targetLayout.getGPOffset() : 0;

  setAtomValue("_gp", gp);
  setAtomValue("_gp_disp", gp);
  setAtomValue("__gnu_local_gp", gp);
}

template <class ELFT>
std::unique_ptr<RuntimeFile<ELFT>> MipsELFWriter<ELFT>::createRuntimeFile() {
  auto file = llvm::make_unique<RuntimeFile<ELFT>>(_ctx, "Mips runtime file");
  file->addAbsoluteAtom("_gp");
  file->addAbsoluteAtom("_gp_disp");
  file->addAbsoluteAtom("__gnu_local_gp");
  if (_ctx.isDynamic())
    file->addAtom(*new (file->allocator()) MipsDynamicAtom(*file));
  return file;
}

template <class ELFT>
unique_bump_ptr<Section<ELFT>>
MipsELFWriter<ELFT>::createOptionsSection(llvm::BumpPtrAllocator &alloc) {
  typedef unique_bump_ptr<Section<ELFT>> Ptr;
  const auto &regMask = _abiInfo.getRegistersMask();
  if (!regMask.hasValue())
    return Ptr();
  return ELFT::Is64Bits
             ? Ptr(new (alloc)
                       MipsOptionsSection<ELFT>(_ctx, _targetLayout, *regMask))
             : Ptr(new (alloc)
                       MipsReginfoSection<ELFT>(_ctx, _targetLayout, *regMask));
}

template <class ELFT>
unique_bump_ptr<Section<ELFT>>
MipsELFWriter<ELFT>::createAbiFlagsSection(llvm::BumpPtrAllocator &alloc) {
  typedef unique_bump_ptr<Section<ELFT>> Ptr;
  const auto &abi = _abiInfo.getAbiFlags();
  if (!abi.hasValue())
    return Ptr();
  return Ptr(new (alloc) MipsAbiFlagsSection<ELFT>(_ctx, _targetLayout, *abi));
}

template <class ELFT>
void MipsELFWriter<ELFT>::setAtomValue(StringRef name, uint64_t value) {
  AtomLayout *atom = _targetLayout.findAbsoluteAtom(name);
  assert(atom);
  atom->_virtualAddr = value;
}

template <class ELFT>
MipsDynamicLibraryWriter<ELFT>::MipsDynamicLibraryWriter(
    MipsLinkingContext &ctx, MipsTargetLayout<ELFT> &layout,
    const MipsAbiInfoHandler<ELFT> &abiInfo)
    : DynamicLibraryWriter<ELFT>(ctx, layout),
      _writeHelper(ctx, layout, abiInfo), _targetLayout(layout) {}

template <class ELFT>
void MipsDynamicLibraryWriter<ELFT>::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  DynamicLibraryWriter<ELFT>::createImplicitFiles(result);
  result.push_back(_writeHelper.createRuntimeFile());
}

template <class ELFT>
void MipsDynamicLibraryWriter<ELFT>::finalizeDefaultAtomValues() {
  DynamicLibraryWriter<ELFT>::finalizeDefaultAtomValues();
  _writeHelper.finalizeMipsRuntimeAtomValues();
}

template <class ELFT>
void MipsDynamicLibraryWriter<ELFT>::createDefaultSections() {
  DynamicLibraryWriter<ELFT>::createDefaultSections();
  _reginfo = _writeHelper.createOptionsSection(this->_alloc);
  if (_reginfo)
    this->_layout.addSection(_reginfo.get());
  _abiFlags = _writeHelper.createAbiFlagsSection(this->_alloc);
  if (_abiFlags)
    this->_layout.addSection(_abiFlags.get());
}

template <class ELFT>
std::error_code MipsDynamicLibraryWriter<ELFT>::setELFHeader() {
  DynamicLibraryWriter<ELFT>::setELFHeader();
  _writeHelper.setELFHeader(*this->_elfHeader);
  return std::error_code();
}

template <class ELFT>
unique_bump_ptr<SymbolTable<ELFT>>
MipsDynamicLibraryWriter<ELFT>::createSymbolTable() {
  return unique_bump_ptr<SymbolTable<ELFT>>(
      new (this->_alloc) MipsSymbolTable<ELFT>(this->_ctx));
}

template <class ELFT>
unique_bump_ptr<DynamicTable<ELFT>>
MipsDynamicLibraryWriter<ELFT>::createDynamicTable() {
  return unique_bump_ptr<DynamicTable<ELFT>>(
      new (this->_alloc) MipsDynamicTable<ELFT>(this->_ctx, _targetLayout));
}

template <class ELFT>
unique_bump_ptr<DynamicSymbolTable<ELFT>>
MipsDynamicLibraryWriter<ELFT>::createDynamicSymbolTable() {
  return unique_bump_ptr<DynamicSymbolTable<ELFT>>(new (
      this->_alloc) MipsDynamicSymbolTable<ELFT>(this->_ctx, _targetLayout));
}

template class MipsDynamicLibraryWriter<ELF32LE>;
template class MipsDynamicLibraryWriter<ELF64LE>;

template <class ELFT>
MipsExecutableWriter<ELFT>::MipsExecutableWriter(
    MipsLinkingContext &ctx, MipsTargetLayout<ELFT> &layout,
    const MipsAbiInfoHandler<ELFT> &abiInfo)
    : ExecutableWriter<ELFT>(ctx, layout), _writeHelper(ctx, layout, abiInfo),
      _targetLayout(layout) {}

template <class ELFT>
std::error_code MipsExecutableWriter<ELFT>::setELFHeader() {
  std::error_code ec = ExecutableWriter<ELFT>::setELFHeader();
  if (ec)
    return ec;

  StringRef entryName = this->_ctx.entrySymbolName();
  if (const AtomLayout *al = this->_layout.findAtomLayoutByName(entryName)) {
    const auto *ea = cast<DefinedAtom>(al->_atom);
    if (ea->codeModel() == DefinedAtom::codeMipsMicro ||
        ea->codeModel() == DefinedAtom::codeMipsMicroPIC)
      // Adjust entry symbol value if this symbol is microMIPS encoded.
      this->_elfHeader->e_entry(al->_virtualAddr | 1);
  }

  _writeHelper.setELFHeader(*this->_elfHeader);
  return std::error_code();
}

template <class ELFT>
void MipsExecutableWriter<ELFT>::buildDynamicSymbolTable(const File &file) {
  // MIPS ABI requires to add to dynsym even undefined symbols
  // if they have a corresponding entries in a global part of GOT.
  for (auto sec : this->_layout.sections())
    if (auto section = dyn_cast<AtomSection<ELFT>>(sec))
      for (const auto &atom : section->atoms()) {
        if (_targetLayout.getGOTSection().hasGlobalGOTEntry(atom->_atom)) {
          this->_dynamicSymbolTable->addSymbol(atom->_atom, section->ordinal(),
                                               atom->_virtualAddr, atom);
          continue;
        }

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

  for (const UndefinedAtom *a : file.undefined())
    // FIXME (simon): Consider to move this check to the
    // MipsELFUndefinedAtom class method. That allows to
    // handle more complex coditions in the future.
    if (_targetLayout.getGOTSection().hasGlobalGOTEntry(a))
      this->_dynamicSymbolTable->addSymbol(a, ELF::SHN_UNDEF);

  // Skip our immediate parent class method
  // ExecutableWriter<ELFT>::buildDynamicSymbolTable because we replaced it
  // with our own version. Call OutputELFWriter directly.
  OutputELFWriter<ELFT>::buildDynamicSymbolTable(file);
}

template <class ELFT>
void MipsExecutableWriter<ELFT>::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  ExecutableWriter<ELFT>::createImplicitFiles(result);
  result.push_back(_writeHelper.createRuntimeFile());
}

template <class ELFT>
void MipsExecutableWriter<ELFT>::finalizeDefaultAtomValues() {
  // Finalize the atom values that are part of the parent.
  ExecutableWriter<ELFT>::finalizeDefaultAtomValues();
  _writeHelper.finalizeMipsRuntimeAtomValues();
}

template <class ELFT> void MipsExecutableWriter<ELFT>::createDefaultSections() {
  ExecutableWriter<ELFT>::createDefaultSections();
  _reginfo = _writeHelper.createOptionsSection(this->_alloc);
  if (_reginfo)
    this->_layout.addSection(_reginfo.get());
  _abiFlags = _writeHelper.createAbiFlagsSection(this->_alloc);
  if (_abiFlags)
    this->_layout.addSection(_abiFlags.get());
}

template <class ELFT>
unique_bump_ptr<SymbolTable<ELFT>>
MipsExecutableWriter<ELFT>::createSymbolTable() {
  return unique_bump_ptr<SymbolTable<ELFT>>(
      new (this->_alloc) MipsSymbolTable<ELFT>(this->_ctx));
}

/// \brief create dynamic table
template <class ELFT>
unique_bump_ptr<DynamicTable<ELFT>>
MipsExecutableWriter<ELFT>::createDynamicTable() {
  return unique_bump_ptr<DynamicTable<ELFT>>(
      new (this->_alloc) MipsDynamicTable<ELFT>(this->_ctx, _targetLayout));
}

/// \brief create dynamic symbol table
template <class ELFT>
unique_bump_ptr<DynamicSymbolTable<ELFT>>
MipsExecutableWriter<ELFT>::createDynamicSymbolTable() {
  return unique_bump_ptr<DynamicSymbolTable<ELFT>>(new (
      this->_alloc) MipsDynamicSymbolTable<ELFT>(this->_ctx, _targetLayout));
}

template class MipsExecutableWriter<ELF32LE>;
template class MipsExecutableWriter<ELF64LE>;

} // elf
} // lld
