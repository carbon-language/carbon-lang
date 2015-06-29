//===- lib/ReaderWriter/ELF/DynamicFile.cpp -------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DynamicFile.h"
#include "FileCommon.h"
#include "lld/ReaderWriter/ELFLinkingContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Path.h"

namespace lld {
namespace elf {

template <class ELFT>
DynamicFile<ELFT>::DynamicFile(std::unique_ptr<MemoryBuffer> mb,
                               ELFLinkingContext &ctx)
    : SharedLibraryFile(mb->getBufferIdentifier()), _mb(std::move(mb)),
      _ctx(ctx), _useShlibUndefines(ctx.useShlibUndefines()) {}

template <typename ELFT>
std::error_code DynamicFile<ELFT>::isCompatible(MemoryBufferRef mb,
                                                ELFLinkingContext &ctx) {
  return elf::isCompatible<ELFT>(mb, ctx);
}

template <class ELFT>
const SharedLibraryAtom *DynamicFile<ELFT>::exports(StringRef name,
                                                    bool dataSymbolOnly) const {
  assert(!dataSymbolOnly && "Invalid option for ELF exports!");
  // See if we have the symbol.
  auto sym = _nameToSym.find(name);
  if (sym == _nameToSym.end())
    return nullptr;
  // Have we already created a SharedLibraryAtom for it?
  if (sym->second._atom)
    return sym->second._atom;
  // Create a SharedLibraryAtom for this symbol.
  return sym->second._atom = new (_alloc)
             ELFDynamicAtom<ELFT>(*this, name, _soname, sym->second._symbol);
}

template <class ELFT> StringRef DynamicFile<ELFT>::getDSOName() const {
  return _soname;
}

template <class ELFT> bool DynamicFile<ELFT>::canParse(file_magic magic) {
  return magic == file_magic::elf_shared_object;
}

template <class ELFT> std::error_code DynamicFile<ELFT>::doParse() {
  std::error_code ec;
  _objFile.reset(new llvm::object::ELFFile<ELFT>(_mb->getBuffer(), ec));
  if (ec)
    return ec;

  llvm::object::ELFFile<ELFT> &obj = *_objFile;

  _soname = obj.getLoadName();
  if (_soname.empty())
    _soname = llvm::sys::path::filename(path());

  // Create a map from names to dynamic symbol table entries.
  // TODO: This should use the object file's build in hash table instead if
  // it exists.
  for (auto i = obj.begin_dynamic_symbols(), e = obj.end_dynamic_symbols();
       i != e; ++i) {
    auto name = obj.getSymbolName(i, true);
    if ((ec = name.getError()))
      return ec;

    // Dont add local symbols to dynamic entries. The first symbol in the
    // dynamic symbol table is a local symbol.
    if (i->getBinding() == llvm::ELF::STB_LOCAL)
      continue;

    // TODO: Add absolute symbols
    if (i->st_shndx == llvm::ELF::SHN_ABS)
      continue;

    if (i->st_shndx == llvm::ELF::SHN_UNDEF) {
      if (!_useShlibUndefines)
        continue;
      // Create an undefined atom.
      if (!name->empty()) {
        auto *newAtom = new (_alloc) ELFUndefinedAtom<ELFT>(*this, *name, &*i);
        _undefinedAtoms.push_back(newAtom);
      }
      continue;
    }
    _nameToSym[*name]._symbol = &*i;
  }
  return std::error_code();
}

template class DynamicFile<ELF32LE>;
template class DynamicFile<ELF32BE>;
template class DynamicFile<ELF64LE>;
template class DynamicFile<ELF64BE>;

} // end namespace elf
} // end namespace lld
