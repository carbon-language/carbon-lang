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
  typedef llvm::object::ELFFile<ELFT> ELFO;
  typedef typename ELFO::Elf_Shdr Elf_Shdr;
  typedef typename ELFO::Elf_Dyn Elf_Dyn;

  std::error_code ec;
  _objFile.reset(new ELFO(_mb->getBuffer(), ec));
  if (ec)
    return ec;

  ELFO &obj = *_objFile;

  const char *base = _mb->getBuffer().data();
  const Elf_Dyn *dynStart = nullptr;
  const Elf_Dyn *dynEnd = nullptr;

  const Elf_Shdr *dynSymSec = nullptr;
  for (const Elf_Shdr &sec : obj.sections()) {
    switch (sec.sh_type) {
    case llvm::ELF::SHT_DYNAMIC: {
      dynStart = reinterpret_cast<const Elf_Dyn *>(base + sec.sh_offset);
      uint64_t size = sec.sh_size;
      if (size % sizeof(Elf_Dyn))
        return llvm::object::object_error::parse_failed;
      dynEnd = dynStart + size / sizeof(Elf_Dyn);
      break;
    }
    case llvm::ELF::SHT_DYNSYM:
      dynSymSec = &sec;
      break;
    }
  }

  ErrorOr<StringRef> strTableOrErr = obj.getStringTableForSymtab(*dynSymSec);
  if (std::error_code ec = strTableOrErr.getError())
    return ec;
  StringRef stringTable = *strTableOrErr;

  for (const Elf_Dyn &dyn : llvm::make_range(dynStart, dynEnd)) {
    if (dyn.d_tag == llvm::ELF::DT_SONAME) {
      uint64_t offset = dyn.getVal();
      if (offset >= stringTable.size())
        return llvm::object::object_error::parse_failed;
      _soname = StringRef(stringTable.data() + offset);
      break;
    }
  }

  if (_soname.empty())
    _soname = llvm::sys::path::filename(path());

  // Create a map from names to dynamic symbol table entries.
  // TODO: This should use the object file's build in hash table instead if
  // it exists.
  for (auto i = obj.symbol_begin(dynSymSec), e = obj.symbol_end(dynSymSec);
       i != e; ++i) {
    auto name = i->getName(stringTable);
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
