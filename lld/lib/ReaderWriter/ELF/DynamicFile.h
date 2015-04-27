//===- lib/ReaderWriter/ELF/DynamicFile.h ---------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_DYNAMIC_FILE_H
#define LLD_READER_WRITER_ELF_DYNAMIC_FILE_H

#include "Atoms.h"
#include "lld/Core/SharedLibraryFile.h"
#include <unordered_map>

namespace lld {
class ELFLinkingContext;

namespace elf {

template <class ELFT> class DynamicFile : public SharedLibraryFile {
public:
  DynamicFile(std::unique_ptr<MemoryBuffer> mb, ELFLinkingContext &ctx);

  static std::error_code isCompatible(MemoryBufferRef mb,
                                      ELFLinkingContext &ctx);

  const SharedLibraryAtom *exports(StringRef name,
                                   bool dataSymbolOnly) const override;

  StringRef getDSOName() const override;

  static bool canParse(file_magic magic);

protected:
  std::error_code doParse() override;

private:
  mutable llvm::BumpPtrAllocator _alloc;
  std::unique_ptr<llvm::object::ELFFile<ELFT>> _objFile;
  /// \brief DT_SONAME
  StringRef _soname;

  struct SymAtomPair {
    const typename llvm::object::ELFFile<ELFT>::Elf_Sym *_symbol = nullptr;
    const SharedLibraryAtom *_atom = nullptr;
  };

  std::unique_ptr<MemoryBuffer> _mb;
  ELFLinkingContext &_ctx;
  bool _useShlibUndefines;
  mutable std::unordered_map<StringRef, SymAtomPair> _nameToSym;
};

} // end namespace elf
} // end namespace lld

#endif
