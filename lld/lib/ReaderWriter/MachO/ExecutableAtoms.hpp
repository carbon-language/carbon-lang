//===- lib/ReaderWriter/MachO/ExecutableAtoms.hpp -------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_MACHO_EXECUTABLE_ATOMS_H
#define LLD_READER_WRITER_MACHO_EXECUTABLE_ATOMS_H

#include "Atoms.h"

#include "llvm/Support/MachO.h"

#include "lld/Core/ArchiveLibraryFile.h"
#include "lld/Core/DefinedAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/LinkingContext.h"
#include "lld/Core/Reference.h"
#include "lld/Core/Simple.h"
#include "lld/Core/UndefinedAtom.h"
#include "lld/ReaderWriter/MachOLinkingContext.h"

namespace lld {
namespace mach_o {


//
// CEntryFile adds an UndefinedAtom for "_main" so that the Resolving
// phase will fail if "_main" is undefined.
//
class CEntryFile : public SimpleFile {
public:
  CEntryFile(const MachOLinkingContext &context)
      : SimpleFile("C entry"),
       _undefMain(*this, context.entrySymbolName()) {
    this->addAtom(_undefMain);
  }

private:
  SimpleUndefinedAtom   _undefMain;
};


//
// StubHelperFile adds an UndefinedAtom for "dyld_stub_binder" so that
// the Resolveing phase will fail if "dyld_stub_binder" is undefined.
//
class StubHelperFile : public SimpleFile {
public:
  StubHelperFile(const MachOLinkingContext &context)
      : SimpleFile("stub runtime"),
        _undefBinder(*this, context.binderSymbolName()) {
    this->addAtom(_undefBinder);
  }

private:
  SimpleUndefinedAtom   _undefBinder;
};


//
// MachHeaderAliasFile lazily instantiates the magic symbols that mark the start
// of the mach_header for final linked images.
//
class MachHeaderAliasFile : public ArchiveLibraryFile {
public:
  MachHeaderAliasFile(const MachOLinkingContext &context)
      : ArchiveLibraryFile("mach_header symbols") {
      switch (context.outputMachOType()) {
      case llvm::MachO::MH_EXECUTE:
        _machHeaderSymbolName = "__mh_execute_header";
        break;
      case llvm::MachO::MH_DYLIB:
        _machHeaderSymbolName = "__mh_dylib_header";
        break;
      case llvm::MachO::MH_BUNDLE:
        _machHeaderSymbolName = "__mh_bundle_header";
        break;
      case llvm::MachO::MH_DYLINKER:
        _machHeaderSymbolName = "__mh_dylinker_header";
        break;
      case llvm::MachO::MH_PRELOAD:
        _machHeaderSymbolName = "__mh_preload_header";
        break;
      default:
        llvm_unreachable("no mach_header symbol for file type");
      }
  }

  std::error_code
  parseAllMembers(std::vector<std::unique_ptr<File>> &result) override {
    return std::error_code();
  }

  const File *find(StringRef sym, bool dataSymbolOnly) const override {
    if (sym.equals("___dso_handle") || sym.equals(_machHeaderSymbolName)) {
      _definedAtoms._atoms.push_back(new (_alloc) MachODefinedAtom(
          *this, sym, DefinedAtom::scopeLinkageUnit,
          DefinedAtom::typeMachHeader, DefinedAtom::mergeNo, false, false,
          ArrayRef<uint8_t>(), DefinedAtom::Alignment(12,0)));
      return this;
    }
    return nullptr;
  }

  const atom_collection<DefinedAtom> &defined() const override {
    return _definedAtoms;
  }
  const atom_collection<UndefinedAtom> &undefined() const override {
    return _undefinedAtoms;
  }

  const atom_collection<SharedLibraryAtom> &sharedLibrary() const override {
    return _sharedLibraryAtoms;
  }

  const atom_collection<AbsoluteAtom> &absolute() const override {
    return _absoluteAtoms;
  }


private:
  mutable atom_collection_vector<DefinedAtom> _definedAtoms;
  atom_collection_vector<UndefinedAtom> _undefinedAtoms;
  atom_collection_vector<SharedLibraryAtom> _sharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom> _absoluteAtoms;
  StringRef _machHeaderSymbolName;
  mutable llvm::BumpPtrAllocator _alloc;
};

} // namespace mach_o
} // namespace lld

#endif // LLD_READER_WRITER_MACHO_EXECUTABLE_ATOMS_H
