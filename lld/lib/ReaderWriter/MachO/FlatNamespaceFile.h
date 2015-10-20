//===- lib/ReaderWriter/MachO/FlatNamespaceFile.h -------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_MACHO_FLAT_NAMESPACE_FILE_H
#define LLD_READER_WRITER_MACHO_FLAT_NAMESPACE_FILE_H

#include "lld/Core/SharedLibraryFile.h"
#include "llvm/Support/Debug.h"

namespace lld {
namespace mach_o {

//
// A FlateNamespaceFile instance may be added as a resolution source of last
// resort, depending on how -flat_namespace and -undefined are set.
//
class FlatNamespaceFile : public SharedLibraryFile {
public:
  FlatNamespaceFile(const MachOLinkingContext &context)
    : SharedLibraryFile("flat namespace") { }

  const SharedLibraryAtom *exports(StringRef name,
                                   bool dataSymbolOnly) const override {
    _sharedLibraryAtoms.push_back(
      new (allocator()) MachOSharedLibraryAtom(*this, name, getDSOName(),
                                               false));

    return _sharedLibraryAtoms.back();
  }

  StringRef getDSOName() const override { return "flat-namespace"; }

  const AtomVector<DefinedAtom> &defined() const override {
    return _noDefinedAtoms;
  }
  const AtomVector<UndefinedAtom> &undefined() const override {
    return _noUndefinedAtoms;
  }

  const AtomVector<SharedLibraryAtom> &sharedLibrary() const override {
    return _sharedLibraryAtoms;
  }

  const AtomVector<AbsoluteAtom> &absolute() const override {
    return _noAbsoluteAtoms;
  }

private:
  mutable AtomVector<SharedLibraryAtom> _sharedLibraryAtoms;
};

} // namespace mach_o
} // namespace lld

#endif // LLD_READER_WRITER_MACHO_FLAT_NAMESPACE_FILE_H
