//===- Core/File.cpp - A Container of Atoms -------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/File.h"
#include "lld/Core/LLVM.h"

namespace lld {

File::~File() {}

StringRef File::translationUnitSource() const {
  return StringRef();
}


File::atom_collection_empty<DefinedAtom>       File::_noDefinedAtoms;
File::atom_collection_empty<UndefinedAtom>     File::_noUndefinedAtoms;
File::atom_collection_empty<SharedLibraryAtom> File::_noSharedLibraryAtoms;
File::atom_collection_empty<AbsoluteAtom>      File::_noAbsoluteAtoms;




} // namespace lld
