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
#include <mutex>

namespace lld {

File::~File() {}

File::atom_collection_vector<DefinedAtom>       File::_noDefinedAtoms;
File::atom_collection_vector<UndefinedAtom>     File::_noUndefinedAtoms;
File::atom_collection_vector<SharedLibraryAtom> File::_noSharedLibraryAtoms;
File::atom_collection_vector<AbsoluteAtom>      File::_noAbsoluteAtoms;

std::error_code File::parse() {
  std::lock_guard<std::mutex> lock(_parseMutex);
  if (!_lastError.hasValue())
    _lastError = doParse();
  return _lastError.getValue();
}

} // namespace lld
