//===- Core/File.cpp - A Contaier of Atoms --------------------------------===//
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

const Atom *File::entryPoint() const {
  return nullptr;
}

}
