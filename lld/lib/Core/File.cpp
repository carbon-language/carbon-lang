//===- Core/File.cpp - A Contaier of Atoms --------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/File.h"

namespace lld {

File::~File() {}

bool File::translationUnitSource(llvm::StringRef &path) const {
  return false;
}


}
