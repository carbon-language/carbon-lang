//===- lib/Core/TargetInfo.cpp - Linker Target Info Interface -------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/TargetInfo.h"

#include "lld/Core/LinkerOptions.h"

#include "llvm/ADT/Triple.h"

namespace lld {
TargetInfo::~TargetInfo() {}

llvm::Triple TargetInfo::getTriple() const {
  return llvm::Triple(llvm::Triple::normalize(_options._target));
}

bool TargetInfo::is64Bits() const {
  return getTriple().isArch64Bit();
}

bool TargetInfo::isLittleEndian() const {
  // TODO: Do this properly. It is not defined purely by arch.
  return true;
}

StringRef TargetInfo::getEntry() const {
  return _options._entrySymbol;
}
} // end namespace lld
