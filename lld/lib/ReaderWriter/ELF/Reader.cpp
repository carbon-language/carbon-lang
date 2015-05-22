//===- lib/ReaderWriter/ELF/Reader.cpp ------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the ELF Reader and all helper sub classes to consume an ELF
/// file and produces atoms out of it.
///
//===----------------------------------------------------------------------===//

#include "ELFReader.h"
#include <map>
#include <vector>

using llvm::support::endianness;
using namespace llvm::object;

namespace lld {

// This dynamic registration of a handler causes support for all ELF
// architectures to be pulled into the linker.  If we want to support making a
// linker that only supports one ELF architecture, we'd need to change this
// to have a different registration method for each architecture.
void Registry::addSupportELFObjects(ELFLinkingContext &ctx) {

  // Tell registry about the ELF object file parser.
  add(ctx.getTargetHandler().getObjReader());

  // Tell registry about the relocation name to number mapping for this arch.
  ctx.registerRelocationNames(*this);
}

void Registry::addSupportELFDynamicSharedObjects(ELFLinkingContext &ctx) {
  // Tell registry about the ELF dynamic shared library file parser.
  add(ctx.getTargetHandler().getDSOReader());
}

} // end namespace lld
