//===- lib/ReaderWriter/ELF/X86_64Reference.cpp ---------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ReferenceKinds.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#include "llvm/Support/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace lld {
namespace elf {
X86_64KindHandler::X86_64KindHandler(){
}

X86_64KindHandler::~X86_64KindHandler() {
}

Reference::Kind X86_64KindHandler::stringToKind(StringRef str) {
  return llvm::StringSwitch<Reference::Kind>(str)
    .Case("none", none)
    .Default(invalid);
}

StringRef X86_64KindHandler::kindToString(Reference::Kind kind) {
  switch ((int32_t)kind) {
  default:
    return "none";
  }
}

bool X86_64KindHandler::isCallSite(Kind kind) {
  llvm_unreachable("Unimplemented: X86KindHandler::isCallSite");
  return false;
}

bool X86_64KindHandler::isPointer(Kind kind) {
  llvm_unreachable("Unimplemented: X86KindHandler::isPointer");
  return false;
}

bool X86_64KindHandler::isLazyImmediate(Kind kind) {
  llvm_unreachable("Unimplemented: X86KindHandler::isLazyImmediate");
  return false;
}

bool X86_64KindHandler::isLazyTarget(Kind kind) {
  llvm_unreachable("Unimplemented: X86KindHandler::isLazyTarget");
  return false;
}

void X86_64KindHandler::applyFixup(int32_t reloc, uint64_t addend,
                                   uint8_t *location, uint64_t fixupAddress,
                                   uint64_t targetAddress) {
  if (_fixupHandler[reloc])
    _fixupHandler[reloc](location, fixupAddress, targetAddress, addend);
  else {
    llvm::errs() << "Unknown relocation type: " << reloc << "\n";
    llvm_unreachable("Unknown relocation type.");
  }
}
} // end namespace elf
} // end namespace lld
