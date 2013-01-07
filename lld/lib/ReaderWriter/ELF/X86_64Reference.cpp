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

using namespace llvm::ELF;

namespace {
/// \brief R_X86_64_PC32 - word32: S + A - P
int relocPC32(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  uint32_t result = (uint32_t)((S + A) - P);
  *reinterpret_cast<llvm::support::ulittle32_t *>(location) = result +
            (uint32_t)*reinterpret_cast<llvm::support::ulittle32_t *>(location);
  return 0;
}

/// \brief R_X86_64_32 - word32:  S + A
int reloc32(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  int32_t result = (uint32_t)(S + A);
  *reinterpret_cast<llvm::support::ulittle32_t *>(location) = result |
            (uint32_t)*reinterpret_cast<llvm::support::ulittle32_t *>(location);
  // TODO: Make sure that the result zero extends to the 64bit value.
  return 0;
}

/// \brief R_X86_64_32S - word32:  S + A
int reloc32S(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  int32_t result = (int32_t)(S + A);
  *reinterpret_cast<llvm::support::little32_t *>(location) = result |
            (int32_t)*reinterpret_cast<llvm::support::little32_t *>(location);
  // TODO: Make sure that the result sign extends to the 64bit value.
  return 0;
}
} // end anon namespace

namespace lld {
namespace elf {
X86_64KindHandler::X86_64KindHandler(){
  _fixupHandler[R_X86_64_PC32] = relocPC32;
  _fixupHandler[R_X86_64_32] = reloc32;
  _fixupHandler[R_X86_64_32S] = reloc32S;
}

X86_64KindHandler::~X86_64KindHandler() {
}

Reference::Kind X86_64KindHandler::stringToKind(StringRef str) {
  return llvm::StringSwitch<Reference::Kind>(str)
    .Case("none", none)
    .Case("R_X86_64_PC32", R_X86_64_PC32)
    .Case("R_X86_64_32S", R_X86_64_32S)
    .Default(invalid);
}

StringRef X86_64KindHandler::kindToString(Reference::Kind kind) {
  switch ((int32_t)kind) {
  case R_X86_64_PC32:
    return "R_X86_64_PC32";
  case R_X86_64_32S:
    return "R_X86_64_32S";
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
