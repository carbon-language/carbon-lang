//===- lib/ReaderWriter/ELF/X86Reference.cpp ----------------------------===//
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

#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ELF.h"

namespace lld {
namespace elf {

//===----------------------------------------------------------------------===//
//  X86KindHandler
//  TODO: more to do here
//===----------------------------------------------------------------------===//

X86KindHandler::~X86KindHandler() {
}

/// \brief The following relocation routines are derived from the
///  SYSTEM V APPLICATION BINARY INTERFACE: Intel386 Architecture Processor
///  Supplement (Fourth Edition)
/// Symbols used:
///  P: Place, address of the field being relocated, r_offset
///  S: Value of the symbol whose index resides in the relocation entry.
///  A: Addend used to compute the value, r_addend

namespace x86 {
int relocNone(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  return X86KindHandler::NoError;
}

/// \brief R_386_32 - word32:  S + A
int reloc32(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  int32_t result = (uint32_t)(S + A);
  *reinterpret_cast<llvm::support::ulittle32_t *>(location) = result |
            (uint32_t)*reinterpret_cast<llvm::support::ulittle32_t *>(location);
  return X86KindHandler::NoError;
}
/// \brief R_386_PC32 - word32: S + A - P
int relocPC32(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  uint32_t result = (uint32_t)((S + A) - P);
  *reinterpret_cast<llvm::support::ulittle32_t *>(location) = result +
            (uint32_t)*reinterpret_cast<llvm::support::ulittle32_t *>(location);
  return X86KindHandler::NoError;
}

} // namespace x86

X86KindHandler::X86KindHandler(){
  _fixupHandler[llvm::ELF::R_386_32] = x86::reloc32;
  _fixupHandler[llvm::ELF::R_386_PC32] = x86::relocPC32;
}

Reference::Kind X86KindHandler::stringToKind(StringRef str) {
  return llvm::StringSwitch<Reference::Kind>(str)
    .Case("none", none)
    .Case("R_386_32", llvm::ELF::R_386_32)
    .Case("R_386_PC32", llvm::ELF::R_386_PC32)
    .Default(invalid);
}

StringRef X86KindHandler::kindToString(Reference::Kind kind) {
  switch ((int32_t)kind) {
  case llvm::ELF::R_386_32:
    return "R_386_32";
  case llvm::ELF::R_386_PC32:
    return "R_386_PC32";
  default:
    return "none";
  }
}

bool X86KindHandler::isCallSite(Kind kind) {
  llvm_unreachable("Unimplemented: X86KindHandler::isCallSite");
  return false;
}

bool X86KindHandler::isPointer(Kind kind) {
  llvm_unreachable("Unimplemented: X86KindHandler::isPointer");
  return false;
}

bool X86KindHandler::isLazyImmediate(Kind kind) {
  llvm_unreachable("Unimplemented: X86KindHandler::isLazyImmediate");
  return false;
}

bool X86KindHandler::isLazyTarget(Kind kind) {
  llvm_unreachable("Unimplemented: X86KindHandler::isLazyTarget");
  return false;
}

void X86KindHandler::applyFixup(int32_t reloc, uint64_t addend,
                                 uint8_t *location, uint64_t fixupAddress,
                                 uint64_t targetAddress) {
  int error;
  if (_fixupHandler[reloc]) {
    error = (_fixupHandler[reloc])(location,
                                   fixupAddress, targetAddress, addend);

    switch ((RelocationError)error) {
    case NoError:
      return;
    }
  }
}

} // namespace elf
} // namespace lld
