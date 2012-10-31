//===- lib/ReaderWriter/ELF/PPCReference.cpp ----------------------------===//
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
//  PPCKindHandler
//  TODO: more to do here
//===----------------------------------------------------------------------===//

PPCKindHandler::~PPCKindHandler() {
}

/// \brief The following relocation routines are derived from the
///  SYSTEM V APPLICATION BINARY INTERFACE: PowerPC Processor Supplement
/// Symbols used:
///  A: Added used to compute the value, r_addend
///  P: Place address of the field being relocated, r_offset
///  S: Value of the symbol whose index resides in the relocation entry.

namespace ppc {
int relocNONE(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  return PPCKindHandler::NoError;
}

/// \brief low24 (S + A - P) >> 2 : Verify
int relocB24PCREL(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  int32_t result = (uint32_t)(((S + A) - P));
  if ((result < 0x1000000) && (result > -0x1000000)) {
    result &= ~-(0x1000000);
    *reinterpret_cast<llvm::support::ubig32_t *>(location) = result |
               (uint32_t)*reinterpret_cast<llvm::support::ubig32_t *>(location);
    return PPCKindHandler::NoError;
  }
  return PPCKindHandler::Overflow;
}
} // namespace ppc

PPCKindHandler::PPCKindHandler(llvm::support::endianness endian){
  _fixupHandler[llvm::ELF::R_PPC_REL24] = ppc::relocB24PCREL;
}

Reference::Kind PPCKindHandler::stringToKind(StringRef str) {
  return llvm::StringSwitch<Reference::Kind>(str)
    .Case("none",                  none)
    .Case("R_PPC_REL24", llvm::ELF::R_PPC_REL24)
    .Default(invalid);
}

StringRef PPCKindHandler::kindToString(Reference::Kind kind) {
  switch ((int32_t)kind) {
  case llvm::ELF::R_PPC_REL24:
    return "R_PPC_REL24";
  default:
    return "none";
  }
}

bool PPCKindHandler::isCallSite(Kind kind) {
  llvm_unreachable("Unimplemented: PPCKindHandler::isCallSite");
  return false;
}

bool PPCKindHandler::isPointer(Kind kind) {
  llvm_unreachable("Unimplemented: PPCKindHandler::isPointer");
  return false;
}

bool PPCKindHandler::isLazyImmediate(Kind kind) {
  llvm_unreachable("Unimplemented: PPCKindHandler::isLazyImmediate");
  return false;
}

bool PPCKindHandler::isLazyTarget(Kind kind) {
  llvm_unreachable("Unimplemented: PPCKindHandler::isLazyTarget");
  return false;
}

void PPCKindHandler::applyFixup(int32_t reloc, uint64_t addend,
                                 uint8_t *location, uint64_t fixupAddress,
                                 uint64_t targetAddress) {
  int error;
  if (_fixupHandler[reloc])
  {
    error = (_fixupHandler[reloc])(location,
                                   fixupAddress, targetAddress, addend);

    switch ((RelocationError)error) {
    case NoError:
      return;
    case Overflow:
      llvm::report_fatal_error("applyFixup relocation overflow");
      return;
    }
  }
}

} // namespace elf
} // namespace lld
