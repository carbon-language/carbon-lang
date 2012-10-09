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
//  KindHandler_ppc
//  TODO: more to do here
//===----------------------------------------------------------------------===//

KindHandler_ppc::~KindHandler_ppc() {
}

/// \brief The following relocation routines are derived from the
///  SYSTEM V APPLICATION BINARY INTERFACE: PowerPC Processor Supplement
/// Symbols used:
///  A: Added used to compute the value, r_addend
///  P: Place address of the field being relocated, r_offset
///  S: Value of the symbol whose index resides in the relocation entry.

namespace ppc {
int reloc_NONE(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  return KindHandler_ppc::NoError;
}

/// \brief low24 (S + A - P) >> 2 : Verify
int reloc_B24_PCREL(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  int32_t result = (uint32_t)(((S + A) - P));
  if ((result < 0x1000000) && (result > -0x1000000)) {
    result &= ~-(0x1000000);
    *reinterpret_cast<llvm::support::ubig32_t *>(location) = result |
                      *reinterpret_cast<llvm::support::ubig32_t *>(location);
    return KindHandler_ppc::NoError;
  }
  return KindHandler_ppc::Overflow;
}
} // namespace ppc

KindHandler_ppc::KindHandler_ppc(llvm::support::endianness endian){
  _fixupHandler[llvm::ELF::R_PPC_REL24] = ppc::reloc_B24_PCREL;
}

Reference::Kind KindHandler_ppc::stringToKind(StringRef str) {
  return llvm::StringSwitch<Reference::Kind>(str)
    .Case("none",                  none)
    .Case("R_PPC_REL24", llvm::ELF::R_PPC_REL24)
    .Default(invalid);
}

StringRef KindHandler_ppc::kindToString(Reference::Kind kind) {
  switch ((int32_t)kind) {
  case llvm::ELF::R_PPC_REL24:
    return "R_PPC_REL24";
  default:
    return "none";
  }
}

bool KindHandler_ppc::isCallSite(Kind kind) {
  llvm_unreachable("Unimplemented: KindHandler_ppc::isCallSite");
  return false;
}

bool KindHandler_ppc::isPointer(Kind kind) {
  llvm_unreachable("Unimplemented: KindHandler_ppc::isPointer");
  return false;
}

bool KindHandler_ppc::isLazyImmediate(Kind kind) {
  llvm_unreachable("Unimplemented: KindHandler_ppc::isLazyImmediate");
  return false;
}

bool KindHandler_ppc::isLazyTarget(Kind kind) {
  llvm_unreachable("Unimplemented: KindHandler_ppc::isLazyTarget");
  return false;
}

void KindHandler_ppc::applyFixup(int32_t reloc, uint64_t addend,
                                 uint8_t *location, uint64_t fixupAddress,
                                 uint64_t targetAddress) {
  int error;
  if (_fixupHandler[reloc])
  {
    error = (*_fixupHandler[reloc])(location,
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
