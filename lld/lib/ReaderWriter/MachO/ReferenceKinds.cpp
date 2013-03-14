//===- lib/FileFormat/MachO/ReferenceKinds.cpp ----------------------===//
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
#include "llvm/ADT/Triple.h"

#include "llvm/Support/ErrorHandling.h"

namespace lld {
namespace mach_o {

//===----------------------------------------------------------------------===//
//  KindHandler
//===----------------------------------------------------------------------===//

KindHandler::KindHandler() {
}

KindHandler::~KindHandler() {
}

KindHandler *KindHandler::makeHandler(llvm::Triple::ArchType arch) {
  switch( arch ) {
    case llvm::Triple::x86_64:
      return new KindHandler_x86_64();
    case llvm::Triple::x86:
      return new KindHandler_x86();
    case llvm::Triple::arm:
      return new KindHandler_arm();
    default:
      llvm_unreachable("Unknown arch");
  }
}


//===----------------------------------------------------------------------===//
//  KindHandler_x86_64
//===----------------------------------------------------------------------===//

KindHandler_x86_64::~KindHandler_x86_64() {
}

Reference::Kind KindHandler_x86_64::stringToKind(StringRef str) {
  return llvm::StringSwitch<Reference::Kind>(str)
    .Case("none",                  none)
    .Case("branch32",              branch32)
    .Case("ripRel32",              ripRel32)
    .Case("ripRel32_1",            ripRel32_1)
    .Case("ripRel32_2",            ripRel32_2)
    .Case("ripRel32_4",            ripRel32_4)
    .Case("gotLoadRipRel32",       gotLoadRipRel32)
    .Case("gotLoadRipRel32NowLea", gotLoadRipRel32NowLea)
    .Case("gotUseRipRel32",        gotUseRipRel32)
    .Case("tlvLoadRipRel32",       tlvLoadRipRel32)
    .Case("tlvLoadRipRel32NowLea", tlvLoadRipRel32NowLea)
    .Case("pointer64",             pointer64)
    .Case("pointerRel32",          pointerRel32)
    .Case("lazyTarget",            lazyTarget)
    .Case("lazyImmediate",         lazyImmediate)
    .Case("subordinateFDE",        subordinateFDE)
    .Case("subordinateLSDA",       subordinateLSDA)
    .Default(invalid);

  llvm_unreachable("invalid x86_64 Reference kind");
}

StringRef KindHandler_x86_64::kindToString(Reference::Kind kind) {
  switch ( (Kinds)kind ) {
    case invalid:
      return StringRef("invalid");
    case none:
      return StringRef("none");
    case branch32:
      return StringRef("branch32");
    case ripRel32:
      return StringRef("ripRel32");
    case ripRel32_1:
      return StringRef("ripRel32_1");
    case ripRel32_2:
      return StringRef("ripRel32_2");
    case ripRel32_4:
      return StringRef("ripRel32_4");
    case gotLoadRipRel32:
      return StringRef("gotLoadRipRel32");
    case gotLoadRipRel32NowLea:
      return StringRef("gotLoadRipRel32NowLea");
    case gotUseRipRel32:
      return StringRef("gotUseRipRel32");
    case tlvLoadRipRel32:
      return StringRef("tlvLoadRipRel32");
    case tlvLoadRipRel32NowLea:
      return StringRef("tlvLoadRipRel32NowLea");
    case pointer64:
      return StringRef("pointer64");
    case pointerRel32:
      return StringRef("pointerRel32");
    case lazyTarget:
      return StringRef("lazyTarget");
    case lazyImmediate:
      return StringRef("lazyImmediate");
    case subordinateFDE:
      return StringRef("subordinateFDE");
    case subordinateLSDA:
      return StringRef("subordinateLSDA");
  }
  llvm_unreachable("invalid x86_64 Reference kind");
}

bool KindHandler_x86_64::isCallSite(Kind kind) {
  return (kind == branch32);
}

bool KindHandler_x86_64::isPointer(Kind kind) {
  return (kind == pointer64);
}

bool KindHandler_x86_64::isLazyImmediate(Kind kind) {
  return (kind == lazyImmediate);
}

bool KindHandler_x86_64::isLazyTarget(Kind kind) {
  return (kind == lazyTarget);
}


void KindHandler_x86_64::applyFixup(Kind kind, uint64_t addend,
                                    uint8_t *location, uint64_t fixupAddress,
                                    uint64_t targetAddress) {
  int32_t *loc32 = reinterpret_cast<int32_t*>(location);
  uint64_t* loc64 = reinterpret_cast<uint64_t*>(location);
  switch ( (Kinds)kind ) {
    case branch32:
    case ripRel32:
    case gotLoadRipRel32:
    case gotUseRipRel32:
    case tlvLoadRipRel32:
      *loc32 = (targetAddress - (fixupAddress+4)) + addend;
      break;
    case pointer64:
      *loc64 = targetAddress + addend;
      break;
    case ripRel32_1:
      *loc32 = (targetAddress - (fixupAddress+5)) + addend;
      break;
    case ripRel32_2:
      *loc32 = (targetAddress - (fixupAddress+6)) + addend;
      break;
    case ripRel32_4:
      *loc32 = (targetAddress - (fixupAddress+8)) + addend;
      break;
    case pointerRel32:
      *loc32 = (targetAddress - fixupAddress) + addend;
      break;
    case gotLoadRipRel32NowLea:
    case tlvLoadRipRel32NowLea:
      // Change MOVQ to LEA
      assert(location[-2] == 0x8B);
      location[-2] = 0x8D;
      *loc32 = (targetAddress - (fixupAddress+4)) + addend;
      break;
    case none:
    case lazyTarget:
    case lazyImmediate:
    case subordinateFDE:
    case subordinateLSDA:
      // do nothing
      break;
    case invalid:
      assert(0 && "invalid Reference Kind");
      break;
  }
}


//===----------------------------------------------------------------------===//
//  KindHandler_x86
//===----------------------------------------------------------------------===//

KindHandler_x86::~KindHandler_x86() {
}

Reference::Kind KindHandler_x86::stringToKind(StringRef str) {
  return llvm::StringSwitch<Reference::Kind>(str)
    .Case("none",                  none)
    .Case("branch32",              branch32)
    .Case("abs32",                 abs32)
    .Case("funcRel32",             funcRel32)
    .Case("pointer32",             pointer32)
    .Case("lazyTarget",            lazyTarget)
    .Case("lazyImmediate",         lazyImmediate)
    .Default(invalid);

  llvm_unreachable("invalid x86 Reference kind");
}

StringRef KindHandler_x86::kindToString(Reference::Kind kind) {
  switch ( (Kinds)kind ) {
    case invalid:
      return StringRef("invalid");
    case none:
      return StringRef("none");
    case branch32:
      return StringRef("branch32");
    case abs32:
      return StringRef("abs32");
    case funcRel32:
      return StringRef("funcRel32");
    case pointer32:
      return StringRef("pointer32");
    case lazyTarget:
      return StringRef("lazyTarget");
    case lazyImmediate:
      return StringRef("lazyImmediate");
    case subordinateFDE:
      return StringRef("subordinateFDE");
    case subordinateLSDA:
      return StringRef("subordinateLSDA");
  }
  llvm_unreachable("invalid x86 Reference kind");
}

bool KindHandler_x86::isCallSite(Kind kind) {
  return (kind == branch32);
}

bool KindHandler_x86::isPointer(Kind kind) {
  return (kind == pointer32);
}


bool KindHandler_x86::isLazyImmediate(Kind kind) {
  return (kind == lazyImmediate);
}


bool KindHandler_x86::isLazyTarget(Kind kind) {
  return (kind == lazyTarget);
}


void KindHandler_x86::applyFixup(Kind kind, uint64_t addend, uint8_t *location,
                  uint64_t fixupAddress, uint64_t targetAddress) {
  int32_t *loc32 = reinterpret_cast<int32_t*>(location);
  switch ( (Kinds)kind ) {
    case branch32:
      *loc32 = (targetAddress - (fixupAddress+4)) + addend;
      break;
    case pointer32:
    case abs32:
      *loc32 = targetAddress + addend;
      break;
    case funcRel32:
      *loc32 = targetAddress + addend;
      break;
    case none:
    case lazyTarget:
    case lazyImmediate:
    case subordinateFDE:
    case subordinateLSDA:
      // do nothing
      break;
    case invalid:
      assert(0 && "invalid Reference Kind");
      break;
  }
}


//===----------------------------------------------------------------------===//
//  KindHandler_arm
//===----------------------------------------------------------------------===//

KindHandler_arm::~KindHandler_arm() {
}

Reference::Kind KindHandler_arm::stringToKind(StringRef str) {
 return llvm::StringSwitch<Reference::Kind>(str)
    .Case("none",               none)
    .Case("thumbBranch22",      thumbBranch22)
    .Case("armBranch24",        armBranch24)
    .Case("thumbAbsLow16",      thumbAbsLow16)
    .Case("thumbAbsHigh16",     thumbAbsHigh16)
    .Case("thumbPcRelLow16",    thumbPcRelLow16)
    .Case("thumbPcRelHigh16",   thumbPcRelHigh16)
    .Case("abs32",              abs32)
    .Case("pointer32",          pointer32)
    .Case("lazyTarget",         lazyTarget)
    .Case("lazyImmediate",      lazyImmediate)
    .Case("subordinateLSDA",    subordinateLSDA)
    .Default(invalid);

  llvm_unreachable("invalid ARM Reference kind");
}

StringRef KindHandler_arm::kindToString(Reference::Kind kind) {
  switch ( (Kinds)kind ) {
    case invalid:
      return StringRef("invalid");
    case none:
      return StringRef("none");
    case thumbBranch22:
      return StringRef("thumbBranch22");
    case armBranch24:
      return StringRef("armBranch24");
    case thumbAbsLow16:
      return StringRef("thumbAbsLow16");
    case thumbAbsHigh16:
      return StringRef("thumbAbsHigh16");
    case thumbPcRelLow16:
      return StringRef("thumbPcRelLow16");
    case thumbPcRelHigh16:
      return StringRef("thumbPcRelHigh16");
    case abs32:
      return StringRef("abs32");
    case pointer32:
      return StringRef("pointer32");
    case lazyTarget:
      return StringRef("lazyTarget");
    case lazyImmediate:
      return StringRef("lazyImmediate");
    case subordinateLSDA:
      return StringRef("subordinateLSDA");
  }
  llvm_unreachable("invalid ARM Reference kind");
}

bool KindHandler_arm::isCallSite(Kind kind) {
  return (kind == thumbBranch22) || (kind == armBranch24);
}

bool KindHandler_arm::isPointer(Kind kind) {
  return (kind == pointer32);
}


bool KindHandler_arm::isLazyImmediate(Kind kind) {
  return (kind == lazyImmediate);
}


bool KindHandler_arm::isLazyTarget(Kind kind) {
  return (kind == lazyTarget);
}


void KindHandler_arm::applyFixup(Kind kind, uint64_t addend, uint8_t *location,
                  uint64_t fixupAddress, uint64_t targetAddress) {
  //int32_t *loc32 = reinterpret_cast<int32_t*>(location);
  switch ( (Kinds)kind ) {
    case thumbBranch22:
      // FIXME
      break;
    case armBranch24:
      // FIXME
      break;
    case thumbAbsLow16:
      // FIXME
      break;
    case thumbAbsHigh16:
      // FIXME
      break;
    case thumbPcRelLow16:
      // FIXME
      break;
    case thumbPcRelHigh16:
      // FIXME
      break;
    case abs32:
      // FIXME
      break;
    case pointer32:
      // FIXME
      break;
    case none:
    case lazyTarget:
    case lazyImmediate:
    case subordinateLSDA:
      // do nothing
      break;
    case invalid:
      assert(0 && "invalid Reference Kind");
      break;
  }
}


} // namespace mach_o
} // namespace lld



