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


namespace lld {
namespace mach_o {

//===----------------------------------------------------------------------===//
//  KindHandler
//===----------------------------------------------------------------------===//

KindHandler::KindHandler() {
}

KindHandler::~KindHandler() {
}

KindHandler *KindHandler::makeHandler(WriterOptionsMachO::Architecture arch) {
  switch( arch ) {
   case WriterOptionsMachO::arch_x86_64:
      return new KindHandler_x86_64();
      break;
    case WriterOptionsMachO::arch_x86:
      return new KindHandler_x86();
      break;
    case WriterOptionsMachO::arch_armv6:
    case WriterOptionsMachO::arch_armv7:
      return new KindHandler_arm();
      break;
    default:
      assert(0 && "arch not supported");
  }
}


//===----------------------------------------------------------------------===//
//  KindHandler_x86_64
//===----------------------------------------------------------------------===//

KindHandler_x86_64::~KindHandler_x86_64() {
}

Reference::Kind KindHandler_x86_64::stringToKind(StringRef str) {
  if ( str.equals("none") )
    return none;
  else if ( str.equals("branch32") )
    return branch32;
  else if ( str.equals("ripRel32") )
    return ripRel32;
  else if ( str.equals("ripRel32_1") )
    return ripRel32_1;
  else if ( str.equals("ripRel32_2") )
    return ripRel32_2;
  else if ( str.equals("ripRel32_4") )
    return ripRel32_4;
  else if ( str.equals("gotLoadRipRel32") )
    return gotLoadRipRel32;
  else if ( str.equals("gotLoadRipRel32NowLea") )
    return gotLoadRipRel32NowLea;
  else if ( str.equals("gotUseRipRel32") )
    return gotUseRipRel32;
  else if ( str.equals("tlvLoadRipRel32") )
    return tlvLoadRipRel32;
  else if ( str.equals("tlvLoadRipRel32NowLea") )
    return tlvLoadRipRel32NowLea;
  else if ( str.equals("pointer64") )
    return pointer64;
  else if ( str.equals("pointerRel32") )
    return pointerRel32;
  else if ( str.equals("lazyTarget") )
    return lazyTarget;
  else if ( str.equals("lazyImmediate") )
    return lazyImmediate;
  else if ( str.equals("subordinateFDE") )
    return subordinateFDE;
  else if ( str.equals("subordinateLSDA") )
    return subordinateLSDA;
  
  assert(0 && "invalid x86_64 Reference kind");
  return 0;
}

StringRef KindHandler_x86_64::kindToString(Reference::Kind kind) {
  switch ( (Kinds)kind ) {
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
  assert(0 && "invalid x86_64 Reference kind");
  return StringRef();
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
  }
}


//===----------------------------------------------------------------------===//
//  KindHandler_x86
//===----------------------------------------------------------------------===//

KindHandler_x86::~KindHandler_x86() {
}

Reference::Kind KindHandler_x86::stringToKind(StringRef str) {
  if ( str.equals("none") )
    return none;
  else if ( str.equals("branch32") )
    return branch32;
  else if ( str.equals("abs32") )
    return abs32;
  else if ( str.equals("funcRel32") )
    return funcRel32;
  else if ( str.equals("pointer32") )
    return pointer32;
  else if ( str.equals("lazyTarget") )
    return lazyTarget;
  else if ( str.equals("lazyImmediate") )
    return lazyImmediate;
  
  assert(0 && "invalid x86 Reference kind");
  return 0;
}

StringRef KindHandler_x86::kindToString(Reference::Kind kind) {
  switch ( (Kinds)kind ) {
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
  assert(0 && "invalid x86 Reference kind");
  return StringRef();
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
  }
}


//===----------------------------------------------------------------------===//
//  KindHandler_arm
//===----------------------------------------------------------------------===//

KindHandler_arm::~KindHandler_arm() {
}

Reference::Kind KindHandler_arm::stringToKind(StringRef str) {
  if ( str.equals("none") )
    return none;
  else if ( str.equals("thumbBranch22") )
    return thumbBranch22;
  else if ( str.equals("armBranch24") )
    return armBranch24;
  else if ( str.equals("thumbAbsLow16") )
    return thumbAbsLow16;
  else if ( str.equals("thumbAbsHigh16") )
    return thumbAbsHigh16;
  else if ( str.equals("thumbPcRelLow16") )
    return thumbPcRelLow16;
  else if ( str.equals("thumbPcRelHigh16") )
    return thumbPcRelHigh16;
  else if ( str.equals("abs32") )
    return abs32;
  else if ( str.equals("pointer32") )
    return pointer32;
  else if ( str.equals("lazyTarget") )
    return lazyTarget;
  else if ( str.equals("lazyImmediate") )
    return lazyImmediate;
  else if ( str.equals("subordinateLSDA") )
    return subordinateLSDA;

  assert(0 && "invalid ARM Reference kind");
  return 0;
}

StringRef KindHandler_arm::kindToString(Reference::Kind kind) {
  switch ( (Kinds)kind ) {
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
  assert(0 && "invalid ARM Reference kind");
  return StringRef();
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
  }
}
 

} // namespace mach_o
} // namespace lld



