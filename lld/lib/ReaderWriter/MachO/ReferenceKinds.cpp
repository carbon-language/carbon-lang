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
    return KindHandler_x86_64::none;
  else if ( str.equals("call32") )
    return KindHandler_x86_64::call32;
  else if ( str.equals("ripRel32") )
    return KindHandler_x86_64::ripRel32;
  else if ( str.equals("gotLoad32") )
    return KindHandler_x86_64::gotLoad32;
  else if ( str.equals("gotUse32") )
    return KindHandler_x86_64::gotUse32;
  else if ( str.equals("pointer64") )
    return KindHandler_x86_64::pointer64;
  else if ( str.equals("lea32WasGot") )
    return KindHandler_x86_64::lea32WasGot;
  else if ( str.equals("lazyTarget") )
    return KindHandler_x86_64::lazyTarget;
  else if ( str.equals("lazyImm") )
    return KindHandler_x86_64::lazyImm;
  else if ( str.equals("gotTarget") )
    return KindHandler_x86_64::gotTarget;
  
  assert(0 && "invalid x86_64 Reference kind");
  return 0;
}

StringRef KindHandler_x86_64::kindToString(Reference::Kind kind) {
  switch ( (Kinds)kind ) {
    case none:
      return StringRef("none");
    case call32: 
      return StringRef("call32");
    case ripRel32:
      return StringRef("ripRel32");
    case gotLoad32:
      return StringRef("gotLoad32");
    case gotUse32:
      return StringRef("gotUse32");
    case pointer64:
      return StringRef("pointer64");
    case lea32WasGot:
      return StringRef("lea32WasGot");
    case lazyTarget:
      return StringRef("lazyTarget");
    case lazyImm:
      return StringRef("lazyImm");
    case gotTarget:
      return StringRef("gotTarget");
  }
  assert(0 && "invalid x86_64 Reference kind");
  return StringRef();
}

bool KindHandler_x86_64::isCallSite(Kind kind) {
  return (kind == call32);
}

bool KindHandler_x86_64::isPointer(Kind kind) {
  return (kind == pointer64);
}

 
bool KindHandler_x86_64::isLazyImmediate(Kind kind) {
  return (kind == lazyImm);
}

 
bool KindHandler_x86_64::isLazyTarget(Kind kind) {
  return (kind == lazyTarget);
}

 
void KindHandler_x86_64::applyFixup(Kind kind, uint64_t addend, uint8_t *location, 
                  uint64_t fixupAddress, uint64_t targetAddress) {
  int32_t *loc32 = reinterpret_cast<int32_t*>(location);
  uint64_t* loc64 = reinterpret_cast<uint64_t*>(location);
  switch ( (Kinds)kind ) {
    case call32: 
    case ripRel32:
    case gotLoad32:
    case gotUse32:
      *loc32 = (targetAddress - (fixupAddress+4)) + addend;
      break;
    case pointer64:
      *loc64 = targetAddress + addend;
      break;
    case lea32WasGot:
      break;
    case none:
    case lazyTarget:
    case lazyImm:
    case gotTarget:
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
    return KindHandler_x86::none;
  else if ( str.equals("call32") )
    return KindHandler_x86::call32;
  else if ( str.equals("abs32") )
    return KindHandler_x86::abs32;
  else if ( str.equals("pointer32") )
    return KindHandler_x86::pointer32;
  else if ( str.equals("lazyTarget") )
    return KindHandler_x86::lazyTarget;
  else if ( str.equals("lazyImm") )
    return KindHandler_x86::lazyImm;
  
  assert(0 && "invalid x86 Reference kind");
  return 0;
}

StringRef KindHandler_x86::kindToString(Reference::Kind kind) {
  switch ( (Kinds)kind ) {
    case none:
      return StringRef("none");
    case call32: 
      return StringRef("call32");
    case abs32:
      return StringRef("abs32");
    case pointer32:
      return StringRef("pointer32");
    case lazyTarget:
      return StringRef("lazyTarget");
    case lazyImm:
      return StringRef("lazyImm");
  }
  assert(0 && "invalid x86 Reference kind");
  return StringRef();
}

bool KindHandler_x86::isCallSite(Kind kind) {
  return (kind == call32);
}

bool KindHandler_x86::isPointer(Kind kind) {
  return (kind == pointer32);
}

 
bool KindHandler_x86::isLazyImmediate(Kind kind) {
  return (kind == lazyImm);
}

 
bool KindHandler_x86::isLazyTarget(Kind kind) {
  return (kind == lazyTarget);
}

 
void KindHandler_x86::applyFixup(Kind kind, uint64_t addend, uint8_t *location, 
                  uint64_t fixupAddress, uint64_t targetAddress) {
  int32_t *loc32 = reinterpret_cast<int32_t*>(location);
  switch ( (Kinds)kind ) {
    case call32: 
      *loc32 = (targetAddress - (fixupAddress+4)) + addend;
      break;
    case pointer32:
    case abs32:
      *loc32 = targetAddress + addend;
      break;
    case none:
    case lazyTarget:
    case lazyImm:
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
    return KindHandler_arm::none;
  else if ( str.equals("br22") )
    return KindHandler_arm::br22;
  else if ( str.equals("pointer32") )
    return KindHandler_arm::pointer32;
  else if ( str.equals("lazyTarget") )
    return KindHandler_arm::lazyTarget;
  else if ( str.equals("lazyImm") )
    return KindHandler_arm::lazyImm;

  assert(0 && "invalid ARM Reference kind");
  return 0;
}

StringRef KindHandler_arm::kindToString(Reference::Kind kind) {
  switch ( (Kinds)kind ) {
    case none:
      return StringRef("none");
    case br22: 
      return StringRef("br22");
    case pointer32:
      return StringRef("pointer32");
    case lazyTarget:
      return StringRef("lazyTarget");
    case lazyImm:
      return StringRef("lazyImm");
  }
  assert(0 && "invalid ARM Reference kind");
  return StringRef();
}

bool KindHandler_arm::isCallSite(Kind kind) {
  return (kind == br22);
}

bool KindHandler_arm::isPointer(Kind kind) {
  return (kind == pointer32);
}

 
bool KindHandler_arm::isLazyImmediate(Kind kind) {
  return (kind == lazyImm);
}

 
bool KindHandler_arm::isLazyTarget(Kind kind) {
  return (kind == lazyTarget);
}

 
void KindHandler_arm::applyFixup(Kind kind, uint64_t addend, uint8_t *location, 
                  uint64_t fixupAddress, uint64_t targetAddress) {
  //int32_t *loc32 = reinterpret_cast<int32_t*>(location);
  switch ( (Kinds)kind ) {
    case br22: 
      // FIXME
      break;
    case pointer32: 
      // FIXME
      break;
    case none:
    case lazyTarget:
    case lazyImm:
      // do nothing
      break;
  }
}
 

} // namespace mach_o
} // namespace lld



