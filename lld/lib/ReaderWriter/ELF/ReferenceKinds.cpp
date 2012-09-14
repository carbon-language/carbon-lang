//===- lib/ReaderWriter/ELF/ReferenceKinds.cpp ----------------------------===//
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

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ELF.h"
namespace lld {
namespace elf {

//===----------------------------------------------------------------------===//
//  KindHandler
//===----------------------------------------------------------------------===//

KindHandler::KindHandler() {
}

KindHandler::~KindHandler() {
}

std::unique_ptr<KindHandler> KindHandler::makeHandler(uint16_t arch) {
  switch(arch) {
  case llvm::ELF::EM_HEXAGON:
    return std::unique_ptr<KindHandler>(new KindHandler_hexagon());
  case llvm::ELF::EM_386:
    return std::unique_ptr<KindHandler>(new KindHandler_x86());
  default:
    llvm_unreachable("arch not supported");
  }
}

//===----------------------------------------------------------------------===//
//  KindHandler_x86
//  TODO: more to do here
//===----------------------------------------------------------------------===//

KindHandler_x86::~KindHandler_x86() {}

Reference::Kind KindHandler_x86::stringToKind(StringRef str) {
  return llvm::StringSwitch<Reference::Kind>(str)
    .Case("none", none)
    .Default(invalid);
}

StringRef KindHandler_x86::kindToString(Reference::Kind kind) {
  switch ( (Kinds)kind ) {
    case invalid:
      return StringRef("invalid");
    case none:
      return StringRef("none");
  }
  llvm_unreachable("invalid x86 Reference kind");
}

bool KindHandler_x86::isCallSite(Kind kind) {
  llvm_unreachable("Unimplemented: KindHandler_x86::isCallSite");
  return false;
}

bool KindHandler_x86::isPointer(Kind kind) {
  llvm_unreachable("Unimplemented: KindHandler_x86::isPointer");
  return false;
}
 
bool KindHandler_x86::isLazyImmediate(Kind kind) {
  llvm_unreachable("Unimplemented: KindHandler_x86::isLazyImmediate");
  return false;
}
 
bool KindHandler_x86::isLazyTarget(Kind kind) {
  llvm_unreachable("Unimplemented: KindHandler_x86::isLazyTarget");
  return false;
}

 
void KindHandler_x86::applyFixup(Kind kind, uint64_t addend,  
                                    uint8_t *location, uint64_t fixupAddress, 
                                    uint64_t targetAddress) {
  switch ((Kinds)kind) {
  case none:
    // do nothing
    break;
  case invalid:
    assert(0 && "invalid Reference Kind");
    break;
  }
}

//===----------------------------------------------------------------------===//
//  KindHandler_hexagon
//  TODO: more to do here
//===----------------------------------------------------------------------===//

KindHandler_hexagon::~KindHandler_hexagon() {
}

Reference::Kind KindHandler_hexagon::stringToKind(StringRef str) {
  return llvm::StringSwitch<Reference::Kind>(str)
    .Case("none",                  none)
    .Default(invalid);

  llvm_unreachable("invalid hexagon Reference kind");
}

StringRef KindHandler_hexagon::kindToString(Reference::Kind kind) {
  switch ( (Kinds)kind ) {
    case invalid:
      return StringRef("invalid");
    case none:
      return StringRef("none");
  }
  llvm_unreachable("invalid hexagon Reference kind");
}

bool KindHandler_hexagon::isCallSite(Kind kind) {
  llvm_unreachable("Unimplemented: KindHandler_hexagon::isCallSite");
  return false;
}

bool KindHandler_hexagon::isPointer(Kind kind) {
  llvm_unreachable("Unimplemented: KindHandler_hexagon::isPointer");
  return false;
}
 
bool KindHandler_hexagon::isLazyImmediate(Kind kind) {
  llvm_unreachable("Unimplemented: KindHandler_hexagon::isLazyImmediate");
  return false;
}
 
bool KindHandler_hexagon::isLazyTarget(Kind kind) {
  llvm_unreachable("Unimplemented: KindHandler_hexagon::isLazyTarget");
  return false;
}

 
void KindHandler_hexagon::applyFixup(Kind kind, uint64_t addend,  
                                    uint8_t *location, uint64_t fixupAddress, 
                                    uint64_t targetAddress) {
  switch ((Kinds)kind) {
  case none:
    // do nothing
    break;
  case invalid:
    llvm_unreachable("invalid Reference Kind");
    break;
  }
}



} // namespace elf
} // namespace lld



