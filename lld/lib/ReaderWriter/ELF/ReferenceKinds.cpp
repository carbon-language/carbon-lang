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

std::unique_ptr<KindHandler> KindHandler::makeHandler(uint16_t arch,
                                          llvm::support::endianness endian) {
  switch(arch) {
  case llvm::ELF::EM_HEXAGON:
    return std::unique_ptr<KindHandler>(new HexagonKindHandler());
  case llvm::ELF::EM_386:
    return std::unique_ptr<KindHandler>(new X86KindHandler());
  case llvm::ELF::EM_PPC:
    return std::unique_ptr<KindHandler>(new PPCKindHandler(endian));
  default:
    llvm_unreachable("arch not supported");
  }
}

} // namespace elf
} // namespace lld



