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
#include "llvm/ADT/Triple.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"

namespace lld {
namespace elf {
KindHandler::KindHandler() {}

KindHandler::~KindHandler() {}

std::unique_ptr<KindHandler>
KindHandler::makeHandler(llvm::Triple::ArchType arch, bool isLittleEndian) {
  switch(arch) {
  case llvm::Triple::hexagon:
    return std::unique_ptr<KindHandler>(new HexagonKindHandler());
  case llvm::Triple::x86:
    return std::unique_ptr<KindHandler>(new X86KindHandler());
  case llvm::Triple::x86_64:
    return std::unique_ptr<KindHandler>(new X86_64KindHandler());
  case llvm::Triple::ppc:
    return std::unique_ptr<KindHandler>(
        new PPCKindHandler(isLittleEndian ? llvm::support::little
                                          : llvm::support::big));
  default:
    llvm_unreachable("arch not supported");
  }
}
} // namespace elf
} // namespace lld
