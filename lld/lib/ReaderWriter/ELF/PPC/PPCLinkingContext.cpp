//===- lib/ReaderWriter/ELF/PPC/PPCLinkingContext.cpp ---------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PPCLinkingContext.h"
#include "PPCTargetHandler.h"
#include "lld/Core/LLVM.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorOr.h"

using namespace lld;

elf::PPCLinkingContext::PPCLinkingContext(llvm::Triple triple)
    : ELFLinkingContext(triple, std::unique_ptr<TargetHandlerBase>(
                        new PPCTargetHandler(*this))) {}

