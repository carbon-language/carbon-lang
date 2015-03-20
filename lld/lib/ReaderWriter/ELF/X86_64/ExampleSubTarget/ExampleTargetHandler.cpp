//===- lib/ReaderWriter/ELF/X86_64/ExampleTarget/ExampleTargetHandler.cpp -===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ExampleTargetHandler.h"
#include "X86_64ExecutableWriter.h"
#include "ExampleLinkingContext.h"

using namespace lld;
using namespace elf;

ExampleTargetHandler::ExampleTargetHandler(ExampleLinkingContext &c)
    : X86_64TargetHandler(c), _exampleContext(c) {}

std::unique_ptr<Writer> ExampleTargetHandler::getWriter() {
  return std::unique_ptr<Writer>(
      new X86_64ExecutableWriter(_exampleContext, *_x86_64TargetLayout));
}
