//===- lib/ReaderWriter/ELF/Hexagon/HexagonTargetHandler.cpp --------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "HexagonTargetHandler.h"
#include "HexagonTargetInfo.h"

using namespace lld;
using namespace elf;

using namespace llvm::ELF;

HexagonTargetHandler::HexagonTargetHandler(HexagonTargetInfo &targetInfo)
    : DefaultTargetHandler(targetInfo), _relocationHandler(targetInfo),
      _targetLayout(targetInfo), _hexagonRuntimeFile(targetInfo) {
}
