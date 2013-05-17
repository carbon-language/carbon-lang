//===- lib/ReaderWriter/ELF/X86_64/X86_64TargetHandler.cpp ----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Atoms.h"
#include "X86_64TargetHandler.h"
#include "X86_64TargetInfo.h"

using namespace lld;
using namespace elf;

X86_64TargetHandler::X86_64TargetHandler(X86_64TargetInfo &targetInfo)
    : DefaultTargetHandler(targetInfo), _gotFile(targetInfo),
      _relocationHandler(targetInfo), _targetLayout(targetInfo) {}

void X86_64TargetHandler::addFiles(InputFiles &f) {
  _gotFile.addAtom(*new (_gotFile._alloc) GLOBAL_OFFSET_TABLEAtom(_gotFile));
  _gotFile.addAtom(*new (_gotFile._alloc) TLSGETADDRAtom(_gotFile));
  f.appendFile(_gotFile);
}
