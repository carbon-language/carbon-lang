//===- lib/ReaderWriter/ELF/ARM/ARMELFWriters.cpp -------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ARMELFWriters.h"
#include "ARMExecutableWriter.h"
#include "ARMDynamicLibraryWriter.h"

using namespace lld;
using namespace elf;

template class ARMELFWriter<ExecutableWriter<ELF32LE>>;
template class ARMELFWriter<DynamicLibraryWriter<ELF32LE>>;
