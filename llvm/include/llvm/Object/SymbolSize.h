//===- SymbolSize.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Declares the writeArchive function for writing an archive file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_SYMBOLSIZE_H
#define LLVM_OBJECT_SYMBOLSIZE_H

#include "llvm/Object/ObjectFile.h"

namespace llvm {
namespace object {
ErrorOr<std::vector<std::pair<SymbolRef, uint64_t>>>
computeSymbolSizes(const ObjectFile &O);
}
} // namespace llvm

#endif
