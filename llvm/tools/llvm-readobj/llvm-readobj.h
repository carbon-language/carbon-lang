//===- llvm-readobj.h - Dump contents of an Object File -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_READ_OBJ_H
#define LLVM_TOOLS_READ_OBJ_H

#include "llvm/Support/ErrorOr.h"

namespace llvm {
namespace object { class ObjectFile; }
class raw_ostream;

ErrorOr<void> dumpELFDynamicTable(object::ObjectFile *O, raw_ostream &OS);
} // end namespace llvm

#endif
