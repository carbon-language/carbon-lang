//===- PDB.h - base header file for creating a PDB reader -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDB_H
#define LLVM_DEBUGINFO_PDB_PDB_H

#include <memory>

#include "PDBTypes.h"

namespace llvm {
class StringRef;

std::unique_ptr<IPDBSession> createPDBReader(PDB_ReaderType Type,
                                             StringRef Path);
}

#endif
