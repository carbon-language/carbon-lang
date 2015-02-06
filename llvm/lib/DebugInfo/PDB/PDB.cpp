//===- PDB.cpp - base header file for creating a PDB reader -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"

#include "llvm/DebugInfo/PDB/PDB.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"

using namespace llvm;

std::unique_ptr<IPDBSession> llvm::createPDBReader(PDB_ReaderType Type,
                                                   StringRef Path) {
  // Create the correct concrete instance type based on the value of Type.
  return nullptr;
}
