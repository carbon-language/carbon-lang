//===- PDB.cpp - base header file for creating a PDB reader -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"

#include "llvm/ADT/StringRef.h"

#include "llvm/DebugInfo/PDB/PDB.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"
#if HAVE_DIA_SDK
#include "llvm/DebugInfo/PDB/DIA/DIASession.h"
#endif

using namespace llvm;

std::unique_ptr<IPDBSession> llvm::createPDBReader(PDB_ReaderType Type,
                                                   StringRef Path) {
  // Create the correct concrete instance type based on the value of Type.
#if HAVE_DIA_SDK
  return std::unique_ptr<DIASession>(DIASession::createFromPdb(Path));
#endif
  return nullptr;
}
