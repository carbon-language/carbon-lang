//===- NameMapBuilder.cpp - PDB Name Map Builder ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/NameMapBuilder.h"

#include "llvm/DebugInfo/PDB/Raw/NameMap.h"

using namespace llvm;
using namespace llvm::pdb;

NameMapBuilder::NameMapBuilder() {}

Expected<std::unique_ptr<NameMap>> NameMapBuilder::build() {
  return llvm::make_unique<NameMap>();
}

uint32_t NameMapBuilder::calculateSerializedLength() const {
  // For now we write an empty name map, nothing else.
  return 5 * sizeof(uint32_t);
}
