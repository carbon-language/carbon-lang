//===- ModuleSubstreamRecord.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/ModuleSubstreamRecord.h"

#include "llvm/DebugInfo/CodeView/StreamReader.h"
#include "llvm/DebugInfo/PDB/Raw/RawTypes.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

ModuleSubstreamRecord::ModuleSubstreamRecord()
    : Kind(ModuleSubstreamKind::None) {}

ModuleSubstreamRecord::ModuleSubstreamRecord(ModuleSubstreamKind Kind,
                                             StreamRef Data)
    : Kind(Kind), Data(Data) {}

Error ModuleSubstreamRecord::initialize(StreamRef Stream,
                                        ModuleSubstreamRecord &Info) {
  const ModuleSubsectionHeader *Header;
  StreamReader Reader(Stream);
  if (auto EC = Reader.readObject(Header))
    return EC;

  ModuleSubstreamKind Kind =
      static_cast<ModuleSubstreamKind>(uint32_t(Header->Kind));
  if (auto EC = Reader.readStreamRef(Info.Data, Header->Length))
    return EC;
  Info.Kind = Kind;
  return Error::success();
}

uint32_t ModuleSubstreamRecord::getRecordLength() const {
  return sizeof(ModuleSubsectionHeader) + Data.getLength();
}

ModuleSubstreamKind ModuleSubstreamRecord::getSubstreamKind() const {
  return Kind;
}

StreamRef ModuleSubstreamRecord::getRecordData() const { return Data; }
