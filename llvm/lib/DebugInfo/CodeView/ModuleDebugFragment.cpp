//===- ModuleDebugFragment.cpp --------------------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/ModuleDebugFragment.h"

#include "llvm/Support/BinaryStreamReader.h"

using namespace llvm;
using namespace llvm::codeview;

ModuleDebugFragment::ModuleDebugFragment()
    : Kind(ModuleDebugFragmentKind::None) {}

ModuleDebugFragment::ModuleDebugFragment(ModuleDebugFragmentKind Kind,
                                         BinaryStreamRef Data)
    : Kind(Kind), Data(Data) {}

Error ModuleDebugFragment::initialize(BinaryStreamRef Stream,
                                      ModuleDebugFragment &Info) {
  const ModuleDebugFragmentHeader *Header;
  BinaryStreamReader Reader(Stream);
  if (auto EC = Reader.readObject(Header))
    return EC;

  ModuleDebugFragmentKind Kind =
      static_cast<ModuleDebugFragmentKind>(uint32_t(Header->Kind));
  if (auto EC = Reader.readStreamRef(Info.Data, Header->Length))
    return EC;
  Info.Kind = Kind;
  return Error::success();
}

uint32_t ModuleDebugFragment::getRecordLength() const {
  return sizeof(ModuleDebugFragmentHeader) + Data.getLength();
}

ModuleDebugFragmentKind ModuleDebugFragment::kind() const { return Kind; }

BinaryStreamRef ModuleDebugFragment::getRecordData() const { return Data; }
