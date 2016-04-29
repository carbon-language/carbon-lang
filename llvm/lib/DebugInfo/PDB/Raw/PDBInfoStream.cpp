//===- PDBInfoStream.cpp - PDB Info Stream (Stream 1) Access ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/PDBInfoStream.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/PDB/Raw/StreamReader.h"

using namespace llvm;

PDBInfoStream::PDBInfoStream(PDBFile &File) : Pdb(File), Stream(1, File) {}

std::error_code PDBInfoStream::reload() {
  StreamReader Reader(Stream);

  support::ulittle32_t Value;

  Reader.readObject(&Value);
  Version = Value;
  if (Version < PdbRaw_ImplVer::PdbImplVC70)
    return std::make_error_code(std::errc::not_supported);

  Reader.readObject(&Value);
  Signature = Value;

  Reader.readObject(&Value);
  Age = Value;

  Reader.readObject(&Guid);
  NamedStreams.load(Reader);

  return std::error_code();
}

uint32_t PDBInfoStream::getNamedStreamIndex(llvm::StringRef Name) const {
  uint32_t Result;
  if (!NamedStreams.tryGetValue(Name, Result))
    return 0;
  return Result;
}

PdbRaw_ImplVer PDBInfoStream::getVersion() const {
  return static_cast<PdbRaw_ImplVer>(Version);
}

uint32_t PDBInfoStream::getSignature() const { return Signature; }

uint32_t PDBInfoStream::getAge() const { return Age; }

PDB_UniqueId PDBInfoStream::getGuid() const { return Guid; }
