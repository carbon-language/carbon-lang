//===- InfoStream.cpp - PDB Info Stream (Stream 1) Access -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/DebugInfo/PDB/Native/RawTypes.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/BinaryStreamWriter.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::msf;
using namespace llvm::pdb;

InfoStream::InfoStream(std::unique_ptr<MappedBlockStream> Stream)
    : Stream(std::move(Stream)) {}

Error InfoStream::reload() {
  BinaryStreamReader Reader(*Stream);

  const InfoStreamHeader *H;
  if (auto EC = Reader.readObject(H))
    return joinErrors(
        std::move(EC),
        make_error<RawError>(raw_error_code::corrupt_file,
                             "PDB Stream does not contain a header."));

  switch (H->Version) {
  case PdbImplVC70:
  case PdbImplVC80:
  case PdbImplVC110:
  case PdbImplVC140:
    break;
  default:
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Unsupported PDB stream version.");
  }

  Version = H->Version;
  Signature = H->Signature;
  Age = H->Age;
  Guid = H->Guid;

  uint32_t Offset = Reader.getOffset();
  if (auto EC = NamedStreams.load(Reader))
    return EC;
  uint32_t NewOffset = Reader.getOffset();
  NamedStreamMapByteSize = NewOffset - Offset;

  bool Stop = false;
  while (!Stop && !Reader.empty()) {
    PdbRaw_FeatureSig Sig;
    if (auto EC = Reader.readEnum(Sig))
      return EC;
    switch (Sig) {
    case PdbRaw_FeatureSig::VC110:
      // No other flags for VC110 PDB.
      Stop = true;
      LLVM_FALLTHROUGH;
    case PdbRaw_FeatureSig::VC140:
      Features |= PdbFeatureContainsIdStream;
      break;
    case PdbRaw_FeatureSig::NoTypeMerge:
      Features |= PdbFeatureNoTypeMerging;
      break;
    case PdbRaw_FeatureSig::MinimalDebugInfo:
      Features |= PdbFeatureMinimalDebugInfo;
    default:
      continue;
    }
    FeatureSignatures.push_back(Sig);
  }
  return Error::success();
}

uint32_t InfoStream::getStreamSize() const { return Stream->getLength(); }

uint32_t InfoStream::getNamedStreamIndex(llvm::StringRef Name) const {
  uint32_t Result;
  if (!NamedStreams.get(Name, Result))
    return 0;
  return Result;
}

iterator_range<StringMapConstIterator<uint32_t>>
InfoStream::named_streams() const {
  return NamedStreams.entries();
}

PdbRaw_ImplVer InfoStream::getVersion() const {
  return static_cast<PdbRaw_ImplVer>(Version);
}

uint32_t InfoStream::getSignature() const { return Signature; }

uint32_t InfoStream::getAge() const { return Age; }

PDB_UniqueId InfoStream::getGuid() const { return Guid; }

uint32_t InfoStream::getNamedStreamMapByteSize() const {
  return NamedStreamMapByteSize;
}

PdbRaw_Features InfoStream::getFeatures() const { return Features; }

ArrayRef<PdbRaw_FeatureSig> InfoStream::getFeatureSignatures() const {
  return FeatureSignatures;
}

const NamedStreamMap &InfoStream::getNamedStreams() const {
  return NamedStreams;
}
