//===- BytesOutputStyle.cpp ----------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "BytesOutputStyle.h"

#include "StreamUtil.h"
#include "llvm-pdbutil.h"

#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;
using namespace llvm::msf;
using namespace llvm::pdb;

namespace {
struct StreamSpec {
  uint32_t SI = 0;
  uint32_t Begin = 0;
  uint32_t Size = 0;
};
} // namespace

static Expected<StreamSpec> parseStreamSpec(StringRef Str) {
  StreamSpec Result;
  if (Str.consumeInteger(0, Result.SI))
    return make_error<RawError>(raw_error_code::invalid_format,
                                "Invalid Stream Specification");
  if (Str.consume_front(":")) {
    if (Str.consumeInteger(0, Result.Begin))
      return make_error<RawError>(raw_error_code::invalid_format,
                                  "Invalid Stream Specification");
  }
  if (Str.consume_front("@")) {
    if (Str.consumeInteger(0, Result.Size))
      return make_error<RawError>(raw_error_code::invalid_format,
                                  "Invalid Stream Specification");
  }

  if (!Str.empty())
    return make_error<RawError>(raw_error_code::invalid_format,
                                "Invalid Stream Specification");
  return Result;
}

static SmallVector<StreamSpec, 2> parseStreamSpecs(LinePrinter &P) {
  SmallVector<StreamSpec, 2> Result;

  for (auto &Str : opts::bytes::DumpStreamData) {
    auto ESS = parseStreamSpec(Str);
    if (!ESS) {
      P.formatLine("Error parsing stream spec {0}: {1}", Str,
                   toString(ESS.takeError()));
      continue;
    }
    Result.push_back(*ESS);
  }
  return Result;
}

static void printHeader(LinePrinter &P, const Twine &S) {
  P.NewLine();
  P.formatLine("{0,=60}", S);
  P.formatLine("{0}", fmt_repeat('=', 60));
}

BytesOutputStyle::BytesOutputStyle(PDBFile &File)
    : File(File), P(2, false, outs()) {}

Error BytesOutputStyle::dump() {

  if (opts::bytes::DumpBlockRange.hasValue()) {
    auto &R = *opts::bytes::DumpBlockRange;
    uint32_t Max = R.Max.getValueOr(R.Min);

    if (Max < R.Min)
      return make_error<StringError>(
          "Invalid block range specified.  Max < Min",
          inconvertibleErrorCode());
    if (Max >= File.getBlockCount())
      return make_error<StringError>(
          "Invalid block range specified.  Requested block out of bounds",
          inconvertibleErrorCode());

    dumpBlockRanges(R.Min, Max);
    P.NewLine();
  }

  if (opts::bytes::DumpByteRange.hasValue()) {
    auto &R = *opts::bytes::DumpByteRange;
    uint32_t Max = R.Max.getValueOr(File.getFileSize());

    if (Max < R.Min)
      return make_error<StringError>("Invalid byte range specified.  Max < Min",
                                     inconvertibleErrorCode());
    if (Max >= File.getFileSize())
      return make_error<StringError>(
          "Invalid byte range specified.  Requested byte larger than file size",
          inconvertibleErrorCode());

    dumpByteRanges(R.Min, Max);
    P.NewLine();
  }

  if (!opts::bytes::DumpStreamData.empty()) {
    dumpStreamBytes();
    P.NewLine();
  }

  if (opts::bytes::NameMap) {
    dumpNameMap();
    P.NewLine();
  }
  return Error::success();
}

void BytesOutputStyle::dumpNameMap() {
  printHeader(P, "Named Stream Map");

  AutoIndent Indent(P);

  auto &InfoS = Err(File.getPDBInfoStream());
  BinarySubstreamRef NS = InfoS.getNamedStreamsBuffer();
  auto Layout = File.getStreamLayout(StreamPDB);
  P.formatMsfStreamData("Named Stream Map", File, Layout, NS);
}

void BytesOutputStyle::dumpBlockRanges(uint32_t Min, uint32_t Max) {
  printHeader(P, "MSF Blocks");

  AutoIndent Indent(P);
  for (uint32_t I = Min; I <= Max; ++I) {
    uint64_t Base = I;
    Base *= File.getBlockSize();

    auto ExpectedData = File.getBlockData(I, File.getBlockSize());
    if (!ExpectedData) {
      P.formatLine("Could not get block {0}.  Reason = {1}", I,
                   toString(ExpectedData.takeError()));
      continue;
    }
    std::string Label = formatv("Block {0}", I).str();
    P.formatBinary(Label, *ExpectedData, Base, 0);
  }
}

void BytesOutputStyle::dumpByteRanges(uint32_t Min, uint32_t Max) {
  printHeader(P, "MSF Bytes");

  AutoIndent Indent(P);

  BinaryStreamReader Reader(File.getMsfBuffer());
  ArrayRef<uint8_t> Data;
  consumeError(Reader.skip(Min));
  uint32_t Size = Max - Min + 1;
  auto EC = Reader.readBytes(Data, Size);
  assert(!EC);
  consumeError(std::move(EC));
  P.formatBinary("Bytes", Data, Min);
}

void BytesOutputStyle::dumpStreamBytes() {
  if (StreamPurposes.empty())
    discoverStreamPurposes(File, StreamPurposes);

  printHeader(P, "Stream Data");
  ExitOnError Err("Unexpected error reading stream data");

  auto Specs = parseStreamSpecs(P);

  for (const auto &Spec : Specs) {
    AutoIndent Indent(P);
    if (Spec.SI >= StreamPurposes.size()) {
      P.formatLine("Stream {0}: Not present", Spec.SI);
      continue;
    }
    P.formatMsfStreamData("Data", File, Spec.SI, StreamPurposes[Spec.SI],
                          Spec.Begin, Spec.Size);
  }
}
