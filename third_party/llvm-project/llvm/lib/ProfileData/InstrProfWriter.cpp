//===- InstrProfWriter.cpp - Instrumented profiling writer ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing profiling data for clang's
// instrumentation based PGO and coverage.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/InstrProfWriter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/ProfileSummary.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/MemProf.h"
#include "llvm/ProfileData/ProfileCommon.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/OnDiskHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace llvm;

// A struct to define how the data stream should be patched. For Indexed
// profiling, only uint64_t data type is needed.
struct PatchItem {
  uint64_t Pos; // Where to patch.
  uint64_t *D;  // Pointer to an array of source data.
  int N;        // Number of elements in \c D array.
};

namespace llvm {

// A wrapper class to abstract writer stream with support of bytes
// back patching.
class ProfOStream {
public:
  ProfOStream(raw_fd_ostream &FD)
      : IsFDOStream(true), OS(FD), LE(FD, support::little) {}
  ProfOStream(raw_string_ostream &STR)
      : IsFDOStream(false), OS(STR), LE(STR, support::little) {}

  uint64_t tell() { return OS.tell(); }
  void write(uint64_t V) { LE.write<uint64_t>(V); }

  // \c patch can only be called when all data is written and flushed.
  // For raw_string_ostream, the patch is done on the target string
  // directly and it won't be reflected in the stream's internal buffer.
  void patch(PatchItem *P, int NItems) {
    using namespace support;

    if (IsFDOStream) {
      raw_fd_ostream &FDOStream = static_cast<raw_fd_ostream &>(OS);
      const uint64_t LastPos = FDOStream.tell();
      for (int K = 0; K < NItems; K++) {
        FDOStream.seek(P[K].Pos);
        for (int I = 0; I < P[K].N; I++)
          write(P[K].D[I]);
      }
      // Reset the stream to the last position after patching so that users
      // don't accidentally overwrite data. This makes it consistent with
      // the string stream below which replaces the data directly.
      FDOStream.seek(LastPos);
    } else {
      raw_string_ostream &SOStream = static_cast<raw_string_ostream &>(OS);
      std::string &Data = SOStream.str(); // with flush
      for (int K = 0; K < NItems; K++) {
        for (int I = 0; I < P[K].N; I++) {
          uint64_t Bytes = endian::byte_swap<uint64_t, little>(P[K].D[I]);
          Data.replace(P[K].Pos + I * sizeof(uint64_t), sizeof(uint64_t),
                       (const char *)&Bytes, sizeof(uint64_t));
        }
      }
    }
  }

  // If \c OS is an instance of \c raw_fd_ostream, this field will be
  // true. Otherwise, \c OS will be an raw_string_ostream.
  bool IsFDOStream;
  raw_ostream &OS;
  support::endian::Writer LE;
};

class InstrProfRecordWriterTrait {
public:
  using key_type = StringRef;
  using key_type_ref = StringRef;

  using data_type = const InstrProfWriter::ProfilingData *const;
  using data_type_ref = const InstrProfWriter::ProfilingData *const;

  using hash_value_type = uint64_t;
  using offset_type = uint64_t;

  support::endianness ValueProfDataEndianness = support::little;
  InstrProfSummaryBuilder *SummaryBuilder;
  InstrProfSummaryBuilder *CSSummaryBuilder;

  InstrProfRecordWriterTrait() = default;

  static hash_value_type ComputeHash(key_type_ref K) {
    return IndexedInstrProf::ComputeHash(K);
  }

  static std::pair<offset_type, offset_type>
  EmitKeyDataLength(raw_ostream &Out, key_type_ref K, data_type_ref V) {
    using namespace support;

    endian::Writer LE(Out, little);

    offset_type N = K.size();
    LE.write<offset_type>(N);

    offset_type M = 0;
    for (const auto &ProfileData : *V) {
      const InstrProfRecord &ProfRecord = ProfileData.second;
      M += sizeof(uint64_t); // The function hash
      M += sizeof(uint64_t); // The size of the Counts vector
      M += ProfRecord.Counts.size() * sizeof(uint64_t);

      // Value data
      M += ValueProfData::getSize(ProfileData.second);
    }
    LE.write<offset_type>(M);

    return std::make_pair(N, M);
  }

  void EmitKey(raw_ostream &Out, key_type_ref K, offset_type N) {
    Out.write(K.data(), N);
  }

  void EmitData(raw_ostream &Out, key_type_ref, data_type_ref V, offset_type) {
    using namespace support;

    endian::Writer LE(Out, little);
    for (const auto &ProfileData : *V) {
      const InstrProfRecord &ProfRecord = ProfileData.second;
      if (NamedInstrProfRecord::hasCSFlagInHash(ProfileData.first))
        CSSummaryBuilder->addRecord(ProfRecord);
      else
        SummaryBuilder->addRecord(ProfRecord);

      LE.write<uint64_t>(ProfileData.first); // Function hash
      LE.write<uint64_t>(ProfRecord.Counts.size());
      for (uint64_t I : ProfRecord.Counts)
        LE.write<uint64_t>(I);

      // Write value data
      std::unique_ptr<ValueProfData> VDataPtr =
          ValueProfData::serializeFrom(ProfileData.second);
      uint32_t S = VDataPtr->getSize();
      VDataPtr->swapBytesFromHost(ValueProfDataEndianness);
      Out.write((const char *)VDataPtr.get(), S);
    }
  }
};

} // end namespace llvm

InstrProfWriter::InstrProfWriter(bool Sparse)
    : Sparse(Sparse), InfoObj(new InstrProfRecordWriterTrait()) {}

InstrProfWriter::~InstrProfWriter() { delete InfoObj; }

// Internal interface for testing purpose only.
void InstrProfWriter::setValueProfDataEndianness(
    support::endianness Endianness) {
  InfoObj->ValueProfDataEndianness = Endianness;
}

void InstrProfWriter::setOutputSparse(bool Sparse) {
  this->Sparse = Sparse;
}

void InstrProfWriter::addRecord(NamedInstrProfRecord &&I, uint64_t Weight,
                                function_ref<void(Error)> Warn) {
  auto Name = I.Name;
  auto Hash = I.Hash;
  addRecord(Name, Hash, std::move(I), Weight, Warn);
}

void InstrProfWriter::overlapRecord(NamedInstrProfRecord &&Other,
                                    OverlapStats &Overlap,
                                    OverlapStats &FuncLevelOverlap,
                                    const OverlapFuncFilters &FuncFilter) {
  auto Name = Other.Name;
  auto Hash = Other.Hash;
  Other.accumulateCounts(FuncLevelOverlap.Test);
  if (FunctionData.find(Name) == FunctionData.end()) {
    Overlap.addOneUnique(FuncLevelOverlap.Test);
    return;
  }
  if (FuncLevelOverlap.Test.CountSum < 1.0f) {
    Overlap.Overlap.NumEntries += 1;
    return;
  }
  auto &ProfileDataMap = FunctionData[Name];
  bool NewFunc;
  ProfilingData::iterator Where;
  std::tie(Where, NewFunc) =
      ProfileDataMap.insert(std::make_pair(Hash, InstrProfRecord()));
  if (NewFunc) {
    Overlap.addOneMismatch(FuncLevelOverlap.Test);
    return;
  }
  InstrProfRecord &Dest = Where->second;

  uint64_t ValueCutoff = FuncFilter.ValueCutoff;
  if (!FuncFilter.NameFilter.empty() && Name.contains(FuncFilter.NameFilter))
    ValueCutoff = 0;

  Dest.overlap(Other, Overlap, FuncLevelOverlap, ValueCutoff);
}

void InstrProfWriter::addRecord(StringRef Name, uint64_t Hash,
                                InstrProfRecord &&I, uint64_t Weight,
                                function_ref<void(Error)> Warn) {
  auto &ProfileDataMap = FunctionData[Name];

  bool NewFunc;
  ProfilingData::iterator Where;
  std::tie(Where, NewFunc) =
      ProfileDataMap.insert(std::make_pair(Hash, InstrProfRecord()));
  InstrProfRecord &Dest = Where->second;

  auto MapWarn = [&](instrprof_error E) {
    Warn(make_error<InstrProfError>(E));
  };

  if (NewFunc) {
    // We've never seen a function with this name and hash, add it.
    Dest = std::move(I);
    if (Weight > 1)
      Dest.scale(Weight, 1, MapWarn);
  } else {
    // We're updating a function we've seen before.
    Dest.merge(I, Weight, MapWarn);
  }

  Dest.sortValueData();
}

void InstrProfWriter::addMemProfRecord(
    const Function::GUID Id, const memprof::IndexedMemProfRecord &Record) {
  auto Result = MemProfRecordData.insert({Id, Record});
  // If we inserted a new record then we are done.
  if (Result.second) {
    return;
  }
  memprof::IndexedMemProfRecord &Existing = Result.first->second;
  Existing.merge(Record);
}

bool InstrProfWriter::addMemProfFrame(const memprof::FrameId Id,
                                      const memprof::Frame &Frame,
                                      function_ref<void(Error)> Warn) {
  auto Result = MemProfFrameData.insert({Id, Frame});
  // If a mapping already exists for the current frame id and it does not
  // match the new mapping provided then reset the existing contents and bail
  // out. We don't support the merging of memprof data whose Frame -> Id
  // mapping across profiles is inconsistent.
  if (!Result.second && Result.first->second != Frame) {
    Warn(make_error<InstrProfError>(instrprof_error::malformed,
                                    "frame to id mapping mismatch"));
    return false;
  }
  return true;
}

void InstrProfWriter::mergeRecordsFromWriter(InstrProfWriter &&IPW,
                                             function_ref<void(Error)> Warn) {
  for (auto &I : IPW.FunctionData)
    for (auto &Func : I.getValue())
      addRecord(I.getKey(), Func.first, std::move(Func.second), 1, Warn);

  MemProfFrameData.reserve(IPW.MemProfFrameData.size());
  for (auto &I : IPW.MemProfFrameData) {
    // If we weren't able to add the frame mappings then it doesn't make sense
    // to try to merge the records from this profile.
    if (!addMemProfFrame(I.first, I.second, Warn))
      return;
  }

  MemProfRecordData.reserve(IPW.MemProfRecordData.size());
  for (auto &I : IPW.MemProfRecordData) {
    addMemProfRecord(I.first, I.second);
  }
}

bool InstrProfWriter::shouldEncodeData(const ProfilingData &PD) {
  if (!Sparse)
    return true;
  for (const auto &Func : PD) {
    const InstrProfRecord &IPR = Func.second;
    if (llvm::any_of(IPR.Counts, [](uint64_t Count) { return Count > 0; }))
      return true;
  }
  return false;
}

static void setSummary(IndexedInstrProf::Summary *TheSummary,
                       ProfileSummary &PS) {
  using namespace IndexedInstrProf;

  const std::vector<ProfileSummaryEntry> &Res = PS.getDetailedSummary();
  TheSummary->NumSummaryFields = Summary::NumKinds;
  TheSummary->NumCutoffEntries = Res.size();
  TheSummary->set(Summary::MaxFunctionCount, PS.getMaxFunctionCount());
  TheSummary->set(Summary::MaxBlockCount, PS.getMaxCount());
  TheSummary->set(Summary::MaxInternalBlockCount, PS.getMaxInternalCount());
  TheSummary->set(Summary::TotalBlockCount, PS.getTotalCount());
  TheSummary->set(Summary::TotalNumBlocks, PS.getNumCounts());
  TheSummary->set(Summary::TotalNumFunctions, PS.getNumFunctions());
  for (unsigned I = 0; I < Res.size(); I++)
    TheSummary->setEntry(I, Res[I]);
}

Error InstrProfWriter::writeImpl(ProfOStream &OS) {
  using namespace IndexedInstrProf;

  OnDiskChainedHashTableGenerator<InstrProfRecordWriterTrait> Generator;

  InstrProfSummaryBuilder ISB(ProfileSummaryBuilder::DefaultCutoffs);
  InfoObj->SummaryBuilder = &ISB;
  InstrProfSummaryBuilder CSISB(ProfileSummaryBuilder::DefaultCutoffs);
  InfoObj->CSSummaryBuilder = &CSISB;

  // Populate the hash table generator.
  for (const auto &I : FunctionData)
    if (shouldEncodeData(I.getValue()))
      Generator.insert(I.getKey(), &I.getValue());

  // Write the header.
  IndexedInstrProf::Header Header;
  Header.Magic = IndexedInstrProf::Magic;
  Header.Version = IndexedInstrProf::ProfVersion::CurrentVersion;
  if (static_cast<bool>(ProfileKind & InstrProfKind::IRInstrumentation))
    Header.Version |= VARIANT_MASK_IR_PROF;
  if (static_cast<bool>(ProfileKind & InstrProfKind::ContextSensitive))
    Header.Version |= VARIANT_MASK_CSIR_PROF;
  if (static_cast<bool>(ProfileKind &
                        InstrProfKind::FunctionEntryInstrumentation))
    Header.Version |= VARIANT_MASK_INSTR_ENTRY;
  if (static_cast<bool>(ProfileKind & InstrProfKind::SingleByteCoverage))
    Header.Version |= VARIANT_MASK_BYTE_COVERAGE;
  if (static_cast<bool>(ProfileKind & InstrProfKind::FunctionEntryOnly))
    Header.Version |= VARIANT_MASK_FUNCTION_ENTRY_ONLY;
  if (static_cast<bool>(ProfileKind & InstrProfKind::MemProf))
    Header.Version |= VARIANT_MASK_MEMPROF;

  Header.Unused = 0;
  Header.HashType = static_cast<uint64_t>(IndexedInstrProf::HashType);
  Header.HashOffset = 0;
  Header.MemProfOffset = 0;
  int N = sizeof(IndexedInstrProf::Header) / sizeof(uint64_t);

  // Only write out all the fields except 'HashOffset' and 'MemProfOffset'. We
  // need to remember the offset of these fields to allow back patching later.
  for (int I = 0; I < N - 2; I++)
    OS.write(reinterpret_cast<uint64_t *>(&Header)[I]);

  // Save the location of Header.HashOffset field in \c OS.
  uint64_t HashTableStartFieldOffset = OS.tell();
  // Reserve the space for HashOffset field.
  OS.write(0);

  // Save the location of MemProf profile data. This is stored in two parts as
  // the schema and as a separate on-disk chained hashtable.
  uint64_t MemProfSectionOffset = OS.tell();
  // Reserve space for the MemProf table field to be patched later if this
  // profile contains memory profile information.
  OS.write(0);

  // Reserve space to write profile summary data.
  uint32_t NumEntries = ProfileSummaryBuilder::DefaultCutoffs.size();
  uint32_t SummarySize = Summary::getSize(Summary::NumKinds, NumEntries);
  // Remember the summary offset.
  uint64_t SummaryOffset = OS.tell();
  for (unsigned I = 0; I < SummarySize / sizeof(uint64_t); I++)
    OS.write(0);
  uint64_t CSSummaryOffset = 0;
  uint64_t CSSummarySize = 0;
  if (static_cast<bool>(ProfileKind & InstrProfKind::ContextSensitive)) {
    CSSummaryOffset = OS.tell();
    CSSummarySize = SummarySize / sizeof(uint64_t);
    for (unsigned I = 0; I < CSSummarySize; I++)
      OS.write(0);
  }

  // Write the hash table.
  uint64_t HashTableStart = Generator.Emit(OS.OS, *InfoObj);

  // Write the MemProf profile data if we have it. This includes a simple schema
  // with the format described below followed by the hashtable:
  // uint64_t RecordTableOffset = RecordTableGenerator.Emit
  // uint64_t FramePayloadOffset = Stream offset before emitting the frame table
  // uint64_t FrameTableOffset = FrameTableGenerator.Emit
  // uint64_t Num schema entries
  // uint64_t Schema entry 0
  // uint64_t Schema entry 1
  // ....
  // uint64_t Schema entry N - 1
  // OnDiskChainedHashTable MemProfRecordData
  // OnDiskChainedHashTable MemProfFrameData
  uint64_t MemProfSectionStart = 0;
  if (static_cast<bool>(ProfileKind & InstrProfKind::MemProf)) {
    MemProfSectionStart = OS.tell();
    OS.write(0ULL); // Reserve space for the memprof record table offset.
    OS.write(0ULL); // Reserve space for the memprof frame payload offset.
    OS.write(0ULL); // Reserve space for the memprof frame table offset.

    auto Schema = memprof::PortableMemInfoBlock::getSchema();
    OS.write(static_cast<uint64_t>(Schema.size()));
    for (const auto Id : Schema) {
      OS.write(static_cast<uint64_t>(Id));
    }

    auto RecordWriter = std::make_unique<memprof::RecordWriterTrait>();
    RecordWriter->Schema = &Schema;
    OnDiskChainedHashTableGenerator<memprof::RecordWriterTrait>
        RecordTableGenerator;
    for (auto &I : MemProfRecordData) {
      // Insert the key (func hash) and value (memprof record).
      RecordTableGenerator.insert(I.first, I.second);
    }

    uint64_t RecordTableOffset =
        RecordTableGenerator.Emit(OS.OS, *RecordWriter);

    uint64_t FramePayloadOffset = OS.tell();

    auto FrameWriter = std::make_unique<memprof::FrameWriterTrait>();
    OnDiskChainedHashTableGenerator<memprof::FrameWriterTrait>
        FrameTableGenerator;
    for (auto &I : MemProfFrameData) {
      // Insert the key (frame id) and value (frame contents).
      FrameTableGenerator.insert(I.first, I.second);
    }

    uint64_t FrameTableOffset = FrameTableGenerator.Emit(OS.OS, *FrameWriter);

    PatchItem PatchItems[] = {
        {MemProfSectionStart, &RecordTableOffset, 1},
        {MemProfSectionStart + sizeof(uint64_t), &FramePayloadOffset, 1},
        {MemProfSectionStart + 2 * sizeof(uint64_t), &FrameTableOffset, 1},
    };
    OS.patch(PatchItems, 3);
  }

  // Allocate space for data to be serialized out.
  std::unique_ptr<IndexedInstrProf::Summary> TheSummary =
      IndexedInstrProf::allocSummary(SummarySize);
  // Compute the Summary and copy the data to the data
  // structure to be serialized out (to disk or buffer).
  std::unique_ptr<ProfileSummary> PS = ISB.getSummary();
  setSummary(TheSummary.get(), *PS);
  InfoObj->SummaryBuilder = nullptr;

  // For Context Sensitive summary.
  std::unique_ptr<IndexedInstrProf::Summary> TheCSSummary = nullptr;
  if (static_cast<bool>(ProfileKind & InstrProfKind::ContextSensitive)) {
    TheCSSummary = IndexedInstrProf::allocSummary(SummarySize);
    std::unique_ptr<ProfileSummary> CSPS = CSISB.getSummary();
    setSummary(TheCSSummary.get(), *CSPS);
  }
  InfoObj->CSSummaryBuilder = nullptr;

  // Now do the final patch:
  PatchItem PatchItems[] = {
      // Patch the Header.HashOffset field.
      {HashTableStartFieldOffset, &HashTableStart, 1},
      // Patch the Header.MemProfOffset (=0 for profiles without MemProf data).
      {MemProfSectionOffset, &MemProfSectionStart, 1},
      // Patch the summary data.
      {SummaryOffset, reinterpret_cast<uint64_t *>(TheSummary.get()),
       (int)(SummarySize / sizeof(uint64_t))},
      {CSSummaryOffset, reinterpret_cast<uint64_t *>(TheCSSummary.get()),
       (int)CSSummarySize}};

  OS.patch(PatchItems, sizeof(PatchItems) / sizeof(*PatchItems));

  for (const auto &I : FunctionData)
    for (const auto &F : I.getValue())
      if (Error E = validateRecord(F.second))
        return E;

  return Error::success();
}

Error InstrProfWriter::write(raw_fd_ostream &OS) {
  // Write the hash table.
  ProfOStream POS(OS);
  return writeImpl(POS);
}

std::unique_ptr<MemoryBuffer> InstrProfWriter::writeBuffer() {
  std::string Data;
  raw_string_ostream OS(Data);
  ProfOStream POS(OS);
  // Write the hash table.
  if (Error E = writeImpl(POS))
    return nullptr;
  // Return this in an aligned memory buffer.
  return MemoryBuffer::getMemBufferCopy(Data);
}

static const char *ValueProfKindStr[] = {
#define VALUE_PROF_KIND(Enumerator, Value, Descr) #Enumerator,
#include "llvm/ProfileData/InstrProfData.inc"
};

Error InstrProfWriter::validateRecord(const InstrProfRecord &Func) {
  for (uint32_t VK = 0; VK <= IPVK_Last; VK++) {
    uint32_t NS = Func.getNumValueSites(VK);
    if (!NS)
      continue;
    for (uint32_t S = 0; S < NS; S++) {
      uint32_t ND = Func.getNumValueDataForSite(VK, S);
      std::unique_ptr<InstrProfValueData[]> VD = Func.getValueForSite(VK, S);
      bool WasZero = false;
      for (uint32_t I = 0; I < ND; I++)
        if ((VK != IPVK_IndirectCallTarget) && (VD[I].Value == 0)) {
          if (WasZero)
            return make_error<InstrProfError>(instrprof_error::invalid_prof);
          WasZero = true;
        }
    }
  }

  return Error::success();
}

void InstrProfWriter::writeRecordInText(StringRef Name, uint64_t Hash,
                                        const InstrProfRecord &Func,
                                        InstrProfSymtab &Symtab,
                                        raw_fd_ostream &OS) {
  OS << Name << "\n";
  OS << "# Func Hash:\n" << Hash << "\n";
  OS << "# Num Counters:\n" << Func.Counts.size() << "\n";
  OS << "# Counter Values:\n";
  for (uint64_t Count : Func.Counts)
    OS << Count << "\n";

  uint32_t NumValueKinds = Func.getNumValueKinds();
  if (!NumValueKinds) {
    OS << "\n";
    return;
  }

  OS << "# Num Value Kinds:\n" << Func.getNumValueKinds() << "\n";
  for (uint32_t VK = 0; VK < IPVK_Last + 1; VK++) {
    uint32_t NS = Func.getNumValueSites(VK);
    if (!NS)
      continue;
    OS << "# ValueKind = " << ValueProfKindStr[VK] << ":\n" << VK << "\n";
    OS << "# NumValueSites:\n" << NS << "\n";
    for (uint32_t S = 0; S < NS; S++) {
      uint32_t ND = Func.getNumValueDataForSite(VK, S);
      OS << ND << "\n";
      std::unique_ptr<InstrProfValueData[]> VD = Func.getValueForSite(VK, S);
      for (uint32_t I = 0; I < ND; I++) {
        if (VK == IPVK_IndirectCallTarget)
          OS << Symtab.getFuncNameOrExternalSymbol(VD[I].Value) << ":"
             << VD[I].Count << "\n";
        else
          OS << VD[I].Value << ":" << VD[I].Count << "\n";
      }
    }
  }

  OS << "\n";
}

Error InstrProfWriter::writeText(raw_fd_ostream &OS) {
  // Check CS first since it implies an IR level profile.
  if (static_cast<bool>(ProfileKind & InstrProfKind::ContextSensitive))
    OS << "# CSIR level Instrumentation Flag\n:csir\n";
  else if (static_cast<bool>(ProfileKind & InstrProfKind::IRInstrumentation))
    OS << "# IR level Instrumentation Flag\n:ir\n";

  if (static_cast<bool>(ProfileKind &
                        InstrProfKind::FunctionEntryInstrumentation))
    OS << "# Always instrument the function entry block\n:entry_first\n";
  InstrProfSymtab Symtab;

  using FuncPair = detail::DenseMapPair<uint64_t, InstrProfRecord>;
  using RecordType = std::pair<StringRef, FuncPair>;
  SmallVector<RecordType, 4> OrderedFuncData;

  for (const auto &I : FunctionData) {
    if (shouldEncodeData(I.getValue())) {
      if (Error E = Symtab.addFuncName(I.getKey()))
        return E;
      for (const auto &Func : I.getValue())
        OrderedFuncData.push_back(std::make_pair(I.getKey(), Func));
    }
  }

  llvm::sort(OrderedFuncData, [](const RecordType &A, const RecordType &B) {
    return std::tie(A.first, A.second.first) <
           std::tie(B.first, B.second.first);
  });

  for (const auto &record : OrderedFuncData) {
    const StringRef &Name = record.first;
    const FuncPair &Func = record.second;
    writeRecordInText(Name, Func.first, Func.second, Symtab, OS);
  }

  for (const auto &record : OrderedFuncData) {
    const FuncPair &Func = record.second;
    if (Error E = validateRecord(Func.second))
      return E;
  }

  return Error::success();
}
