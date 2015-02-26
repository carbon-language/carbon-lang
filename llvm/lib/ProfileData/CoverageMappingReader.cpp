//=-- CoverageMappingReader.cpp - Code coverage mapping reader ----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for reading coverage mapping data for
// instrumentation based coverage.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/CoverageMappingReader.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LEB128.h"

using namespace llvm;
using namespace coverage;
using namespace object;

#define DEBUG_TYPE "coverage-mapping"

void CoverageMappingIterator::increment() {
  // Check if all the records were read or if an error occurred while reading
  // the next record.
  if (Reader->readNextRecord(Record))
    *this = CoverageMappingIterator();
}

std::error_code RawCoverageReader::readULEB128(uint64_t &Result) {
  if (Data.size() < 1)
    return error(instrprof_error::truncated);
  unsigned N = 0;
  Result = decodeULEB128(reinterpret_cast<const uint8_t *>(Data.data()), &N);
  if (N > Data.size())
    return error(instrprof_error::malformed);
  Data = Data.substr(N);
  return success();
}

std::error_code RawCoverageReader::readIntMax(uint64_t &Result,
                                              uint64_t MaxPlus1) {
  if (auto Err = readULEB128(Result))
    return Err;
  if (Result >= MaxPlus1)
    return error(instrprof_error::malformed);
  return success();
}

std::error_code RawCoverageReader::readSize(uint64_t &Result) {
  if (auto Err = readULEB128(Result))
    return Err;
  // Sanity check the number.
  if (Result > Data.size())
    return error(instrprof_error::malformed);
  return success();
}

std::error_code RawCoverageReader::readString(StringRef &Result) {
  uint64_t Length;
  if (auto Err = readSize(Length))
    return Err;
  Result = Data.substr(0, Length);
  Data = Data.substr(Length);
  return success();
}

std::error_code RawCoverageFilenamesReader::read() {
  uint64_t NumFilenames;
  if (auto Err = readSize(NumFilenames))
    return Err;
  for (size_t I = 0; I < NumFilenames; ++I) {
    StringRef Filename;
    if (auto Err = readString(Filename))
      return Err;
    Filenames.push_back(Filename);
  }
  return success();
}

std::error_code RawCoverageMappingReader::decodeCounter(unsigned Value,
                                                        Counter &C) {
  auto Tag = Value & Counter::EncodingTagMask;
  switch (Tag) {
  case Counter::Zero:
    C = Counter::getZero();
    return success();
  case Counter::CounterValueReference:
    C = Counter::getCounter(Value >> Counter::EncodingTagBits);
    return success();
  default:
    break;
  }
  Tag -= Counter::Expression;
  switch (Tag) {
  case CounterExpression::Subtract:
  case CounterExpression::Add: {
    auto ID = Value >> Counter::EncodingTagBits;
    if (ID >= Expressions.size())
      return error(instrprof_error::malformed);
    Expressions[ID].Kind = CounterExpression::ExprKind(Tag);
    C = Counter::getExpression(ID);
    break;
  }
  default:
    return error(instrprof_error::malformed);
  }
  return success();
}

std::error_code RawCoverageMappingReader::readCounter(Counter &C) {
  uint64_t EncodedCounter;
  if (auto Err =
          readIntMax(EncodedCounter, std::numeric_limits<unsigned>::max()))
    return Err;
  if (auto Err = decodeCounter(EncodedCounter, C))
    return Err;
  return success();
}

static const unsigned EncodingExpansionRegionBit = 1
                                                   << Counter::EncodingTagBits;

/// \brief Read the sub-array of regions for the given inferred file id.
/// \param NumFileIDs the number of file ids that are defined for this
/// function.
std::error_code RawCoverageMappingReader::readMappingRegionsSubArray(
    std::vector<CounterMappingRegion> &MappingRegions, unsigned InferredFileID,
    size_t NumFileIDs) {
  uint64_t NumRegions;
  if (auto Err = readSize(NumRegions))
    return Err;
  unsigned LineStart = 0;
  for (size_t I = 0; I < NumRegions; ++I) {
    Counter C;
    CounterMappingRegion::RegionKind Kind = CounterMappingRegion::CodeRegion;

    // Read the combined counter + region kind.
    uint64_t EncodedCounterAndRegion;
    if (auto Err = readIntMax(EncodedCounterAndRegion,
                              std::numeric_limits<unsigned>::max()))
      return Err;
    unsigned Tag = EncodedCounterAndRegion & Counter::EncodingTagMask;
    uint64_t ExpandedFileID = 0;
    if (Tag != Counter::Zero) {
      if (auto Err = decodeCounter(EncodedCounterAndRegion, C))
        return Err;
    } else {
      // Is it an expansion region?
      if (EncodedCounterAndRegion & EncodingExpansionRegionBit) {
        Kind = CounterMappingRegion::ExpansionRegion;
        ExpandedFileID = EncodedCounterAndRegion >>
                         Counter::EncodingCounterTagAndExpansionRegionTagBits;
        if (ExpandedFileID >= NumFileIDs)
          return error(instrprof_error::malformed);
      } else {
        switch (EncodedCounterAndRegion >>
                Counter::EncodingCounterTagAndExpansionRegionTagBits) {
        case CounterMappingRegion::CodeRegion:
          // Don't do anything when we have a code region with a zero counter.
          break;
        case CounterMappingRegion::SkippedRegion:
          Kind = CounterMappingRegion::SkippedRegion;
          break;
        default:
          return error(instrprof_error::malformed);
        }
      }
    }

    // Read the source range.
    uint64_t LineStartDelta, ColumnStart, NumLines, ColumnEnd;
    if (auto Err =
            readIntMax(LineStartDelta, std::numeric_limits<unsigned>::max()))
      return Err;
    if (auto Err = readULEB128(ColumnStart))
      return Err;
    if (ColumnStart > std::numeric_limits<unsigned>::max())
      return error(instrprof_error::malformed);
    if (auto Err = readIntMax(NumLines, std::numeric_limits<unsigned>::max()))
      return Err;
    if (auto Err = readIntMax(ColumnEnd, std::numeric_limits<unsigned>::max()))
      return Err;
    LineStart += LineStartDelta;
    // Adjust the column locations for the empty regions that are supposed to
    // cover whole lines. Those regions should be encoded with the
    // column range (1 -> std::numeric_limits<unsigned>::max()), but because
    // the encoded std::numeric_limits<unsigned>::max() is several bytes long,
    // we set the column range to (0 -> 0) to ensure that the column start and
    // column end take up one byte each.
    // The std::numeric_limits<unsigned>::max() is used to represent a column
    // position at the end of the line without knowing the length of that line.
    if (ColumnStart == 0 && ColumnEnd == 0) {
      ColumnStart = 1;
      ColumnEnd = std::numeric_limits<unsigned>::max();
    }

    DEBUG({
      dbgs() << "Counter in file " << InferredFileID << " " << LineStart << ":"
             << ColumnStart << " -> " << (LineStart + NumLines) << ":"
             << ColumnEnd << ", ";
      if (Kind == CounterMappingRegion::ExpansionRegion)
        dbgs() << "Expands to file " << ExpandedFileID;
      else
        CounterMappingContext(Expressions).dump(C, dbgs());
      dbgs() << "\n";
    });

    MappingRegions.push_back(CounterMappingRegion(
        C, InferredFileID, ExpandedFileID, LineStart, ColumnStart,
        LineStart + NumLines, ColumnEnd, Kind));
  }
  return success();
}

std::error_code RawCoverageMappingReader::read() {

  // Read the virtual file mapping.
  llvm::SmallVector<unsigned, 8> VirtualFileMapping;
  uint64_t NumFileMappings;
  if (auto Err = readSize(NumFileMappings))
    return Err;
  for (size_t I = 0; I < NumFileMappings; ++I) {
    uint64_t FilenameIndex;
    if (auto Err = readIntMax(FilenameIndex, TranslationUnitFilenames.size()))
      return Err;
    VirtualFileMapping.push_back(FilenameIndex);
  }

  // Construct the files using unique filenames and virtual file mapping.
  for (auto I : VirtualFileMapping) {
    Filenames.push_back(TranslationUnitFilenames[I]);
  }

  // Read the expressions.
  uint64_t NumExpressions;
  if (auto Err = readSize(NumExpressions))
    return Err;
  // Create an array of dummy expressions that get the proper counters
  // when the expressions are read, and the proper kinds when the counters
  // are decoded.
  Expressions.resize(
      NumExpressions,
      CounterExpression(CounterExpression::Subtract, Counter(), Counter()));
  for (size_t I = 0; I < NumExpressions; ++I) {
    if (auto Err = readCounter(Expressions[I].LHS))
      return Err;
    if (auto Err = readCounter(Expressions[I].RHS))
      return Err;
  }

  // Read the mapping regions sub-arrays.
  for (unsigned InferredFileID = 0, S = VirtualFileMapping.size();
       InferredFileID < S; ++InferredFileID) {
    if (auto Err = readMappingRegionsSubArray(MappingRegions, InferredFileID,
                                              VirtualFileMapping.size()))
      return Err;
  }

  // Set the counters for the expansion regions.
  // i.e. Counter of expansion region = counter of the first region
  // from the expanded file.
  // Perform multiple passes to correctly propagate the counters through
  // all the nested expansion regions.
  SmallVector<CounterMappingRegion *, 8> FileIDExpansionRegionMapping;
  FileIDExpansionRegionMapping.resize(VirtualFileMapping.size(), nullptr);
  for (unsigned Pass = 1, S = VirtualFileMapping.size(); Pass < S; ++Pass) {
    for (auto &R : MappingRegions) {
      if (R.Kind != CounterMappingRegion::ExpansionRegion)
        continue;
      assert(!FileIDExpansionRegionMapping[R.ExpandedFileID]);
      FileIDExpansionRegionMapping[R.ExpandedFileID] = &R;
    }
    for (auto &R : MappingRegions) {
      if (FileIDExpansionRegionMapping[R.FileID]) {
        FileIDExpansionRegionMapping[R.FileID]->Count = R.Count;
        FileIDExpansionRegionMapping[R.FileID] = nullptr;
      }
    }
  }

  return success();
}

namespace {
/// \brief The coverage mapping data for a single function.
/// It points to the function's name.
template <typename IntPtrT> struct CoverageMappingFunctionRecord {
  IntPtrT FunctionNamePtr;
  uint32_t FunctionNameSize;
  uint32_t CoverageMappingSize;
  uint64_t FunctionHash;
};

/// \brief The coverage mapping data for a single translation unit.
/// It points to the array of function coverage mapping records and the encoded
/// filenames array.
template <typename IntPtrT> struct CoverageMappingTURecord {
  uint32_t FunctionRecordsSize;
  uint32_t FilenamesSize;
  uint32_t CoverageMappingsSize;
  uint32_t Version;
};

/// \brief A helper structure to access the data from a section
/// in an object file.
struct SectionData {
  StringRef Data;
  uint64_t Address;

  std::error_code load(SectionRef &Section) {
    if (auto Err = Section.getContents(Data))
      return Err;
    Address = Section.getAddress();
    return instrprof_error::success;
  }

  std::error_code get(uint64_t Pointer, size_t Size, StringRef &Result) {
    if (Pointer < Address)
      return instrprof_error::malformed;
    auto Offset = Pointer - Address;
    if (Offset + Size > Data.size())
      return instrprof_error::malformed;
    Result = Data.substr(Pointer - Address, Size);
    return instrprof_error::success;
  }
};
}

template <typename T>
std::error_code readCoverageMappingData(
    SectionData &ProfileNames, StringRef Data,
    std::vector<BinaryCoverageReader::ProfileMappingRecord> &Records,
    std::vector<StringRef> &Filenames) {
  llvm::DenseSet<T> UniqueFunctionMappingData;

  // Read the records in the coverage data section.
  while (!Data.empty()) {
    if (Data.size() < sizeof(CoverageMappingTURecord<T>))
      return instrprof_error::malformed;
    auto TU = reinterpret_cast<const CoverageMappingTURecord<T> *>(Data.data());
    Data = Data.substr(sizeof(CoverageMappingTURecord<T>));
    switch (TU->Version) {
    case CoverageMappingVersion1:
      break;
    default:
      return instrprof_error::unsupported_version;
    }
    auto Version = CoverageMappingVersion(TU->Version);

    // Get the function records.
    auto FunctionRecords =
        reinterpret_cast<const CoverageMappingFunctionRecord<T> *>(Data.data());
    if (Data.size() <
        sizeof(CoverageMappingFunctionRecord<T>) * TU->FunctionRecordsSize)
      return instrprof_error::malformed;
    Data = Data.substr(sizeof(CoverageMappingFunctionRecord<T>) *
                       TU->FunctionRecordsSize);

    // Get the filenames.
    if (Data.size() < TU->FilenamesSize)
      return instrprof_error::malformed;
    auto RawFilenames = Data.substr(0, TU->FilenamesSize);
    Data = Data.substr(TU->FilenamesSize);
    size_t FilenamesBegin = Filenames.size();
    RawCoverageFilenamesReader Reader(RawFilenames, Filenames);
    if (auto Err = Reader.read())
      return Err;

    // Get the coverage mappings.
    if (Data.size() < TU->CoverageMappingsSize)
      return instrprof_error::malformed;
    auto CoverageMappings = Data.substr(0, TU->CoverageMappingsSize);
    Data = Data.substr(TU->CoverageMappingsSize);

    for (unsigned I = 0; I < TU->FunctionRecordsSize; ++I) {
      auto &MappingRecord = FunctionRecords[I];

      // Get the coverage mapping.
      if (CoverageMappings.size() < MappingRecord.CoverageMappingSize)
        return instrprof_error::malformed;
      auto Mapping =
          CoverageMappings.substr(0, MappingRecord.CoverageMappingSize);
      CoverageMappings =
          CoverageMappings.substr(MappingRecord.CoverageMappingSize);

      // Ignore this record if we already have a record that points to the same
      // function name.
      // This is useful to ignore the redundant records for the functions
      // with ODR linkage.
      if (!UniqueFunctionMappingData.insert(MappingRecord.FunctionNamePtr)
               .second)
        continue;
      StringRef FunctionName;
      if (auto Err =
              ProfileNames.get(MappingRecord.FunctionNamePtr,
                               MappingRecord.FunctionNameSize, FunctionName))
        return Err;
      Records.push_back(BinaryCoverageReader::ProfileMappingRecord(
          Version, FunctionName, MappingRecord.FunctionHash, Mapping,
          FilenamesBegin, Filenames.size() - FilenamesBegin));
    }
  }

  return instrprof_error::success;
}

static const char *TestingFormatMagic = "llvmcovmtestdata";

static std::error_code loadTestingFormat(StringRef Data,
                                         SectionData &ProfileNames,
                                         StringRef &CoverageMapping,
                                         uint8_t &BytesInAddress) {
  BytesInAddress = 8;

  Data = Data.substr(StringRef(TestingFormatMagic).size());
  if (Data.size() < 1)
    return instrprof_error::truncated;
  unsigned N = 0;
  auto ProfileNamesSize =
      decodeULEB128(reinterpret_cast<const uint8_t *>(Data.data()), &N);
  if (N > Data.size())
    return instrprof_error::malformed;
  Data = Data.substr(N);
  if (Data.size() < 1)
    return instrprof_error::truncated;
  N = 0;
  ProfileNames.Address =
      decodeULEB128(reinterpret_cast<const uint8_t *>(Data.data()), &N);
  if (N > Data.size())
    return instrprof_error::malformed;
  Data = Data.substr(N);
  if (Data.size() < ProfileNamesSize)
    return instrprof_error::malformed;
  ProfileNames.Data = Data.substr(0, ProfileNamesSize);
  CoverageMapping = Data.substr(ProfileNamesSize);
  return instrprof_error::success;
}

static std::error_code loadBinaryFormat(MemoryBufferRef ObjectBuffer,
                                        SectionData &ProfileNames,
                                        StringRef &CoverageMapping,
                                        uint8_t &BytesInAddress) {
  auto ObjectFileOrErr = object::ObjectFile::createObjectFile(ObjectBuffer);
  if (std::error_code EC = ObjectFileOrErr.getError())
    return EC;
  auto OF = std::move(ObjectFileOrErr.get());
  BytesInAddress = OF->getBytesInAddress();

  // Look for the sections that we are interested in.
  int FoundSectionCount = 0;
  SectionRef NamesSection, CoverageSection;
  for (const auto &Section : OF->sections()) {
    StringRef Name;
    if (auto Err = Section.getName(Name))
      return Err;
    if (Name == "__llvm_prf_names") {
      NamesSection = Section;
    } else if (Name == "__llvm_covmap") {
      CoverageSection = Section;
    } else
      continue;
    ++FoundSectionCount;
  }
  if (FoundSectionCount != 2)
    return instrprof_error::bad_header;

  // Get the contents of the given sections.
  if (std::error_code EC = CoverageSection.getContents(CoverageMapping))
    return EC;
  if (std::error_code EC = ProfileNames.load(NamesSection))
    return EC;

  return std::error_code();
}

ErrorOr<std::unique_ptr<BinaryCoverageReader>>
BinaryCoverageReader::create(std::unique_ptr<MemoryBuffer> &ObjectBuffer) {
  std::unique_ptr<BinaryCoverageReader> Reader(new BinaryCoverageReader());

  SectionData Profile;
  StringRef Coverage;
  uint8_t BytesInAddress;
  std::error_code EC;
  if (ObjectBuffer->getBuffer().startswith(TestingFormatMagic))
    // This is a special format used for testing.
    EC = loadTestingFormat(ObjectBuffer->getBuffer(), Profile, Coverage,
                           BytesInAddress);
  else
    EC = loadBinaryFormat(ObjectBuffer->getMemBufferRef(), Profile, Coverage,
                          BytesInAddress);
  if (EC)
    return EC;

  if (BytesInAddress == 4)
    EC = readCoverageMappingData<uint32_t>(
        Profile, Coverage, Reader->MappingRecords, Reader->Filenames);
  else if (BytesInAddress == 8)
    EC = readCoverageMappingData<uint64_t>(
        Profile, Coverage, Reader->MappingRecords, Reader->Filenames);
  else
    return instrprof_error::malformed;
  if (EC)
    return EC;
  return std::move(Reader);
}

std::error_code
BinaryCoverageReader::readNextRecord(CoverageMappingRecord &Record) {
  if (CurrentRecord >= MappingRecords.size())
    return instrprof_error::eof;

  FunctionsFilenames.clear();
  Expressions.clear();
  MappingRegions.clear();
  auto &R = MappingRecords[CurrentRecord];
  RawCoverageMappingReader Reader(
      R.CoverageMapping,
      makeArrayRef(Filenames).slice(R.FilenamesBegin, R.FilenamesSize),
      FunctionsFilenames, Expressions, MappingRegions);
  if (auto Err = Reader.read())
    return Err;

  Record.FunctionName = R.FunctionName;
  Record.FunctionHash = R.FunctionHash;
  Record.Filenames = FunctionsFilenames;
  Record.Expressions = Expressions;
  Record.MappingRegions = MappingRegions;

  ++CurrentRecord;
  return std::error_code();
}
