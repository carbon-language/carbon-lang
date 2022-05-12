//===-------------- llvm-remark-size-diff/RemarkSizeDiff.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Diffs instruction count and stack size remarks between two remark files.
///
/// This is intended for use by compiler developers who want to see how their
/// changes impact program code size.
///
/// TODO: Add structured output (JSON, or YAML, or something...)
///
//===----------------------------------------------------------------------===//

#include "llvm-c/Remarks.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Remarks/Remark.h"
#include "llvm/Remarks/RemarkParser.h"
#include "llvm/Remarks/RemarkSerializer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

enum ParserFormatOptions { yaml, bitstream };
static cl::OptionCategory SizeDiffCategory("llvm-remark-size-diff options");
static cl::opt<std::string> InputFileNameA(cl::Positional, cl::Required,
                                           cl::cat(SizeDiffCategory),
                                           cl::desc("remarks_a"));
static cl::opt<std::string> InputFileNameB(cl::Positional, cl::Required,
                                           cl::cat(SizeDiffCategory),
                                           cl::desc("remarks_b"));
static cl::opt<std::string> OutputFilename("o", cl::init("-"),
                                           cl::cat(SizeDiffCategory),
                                           cl::desc("Output"),
                                           cl::value_desc("file"));
static cl::opt<ParserFormatOptions>
    ParserFormat("parser", cl::cat(SizeDiffCategory), cl::init(bitstream),
                 cl::desc("Set the remark parser format:"),
                 cl::values(clEnumVal(yaml, "YAML format"),
                            clEnumVal(bitstream, "Bitstream format")));

/// Contains information from size remarks.
// This is a little nicer to read than a std::pair.
struct InstCountAndStackSize {
  int64_t InstCount = 0;
  int64_t StackSize = 0;
};

/// Represents which files a function appeared in.
enum FilesPresent { A, B, BOTH };

/// Contains the data from the remarks in file A and file B for some function.
/// E.g. instruction count, stack size...
struct FunctionDiff {
  /// Function name from the remark.
  std::string FuncName;
  // Idx 0 = A, Idx 1 = B.
  int64_t InstCount[2] = {0, 0};
  int64_t StackSize[2] = {0, 0};

  // Calculate diffs between the first and second files.
  int64_t getInstDiff() const { return InstCount[1] - InstCount[0]; }
  int64_t getStackDiff() const { return StackSize[1] - StackSize[0]; }

  // Accessors for the remarks from the first file.
  int64_t getInstCountA() const { return InstCount[0]; }
  int64_t getStackSizeA() const { return StackSize[0]; }

  // Accessors for the remarks from the second file.
  int64_t getInstCountB() const { return InstCount[1]; }
  int64_t getStackSizeB() const { return StackSize[1]; }

  /// \returns which files this function was present in.
  FilesPresent getFilesPresent() const {
    if (getInstCountA() == 0)
      return B;
    if (getInstCountB() == 0)
      return A;
    return BOTH;
  }

  FunctionDiff(StringRef FuncName, const InstCountAndStackSize &A,
               const InstCountAndStackSize &B)
      : FuncName(FuncName) {
    InstCount[0] = A.InstCount;
    InstCount[1] = B.InstCount;
    StackSize[0] = A.StackSize;
    StackSize[1] = B.StackSize;
  }
};

/// Organizes the diffs into 3 categories:
/// - Functions which only appeared in the first file
/// - Functions which only appeared in the second file
/// - Functions which appeared in both files
struct DiffsCategorizedByFilesPresent {
  /// Diffs for functions which only appeared in the first file.
  SmallVector<FunctionDiff> OnlyInA;

  /// Diffs for functions which only appeared in the second file.
  SmallVector<FunctionDiff> OnlyInB;

  /// Diffs for functions which appeared in both files.
  SmallVector<FunctionDiff> InBoth;

  /// Add a diff to the appropriate list.
  void addDiff(FunctionDiff &FD) {
    switch (FD.getFilesPresent()) {
    case A:
      OnlyInA.push_back(FD);
      break;
    case B:
      OnlyInB.push_back(FD);
      break;
    case BOTH:
      InBoth.push_back(FD);
      break;
    }
  }
};

static void printFunctionDiff(const FunctionDiff &FD, llvm::raw_ostream &OS) {
  // Describe which files the function had remarks in.
  FilesPresent FP = FD.getFilesPresent();
  const std::string &FuncName = FD.FuncName;
  const int64_t InstDiff = FD.getInstDiff();
  assert(InstDiff && "Shouldn't get functions with no size change?");
  const int64_t StackDiff = FD.getStackDiff();
  // Output an indicator denoting which files the function was present in.
  switch (FP) {
  case FilesPresent::A:
    OS << "-- ";
    break;
  case FilesPresent::B:
    OS << "++ ";
    break;
  case FilesPresent::BOTH:
    OS << "== ";
    break;
  }
  // Output an indicator denoting if a function changed in size.
  if (InstDiff > 0)
    OS << "> ";
  else
    OS << "< ";
  OS << FuncName << ", ";
  OS << InstDiff << " instrs, ";
  OS << StackDiff << " stack B";
  OS << "\n";
}

/// Print an item in the summary section.
///
/// \p TotalA - Total count of the metric in file A.
/// \p TotalB - Total count of the metric in file B.
/// \p Metric - Name of the metric we want to print (e.g. instruction
/// count).
/// \p OS - The output stream.
static void printSummaryItem(int64_t TotalA, int64_t TotalB, StringRef Metric,
                             llvm::raw_ostream &OS) {
  OS << "  " << Metric << ": ";
  int64_t TotalDiff = TotalB - TotalA;
  if (TotalDiff == 0) {
    OS << "None\n";
    return;
  }
  OS << TotalDiff << " (" << formatv("{0:p}", TotalDiff / (double)TotalA)
     << ")\n";
}

/// Print all contents of \p Diff and a high-level summary of the differences.
static void printDiffsCategorizedByFilesPresent(
    DiffsCategorizedByFilesPresent &DiffsByFilesPresent,
    llvm::raw_ostream &OS) {
  int64_t InstrsA = 0;
  int64_t InstrsB = 0;
  int64_t StackA = 0;
  int64_t StackB = 0;
  // Helper lambda to sort + print a list of diffs.
  auto PrintDiffList = [&](SmallVector<FunctionDiff> &FunctionDiffList) {
    if (FunctionDiffList.empty())
      return;
    stable_sort(FunctionDiffList,
                [](const FunctionDiff &LHS, const FunctionDiff &RHS) {
                  return LHS.getInstDiff() < RHS.getInstDiff();
                });
    for (const auto &FuncDiff : FunctionDiffList) {
      // If there is a difference in instruction count, then print out info for
      // the function.
      if (FuncDiff.getInstDiff())
        printFunctionDiff(FuncDiff, OS);
      InstrsA += FuncDiff.getInstCountA();
      InstrsB += FuncDiff.getInstCountB();
      StackA += FuncDiff.getStackSizeA();
      StackB += FuncDiff.getStackSizeB();
    }
  };
  PrintDiffList(DiffsByFilesPresent.OnlyInA);
  PrintDiffList(DiffsByFilesPresent.OnlyInB);
  PrintDiffList(DiffsByFilesPresent.InBoth);
  OS << "\n### Summary ###\n";
  OS << "Total change: \n";
  printSummaryItem(InstrsA, InstrsB, "instruction count", OS);
  printSummaryItem(StackA, StackB, "stack byte usage", OS);
}

/// Collects an expected integer value from a given argument index in a remark.
///
/// \p Remark - The remark.
/// \p ArgIdx - The index where the integer value should be found.
/// \p ExpectedKeyName - The expected key name for the index
/// (e.g. "InstructionCount")
///
/// \returns the integer value at the index if it exists, and the key-value pair
/// is what is expected. Otherwise, returns an Error.
static Expected<int64_t> getIntValFromKey(const remarks::Remark &Remark,
                                          unsigned ArgIdx,
                                          StringRef ExpectedKeyName) {
  auto KeyName = Remark.Args[ArgIdx].Key;
  if (KeyName != ExpectedKeyName)
    return createStringError(
        inconvertibleErrorCode(),
        Twine("Unexpected key at argument index " + std::to_string(ArgIdx) +
              ": Expected '" + ExpectedKeyName + "', got '" + KeyName + "'"));
  long long Val;
  auto ValStr = Remark.Args[ArgIdx].Val;
  if (getAsSignedInteger(ValStr, 0, Val))
    return createStringError(
        inconvertibleErrorCode(),
        Twine("Could not convert string to signed integer: " + ValStr));
  return static_cast<int64_t>(Val);
}

/// Collects relevant size information from \p Remark if it is an size-related
/// remark of some kind (e.g. instruction count). Otherwise records nothing.
///
/// \p Remark - The remark.
/// \p FuncNameToSizeInfo - Maps function names to relevant size info.
/// \p NumInstCountRemarksParsed - Keeps track of the number of instruction
/// count remarks parsed. We need at least 1 in both files to produce a diff.
static Error processRemark(const remarks::Remark &Remark,
                           StringMap<InstCountAndStackSize> &FuncNameToSizeInfo,
                           unsigned &NumInstCountRemarksParsed) {
  const auto &RemarkName = Remark.RemarkName;
  const auto &PassName = Remark.PassName;
  // Collect remarks which contain the number of instructions in a function.
  if (PassName == "asm-printer" && RemarkName == "InstructionCount") {
    // Expecting the 0-th argument to have the key "NumInstructions" and an
    // integer value.
    auto MaybeInstCount =
        getIntValFromKey(Remark, /*ArgIdx = */ 0, "NumInstructions");
    if (!MaybeInstCount)
      return MaybeInstCount.takeError();
    FuncNameToSizeInfo[Remark.FunctionName].InstCount = *MaybeInstCount;
    ++NumInstCountRemarksParsed;
  }
  // Collect remarks which contain the stack size of a function.
  else if (PassName == "prologepilog" && RemarkName == "StackSize") {
    // Expecting the 0-th argument to have the key "NumStackBytes" and an
    // integer value.
    auto MaybeStackSize =
        getIntValFromKey(Remark, /*ArgIdx = */ 0, "NumStackBytes");
    if (!MaybeStackSize)
      return MaybeStackSize.takeError();
    FuncNameToSizeInfo[Remark.FunctionName].StackSize = *MaybeStackSize;
  }
  // Either we collected a remark, or it's something we don't care about. In
  // both cases, this is a success.
  return Error::success();
}

/// Process all of the size-related remarks in a file.
///
/// \param[in] InputFileName - Name of file to read from.
/// \param[in, out] FuncNameToSizeInfo - Maps function names to relevant
/// size info.
static Error readFileAndProcessRemarks(
    StringRef InputFileName,
    StringMap<InstCountAndStackSize> &FuncNameToSizeInfo) {
  auto Buf = MemoryBuffer::getFile(InputFileName);
  if (auto EC = Buf.getError())
    return createStringError(
        EC, Twine("Cannot open file '" + InputFileName + "': " + EC.message()));
  auto MaybeParser = remarks::createRemarkParserFromMeta(
      ParserFormat == bitstream ? remarks::Format::Bitstream
                                : remarks::Format::YAML,
      (*Buf)->getBuffer());
  if (!MaybeParser)
    return MaybeParser.takeError();
  auto &Parser = **MaybeParser;
  auto MaybeRemark = Parser.next();
  unsigned NumInstCountRemarksParsed = 0;
  for (; MaybeRemark; MaybeRemark = Parser.next()) {
    if (auto E = processRemark(**MaybeRemark, FuncNameToSizeInfo,
                               NumInstCountRemarksParsed))
      return E;
  }
  auto E = MaybeRemark.takeError();
  if (!E.isA<remarks::EndOfFileError>())
    return E;
  consumeError(std::move(E));
  // We need at least one instruction count remark in each file to produce a
  // meaningful diff.
  if (NumInstCountRemarksParsed == 0)
    return createStringError(
        inconvertibleErrorCode(),
        "File '" + InputFileName +
            "' did not contain any instruction-count remarks!");
  return Error::success();
}

/// Wrapper function for readFileAndProcessRemarks which handles errors.
///
/// \param[in] InputFileName - Name of file to read from.
/// \param[out] FuncNameToSizeInfo - Populated with information from size
/// remarks in the input file.
///
/// \returns true if readFileAndProcessRemarks returned no errors. False
/// otherwise.
static bool tryReadFileAndProcessRemarks(
    StringRef InputFileName,
    StringMap<InstCountAndStackSize> &FuncNameToSizeInfo) {
  if (Error E = readFileAndProcessRemarks(InputFileName, FuncNameToSizeInfo)) {
    handleAllErrors(std::move(E), [&](const ErrorInfoBase &PE) {
      PE.log(WithColor::error());
      errs() << '\n';
    });
    return false;
  }
  return true;
}

/// Populates \p FuncDiffs with the difference between \p
/// FuncNameToSizeInfoA and \p FuncNameToSizeInfoB.
///
/// \param[in] FuncNameToSizeInfoA - Size info collected from the first
/// remarks file.
/// \param[in] FuncNameToSizeInfoB - Size info collected from
/// the second remarks file.
/// \param[out] DiffsByFilesPresent - Filled with the diff between \p
/// FuncNameToSizeInfoA and \p FuncNameToSizeInfoB.
static void
computeDiff(const StringMap<InstCountAndStackSize> &FuncNameToSizeInfoA,
            const StringMap<InstCountAndStackSize> &FuncNameToSizeInfoB,
            DiffsCategorizedByFilesPresent &DiffsByFilesPresent) {
  SmallSet<std::string, 10> FuncNames;
  for (const auto &FuncName : FuncNameToSizeInfoA.keys())
    FuncNames.insert(FuncName.str());
  for (const auto &FuncName : FuncNameToSizeInfoB.keys())
    FuncNames.insert(FuncName.str());
  for (const std::string &FuncName : FuncNames) {
    const auto &SizeInfoA = FuncNameToSizeInfoA.lookup(FuncName);
    const auto &SizeInfoB = FuncNameToSizeInfoB.lookup(FuncName);
    FunctionDiff FuncDiff(FuncName, SizeInfoA, SizeInfoB);
    DiffsByFilesPresent.addDiff(FuncDiff);
  }
}

/// Attempt to get the output stream for writing the diff.
static ErrorOr<std::unique_ptr<ToolOutputFile>> getOutputStream() {
  if (OutputFilename == "")
    OutputFilename = "-";
  std::error_code EC;
  auto Out = std::make_unique<ToolOutputFile>(OutputFilename, EC,
                                              sys::fs::OF_TextWithCRLF);
  if (!EC)
    return std::move(Out);
  return EC;
}

/// Output all diffs in \p DiffsByFilesPresent.
/// \returns Error::success() on success, and an Error otherwise.
static Error
outputAllDiffs(DiffsCategorizedByFilesPresent &DiffsByFilesPresent) {
  auto MaybeOF = getOutputStream();
  if (std::error_code EC = MaybeOF.getError())
    return errorCodeToError(EC);
  std::unique_ptr<ToolOutputFile> OF = std::move(*MaybeOF);
  printDiffsCategorizedByFilesPresent(DiffsByFilesPresent, OF->os());
  OF->keep();
  return Error::success();
}

/// Boolean wrapper for outputDiff which handles errors.
static bool
tryOutputAllDiffs(DiffsCategorizedByFilesPresent &DiffsByFilesPresent) {
  if (Error E = outputAllDiffs(DiffsByFilesPresent)) {
    handleAllErrors(std::move(E), [&](const ErrorInfoBase &PE) {
      PE.log(WithColor::error());
      errs() << '\n';
    });
    return false;
  }
  return true;
}

int main(int argc, const char **argv) {
  InitLLVM X(argc, argv);
  cl::HideUnrelatedOptions(SizeDiffCategory);
  cl::ParseCommandLineOptions(argc, argv,
                              "Diff instruction count and stack size remarks "
                              "between two remark files.\n");
  StringMap<InstCountAndStackSize> FuncNameToSizeInfoA;
  StringMap<InstCountAndStackSize> FuncNameToSizeInfoB;
  if (!tryReadFileAndProcessRemarks(InputFileNameA, FuncNameToSizeInfoA) ||
      !tryReadFileAndProcessRemarks(InputFileNameB, FuncNameToSizeInfoB))
    return 1;
  DiffsCategorizedByFilesPresent DiffsByFilesPresent;
  computeDiff(FuncNameToSizeInfoA, FuncNameToSizeInfoB, DiffsByFilesPresent);
  if (!tryOutputAllDiffs(DiffsByFilesPresent))
    return 1;
}
