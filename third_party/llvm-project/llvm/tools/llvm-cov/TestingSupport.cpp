//===- TestingSupport.cpp - Convert objects files into test files --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ObjectFile.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <system_error>

using namespace llvm;
using namespace object;

int convertForTestingMain(int argc, const char *argv[]) {
  cl::opt<std::string> InputSourceFile(cl::Positional, cl::Required,
                                       cl::desc("<Source file>"));

  cl::opt<std::string> OutputFilename(
      "o", cl::Required,
      cl::desc(
          "File with the profile data obtained after an instrumented run"));

  cl::ParseCommandLineOptions(argc, argv, "LLVM code coverage tool\n");

  auto ObjErr = llvm::object::ObjectFile::createObjectFile(InputSourceFile);
  if (!ObjErr) {
    std::string Buf;
    raw_string_ostream OS(Buf);
    logAllUnhandledErrors(ObjErr.takeError(), OS);
    OS.flush();
    errs() << "error: " << Buf;
    return 1;
  }
  ObjectFile *OF = ObjErr.get().getBinary();
  auto BytesInAddress = OF->getBytesInAddress();
  if (BytesInAddress != 8) {
    errs() << "error: 64 bit binary expected\n";
    return 1;
  }

  // Look for the sections that we are interested in.
  int FoundSectionCount = 0;
  SectionRef ProfileNames, CoverageMapping, CoverageRecords;
  auto ObjFormat = OF->getTripleObjectFormat();
  for (const auto &Section : OF->sections()) {
    StringRef Name;
    if (Expected<StringRef> NameOrErr = Section.getName()) {
      Name = *NameOrErr;
    } else {
      consumeError(NameOrErr.takeError());
      return 1;
    }

    if (Name == llvm::getInstrProfSectionName(IPSK_name, ObjFormat,
                                              /*AddSegmentInfo=*/false)) {
      ProfileNames = Section;
    } else if (Name == llvm::getInstrProfSectionName(
                           IPSK_covmap, ObjFormat, /*AddSegmentInfo=*/false)) {
      CoverageMapping = Section;
    } else if (Name == llvm::getInstrProfSectionName(
                           IPSK_covfun, ObjFormat, /*AddSegmentInfo=*/false)) {
      CoverageRecords = Section;
    } else
      continue;
    ++FoundSectionCount;
  }
  if (FoundSectionCount != 3)
    return 1;

  // Get the contents of the given sections.
  uint64_t ProfileNamesAddress = ProfileNames.getAddress();
  StringRef CoverageMappingData;
  StringRef CoverageRecordsData;
  StringRef ProfileNamesData;
  if (Expected<StringRef> E = CoverageMapping.getContents())
    CoverageMappingData = *E;
  else {
    consumeError(E.takeError());
    return 1;
  }
  if (Expected<StringRef> E = CoverageRecords.getContents())
    CoverageRecordsData = *E;
  else {
    consumeError(E.takeError());
    return 1;
  }
  if (Expected<StringRef> E = ProfileNames.getContents())
    ProfileNamesData = *E;
  else {
    consumeError(E.takeError());
    return 1;
  }

  int FD;
  if (auto Err = sys::fs::openFileForWrite(OutputFilename, FD)) {
    errs() << "error: " << Err.message() << "\n";
    return 1;
  }

  raw_fd_ostream OS(FD, true);
  OS << "llvmcovmtestdata";
  encodeULEB128(ProfileNamesData.size(), OS);
  encodeULEB128(ProfileNamesAddress, OS);
  OS << ProfileNamesData;
  // Coverage mapping data is expected to have an alignment of 8.
  for (unsigned Pad = offsetToAlignment(OS.tell(), Align(8)); Pad; --Pad)
    OS.write(uint8_t(0));
  OS << CoverageMappingData;
  // Coverage records data is expected to have an alignment of 8.
  for (unsigned Pad = offsetToAlignment(OS.tell(), Align(8)); Pad; --Pad)
    OS.write(uint8_t(0));
  OS << CoverageRecordsData;

  return 0;
}
