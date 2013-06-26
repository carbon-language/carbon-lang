//===-- cpp11-migrate/Cpp11Migrate.cpp - Main file C++11 migration tool ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides implementations for performance measuring helpers.
///
//===----------------------------------------------------------------------===//

#include "PerfSupport.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Path.h"

void collectSourcePerfData(const Transform &T, SourcePerfData &Data) {
  for (Transform::TimingVec::const_iterator I = T.timing_begin(),
                                            E = T.timing_end();
       I != E; ++I) {
    SourcePerfData::iterator DataI = Data.insert(
        SourcePerfData::value_type(I->first, std::vector<PerfItem>())).first;
    DataI->second
        .push_back(PerfItem(T.getName(), I->second.getProcessTime() * 1000.0));
  }
}

void writePerfDataJSON(
    const llvm::StringRef DirectoryName,
    const SourcePerfData &TimingResults) {
  // Create directory path if it doesn't exist
  llvm::sys::fs::create_directories(DirectoryName);

  // Get PID and current time.
  // FIXME: id_type on Windows is NOT a process id despite the function name.
  // Need to call GetProcessId() providing it what get_id() returns. For now
  // disabling PID-based file names until this is fixed properly.
  //llvm::sys::self_process *SP = llvm::sys::process::get_self();
  //id_type Pid = SP->get_id();
  unsigned Pid = 0;
  llvm::TimeRecord T = llvm::TimeRecord::getCurrentTime();

  std::string FileName;
  llvm::raw_string_ostream SS(FileName);
  SS << DirectoryName << "/" << static_cast<int>(T.getWallTime()) << "_" << Pid
     << ".json";

  std::string ErrorInfo;
  llvm::raw_fd_ostream FileStream(SS.str().c_str(), ErrorInfo);
  FileStream << "{\n";
  FileStream << "  \"Sources\" : [\n";
  for (SourcePerfData::const_iterator I = TimingResults.begin(),
                                      E = TimingResults.end();
       I != E; ++I) {
    // Terminate the last source with a comma before continuing to the next one.
    if (I != TimingResults.begin())
      FileStream << ",\n";

    FileStream << "    {\n";
    FileStream << "      \"Source \" : \"" << I->first << "\",\n";
    FileStream << "      \"Data\" : [\n";
    for (std::vector<PerfItem>::const_iterator IE = I->second.begin(),
                                               EE = I->second.end();
         IE != EE; ++IE) {
      // Terminate the last perf item with a comma before continuing to the next
      // one.
      if (IE != I->second.begin())
        FileStream << ",\n";

      FileStream << "        {\n";
      FileStream << "          \"TimerId\" : \"" << IE->Label << "\",\n";
      FileStream << "          \"Time\" : " << llvm::format("%.2f", IE->Duration)
                 << "\n";

      FileStream << "        }";

    }
    FileStream << "\n      ]\n";
    FileStream << "    }";
  }
  FileStream << "\n  ]\n";
  FileStream << "}";
}

void dumpPerfData(const SourcePerfData &Data) {
  for (SourcePerfData::const_iterator I = Data.begin(), E = Data.end(); I != E;
       ++I) {
    llvm::errs() << I->first << ":\n";
    for (std::vector<PerfItem>::const_iterator VecI = I->second.begin(),
                                               VecE = I->second.end();
         VecI != VecE; ++VecI) {
      llvm::errs() << "  " << VecI->Label << ": "
                   << llvm::format("%.1f", VecI->Duration) << "ms\n";
    }
  }
}
