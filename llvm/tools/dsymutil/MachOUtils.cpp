//===-- MachOUtils.h - Mach-o specific helpers for dsymutil  --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MachOUtils.h"
#include "dsymutil.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace dsymutil {
namespace MachOUtils {

static bool runLipo(SmallVectorImpl<const char *> &Args) {
  auto Path = sys::findProgramByName("lipo");

  if (!Path) {
    errs() << "error: lipo: " << Path.getError().message() << "\n";
    return false;
  }

  std::string ErrMsg;
  int result =
      sys::ExecuteAndWait(*Path, Args.data(), nullptr, nullptr, 0, 0, &ErrMsg);
  if (result) {
    errs() << "error: lipo: " << ErrMsg << "\n";
    return false;
  }

  return true;
}

bool generateUniversalBinary(SmallVectorImpl<ArchAndFilename> &ArchFiles,
                             StringRef OutputFileName,
                             const LinkOptions &Options) {
  // No need to merge one file into a universal fat binary. First, try
  // to move it (rename) to the final location. If that fails because
  // of cross-device link issues then copy and delete.
  if (ArchFiles.size() == 1) {
    StringRef From(ArchFiles.front().Path);
    if (sys::fs::rename(From, OutputFileName)) {
      if (std::error_code EC = sys::fs::copy_file(From, OutputFileName)) {
        errs() << "error: while copying " << From << " to " << OutputFileName
               << ": " << EC.message() << "\n";
        return false;
      }
      sys::fs::remove(From);
    }
    return true;
  }

  SmallVector<const char *, 8> Args;
  Args.push_back("lipo");
  Args.push_back("-create");

  for (auto &Thin : ArchFiles)
    Args.push_back(Thin.Path.c_str());

  // Align segments to match dsymutil-classic alignment
  for (auto &Thin : ArchFiles) {
    Args.push_back("-segalign");
    Args.push_back(Thin.Arch.c_str());
    Args.push_back("20");
  }

  Args.push_back("-output");
  Args.push_back(OutputFileName.data());
  Args.push_back(nullptr);

  if (Options.Verbose) {
    outs() << "Running lipo\n";
    for (auto Arg : Args)
      outs() << ' ' << ((Arg == nullptr) ? "\n" : Arg);
  }

  return Options.NoOutput ? true : runLipo(Args);
}
}
}
}
