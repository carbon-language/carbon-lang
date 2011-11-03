//===-- llvm-config.cpp - LLVM project configuration utility --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tool encapsulates information about an LLVM project configuration for
// use by other project's build environments (to determine installed path,
// available features, required libraries, etc.).
//
// Note that although this tool *may* be used by some parts of LLVM's build
// itself (i.e., the Makefiles use it to compute required libraries when linking
// tools), this tool is primarily designed to support external projects.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/config.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <vector>

using namespace llvm;

// FIXME: Need to get various bits of build time information.
const char LLVM_SRC_ROOT[] = "FIXME";
const char LLVM_OBJ_ROOT[] = "FIXME";
const char LLVM_CPPFLAGS[] = "FIXME";
const char LLVM_CFLAGS[] = "FIXME";
const char LLVM_LDFLAGS[] = "FIXME";
const char LLVM_CXXFLAGS[] = "FIXME";
const char LLVM_BUILDMODE[] = "FIXME";
const char LLVM_SYSTEM_LIBS[] = "FIXME";

// FIXME: Include component table.
struct AvailableComponent {
  const char *Name;
} AvailableComponents[1] = {};
unsigned NumAvailableComponents = 0;

void ComputeLibsForComponents(const std::vector<StringRef> &Components,
                              std::vector<StringRef> &RequiredLibs) {
  // FIXME: Implement.
  RequiredLibs = Components;
}

/* *** */

void usage() {
  errs() << "\
usage: llvm-config <OPTION>... [<COMPONENT>...]\n\
\n\
Get various configuration information needed to compile programs which use\n\
LLVM.  Typically called from 'configure' scripts.  Examples:\n\
  llvm-config --cxxflags\n\
  llvm-config --ldflags\n\
  llvm-config --libs engine bcreader scalaropts\n\
\n\
Options:\n\
  --version         Print LLVM version.\n\
  --prefix          Print the installation prefix.\n\
  --src-root        Print the source root LLVM was built from.\n\
  --obj-root        Print the object root used to build LLVM.\n\
  --bindir          Directory containing LLVM executables.\n\
  --includedir      Directory containing LLVM headers.\n\
  --libdir          Directory containing LLVM libraries.\n\
  --cppflags        C preprocessor flags for files that include LLVM headers.\n\
  --cflags          C compiler flags for files that include LLVM headers.\n\
  --cxxflags        C++ compiler flags for files that include LLVM headers.\n\
  --ldflags         Print Linker flags.\n\
  --libs            Libraries needed to link against LLVM components.\n\
  --libnames        Bare library names for in-tree builds.\n\
  --libfiles        Fully qualified library filenames for makefile depends.\n\
  --components      List of all possible components.\n\
  --targets-built   List of all targets currently built.\n\
  --host-target     Target triple used to configure LLVM.\n\
  --build-mode      Print build mode of LLVM tree (e.g. Debug or Release).\n\
Typical components:\n\
  all               All LLVM libraries (default).\n\
  backend           Either a native backend or the C backend.\n\
  engine            Either a native JIT or a bitcode interpreter.\n";
  exit(1);
}

llvm::sys::Path GetExecutablePath(const char *Argv0) {
  // This just needs to be some symbol in the binary; C++ doesn't
  // allow taking the address of ::main however.
  void *P = (void*) (intptr_t) GetExecutablePath;
  return llvm::sys::Path::GetMainExecutable(Argv0, P);
}

int main(int argc, char **argv) {
  std::vector<StringRef> Components;
  bool PrintLibs = false, PrintLibNames = false, PrintLibFiles = false;
  bool HasAnyOption = false;

  // llvm-config is designed to support being run both from a development tree
  // and from an installed path. We try and auto-detect which case we are in so
  // that we can report the correct information when run from a development
  // tree.
  bool IsInDevelopmentTree, DevelopmentTreeLayoutIsCMakeStyle;
  llvm::SmallString<256> CurrentPath(GetExecutablePath(argv[0]).str());
  std::string CurrentExecPrefix;

  // Create an absolute path, and pop up one directory (we expect to be inside a
  // bin dir).
  sys::fs::make_absolute(CurrentPath);
  CurrentExecPrefix = sys::path::parent_path(
    sys::path::parent_path(CurrentPath)).str();

  // Check to see if we are inside a development tree by comparing to possible
  // locations (prefix style or CMake style). This could be wrong in the face of
  // symbolic links, but is good enough.
  if (CurrentExecPrefix == std::string(LLVM_OBJ_ROOT) + "/" + LLVM_BUILDMODE) {
    IsInDevelopmentTree = true;
    DevelopmentTreeLayoutIsCMakeStyle = false;
  } else if (CurrentExecPrefix == std::string(LLVM_OBJ_ROOT) + "/bin") {
    IsInDevelopmentTree = true;
    DevelopmentTreeLayoutIsCMakeStyle = true;
  } else {
    IsInDevelopmentTree = false;
  }

  // Compute various directory locations based on the derived location
  // information.
  std::string ActivePrefix, ActiveBinDir, ActiveIncludeDir, ActiveLibDir;
  std::string ActiveIncludeOption;
  if (IsInDevelopmentTree) {
    ActivePrefix = CurrentExecPrefix;

    // CMake organizes the products differently than a normal prefix style
    // layout.
    if (DevelopmentTreeLayoutIsCMakeStyle) {
      ActiveIncludeDir = std::string(LLVM_OBJ_ROOT) + "/include";
      ActiveBinDir = std::string(LLVM_OBJ_ROOT) + "/bin/" + LLVM_BUILDMODE;
      ActiveLibDir = std::string(LLVM_OBJ_ROOT) + "/lib/" + LLVM_BUILDMODE;
    } else {
        ActiveIncludeDir = std::string(LLVM_OBJ_ROOT) + "/include";
      ActiveBinDir = std::string(LLVM_OBJ_ROOT) + "/" + LLVM_BUILDMODE + "/bin";
      ActiveLibDir = std::string(LLVM_OBJ_ROOT) + "/" + LLVM_BUILDMODE + "/lib";
    }

    // We need to include files from both the source and object trees.
    ActiveIncludeOption = ("-I" + ActiveIncludeDir + " " +
                           "-I" + LLVM_OBJ_ROOT + "/include");
  } else {
    ActivePrefix = CurrentExecPrefix;
    ActiveIncludeDir = ActivePrefix + "/include";
    ActiveBinDir = ActivePrefix + "/bin";
    ActiveLibDir = ActivePrefix + "/lib";
    ActiveIncludeOption = "-I" + ActiveIncludeDir;
  }

  raw_ostream &OS = outs();
  for (int i = 1; i != argc; ++i) {
    StringRef Arg = argv[i];

    if (Arg.startswith("-")) {
      HasAnyOption = true;
      if (Arg == "--version") {
        OS << PACKAGE_VERSION << '\n';
      } else if (Arg == "--prefix") {
        OS << ActivePrefix << '\n';
      } else if (Arg == "--bindir") {
        OS << ActiveBinDir << '\n';
      } else if (Arg == "--includedir") {
        OS << ActiveIncludeDir << '\n';
      } else if (Arg == "--libdir") {
        OS << ActiveLibDir << '\n';
      } else if (Arg == "--cppflags") {
        OS << ActiveIncludeOption << ' ' << LLVM_CPPFLAGS << '\n';
      } else if (Arg == "--cflags") {
        OS << ActiveIncludeOption << ' ' << LLVM_CFLAGS << '\n';
      } else if (Arg == "--cxxflags") {
        OS << ActiveIncludeOption << ' ' << LLVM_CXXFLAGS << '\n';
      } else if (Arg == "--ldflags") {
        OS << "-L" << ActiveLibDir << ' ' << LLVM_LDFLAGS
           << ' ' << LLVM_SYSTEM_LIBS << '\n';
      } else if (Arg == "--libs") {
        PrintLibs = true;
      } else if (Arg == "--libnames") {
        PrintLibNames = true;
      } else if (Arg == "--libfiles") {
        PrintLibFiles = true;
      } else if (Arg == "--components") {
        for (unsigned j = 0; j != NumAvailableComponents; ++j) {
          if (j)
            OS << ' ';
          OS << AvailableComponents[j].Name;
        }
        OS << '\n';
      } else if (Arg == "--targets-built") {
        bool First = true;
        for (TargetRegistry::iterator I = TargetRegistry::begin(),
               E = TargetRegistry::end(); I != E; First = false, ++I) {
          if (!First)
            OS << ' ';
          OS << I->getName();
        }
        OS << '\n';
      } else if (Arg == "--host-target") {
        OS << LLVM_DEFAULT_TARGET_TRIPLE << '\n';
      } else if (Arg == "--build-mode") {
        OS << LLVM_BUILDMODE << '\n';
      } else if (Arg == "--obj-root") {
        OS << LLVM_OBJ_ROOT << '\n';
      } else if (Arg == "--src-root") {
        OS << LLVM_SRC_ROOT << '\n';
      } else {
        usage();
      }
    } else {
      Components.push_back(Arg);
    }
  }

  if (!HasAnyOption)
    usage();

  if (PrintLibs || PrintLibNames || PrintLibFiles) {
    // Construct the list of all the required libraries.
    std::vector<StringRef> RequiredLibs;
    ComputeLibsForComponents(Components, RequiredLibs);

    for (unsigned i = 0, e = RequiredLibs.size(); i != e; ++i) {
      StringRef Lib = RequiredLibs[i];
      if (i)
        OS << ' ';

      if (PrintLibs) {
        OS << Lib;
      } else if (PrintLibFiles) {
        OS << ActiveLibDir << Lib;
      } else if (PrintLibNames) {
        // If this is a typical library name, include it using -l.
        if (Lib.startswith("lib") && Lib.endswith(".a")) {
          OS << "-l" << Lib.slice(3, Lib.size()-2);
          continue;
        }
        
        // Otherwise, print the full path.
        OS << ActiveLibDir << Lib;
      }
    }
  } else if (!Components.empty()) {
    errs() << "llvm-config: error: components given, but unused\n\n";
    usage();
  }

  return 0;
}
