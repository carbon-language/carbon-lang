//===- llvm-link.cpp - Low-level LLVM linker ------------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This utility may be invoked in the following manner:
//  llvm-link a.bc b.bc c.bc -o x.bc
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Bytecode/Writer.h"
#include "llvm/Support/Linker.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/System/Signals.h"
#include "llvm/System/Path.h"
#include "llvm/ADT/SetVector.h"
#include <fstream>
#include <iostream>
#include <memory>

using namespace llvm;

static cl::list<std::string>
InputFilenames(cl::Positional, cl::OneOrMore,
               cl::desc("<input bytecode files>"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"), cl::init("-"),
               cl::value_desc("filename"));

static cl::opt<bool> Force("f", cl::desc("Overwrite output files"));

static cl::opt<bool>
Verbose("v", cl::desc("Print information about actions taken"));

static cl::opt<bool>
DumpAsm("d", cl::desc("Print assembly as linked"), cl::Hidden);

static cl::list<std::string>
LibPaths("L", cl::desc("Specify a library search path"), cl::ZeroOrMore,
         cl::value_desc("directory"), cl::Prefix);

static cl::list<std::string>
Libraries("l", cl::desc("Specify library names to link with"), cl::ZeroOrMore,
          cl::Prefix, cl::value_desc("library name"));

// GetModule - This function is just factored out of the functions below
static inline Module* GetModule(const sys::Path& Filename) {
  if (Verbose) std::cerr << "Loading '" << Filename.c_str() << "'\n";
  std::string ErrorMessage;
  if (Filename.exists()) {
    Module* Result = ParseBytecodeFile(Filename.get(), &ErrorMessage);
    if (Result) return Result;   // Load successful!

    if (Verbose) {
      std::cerr << "Error opening bytecode file: '" << Filename.c_str() << "'";
      if (ErrorMessage.size()) std::cerr << ": " << ErrorMessage;
      std::cerr << "\n";
    }
  } else {
    std::cerr << "Bytecode file: '" << Filename.c_str() 
              << "' does not exist.\n";
  }
  return 0;
}

// LoadFile - Read the specified bytecode file in and return it.  This routine
// searches the link path for the specified file to try to find it...
//
static inline std::auto_ptr<Module> LoadFile(const std::string &FN) {
  sys::Path Filename;
  if (!Filename.set_file(FN)) {
    std::cerr << "Invalid file name: '" << Filename.c_str() << "'\n";
    return std::auto_ptr<Module>();
  }

  if (Module* Result = GetModule(Filename)) 
    return std::auto_ptr<Module>(Result);

  bool FoundAFile = false;

  for (unsigned i = 0; i < LibPaths.size(); i++) {
    if (!Filename.set_directory(LibPaths[i])) {
      std::cerr << "Invalid library path: '" << LibPaths[i] << "'\n";
    } else if (!Filename.append_file(FN)) {
      std::cerr << "Invalid library path: '" << LibPaths[i]
                << "/" << FN.c_str() << "'\n";
    } else if (Filename.exists()) {
      FoundAFile = true;
      if (Module *Result = GetModule(Filename))
        return std::auto_ptr<Module>(Result);   // Load successful!
    }
  }

  if (FoundAFile)
    std::cerr << "Bytecode file '" << FN << "' corrupt!  "
              << "Use 'llvm-link -v ...' for more info.\n";
  else
    std::cerr << "Could not locate bytecode file: '" << FN << "'\n";
  return std::auto_ptr<Module>();
}

sys::Path GetPathForLinkageItem(const std::string& link_item,
                                const std::string& dir) {
  sys::Path fullpath;
  fullpath.set_directory(dir);

  // Try *.o
  fullpath.append_file(link_item);
  fullpath.append_suffix("o");
  if (fullpath.readable()) 
    return fullpath;

  // Try *.bc
  fullpath.elide_suffix();
  fullpath.append_suffix("bc");
  if (fullpath.readable()) 
    return fullpath;

  // Try *.so
  fullpath.elide_suffix();
  fullpath.append_suffix(sys::Path::GetDLLSuffix());
  if (fullpath.readable())
    return fullpath;

  // Try lib*.a
  fullpath.set_directory(dir);
  fullpath.append_file(std::string("lib") + link_item);
  fullpath.append_suffix("a");
  if (fullpath.readable())
    return fullpath;

  // Didn't find one.
  fullpath.clear();
  return fullpath;
}

static inline bool LoadLibrary(const std::string &FN, Module*& Result) {
  Result = 0;
  sys::Path Filename;
  if (!Filename.set_file(FN)) {
    return false;
  }

  if (Filename.readable() && Filename.is_bytecode_file()) {
    if (Result = GetModule(Filename))
      return true;
  }

  bool foundAFile = false;

  for (unsigned I = 0; I < LibPaths.size(); I++) {
    sys::Path path = GetPathForLinkageItem(FN,LibPaths[I]);
    if (!path.is_empty()) {
      if (path.is_bytecode_file()) {
        if (Result = GetModule(path)) {
          return true;
        } else {
          // We found file but its not a valid bytecode file so we 
          // return false and leave Result null.
          return false;
        }
      } else {
        // We found a file, but its not a bytecode file so we return
        // false and leave Result null.
        return false;
      }
    }
  }

  // We didn't find a file so we leave Result null and return
  // false to indicate that the library should be just left in the
  // emitted module as resolvable at runtime.
  return false;
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm linker\n");
  sys::PrintStackTraceOnErrorSignal();
  assert(InputFilenames.size() > 0 && "OneOrMore is not working");

  unsigned BaseArg = 0;
  std::string ErrorMessage;

  std::auto_ptr<Module> Composite(LoadFile(InputFilenames[BaseArg]));
  if (Composite.get() == 0) return 1;

  for (unsigned i = BaseArg+1; i < InputFilenames.size(); ++i) {
    std::auto_ptr<Module> M(LoadFile(InputFilenames[i]));
    if (M.get() == 0) return 1;

    if (Verbose) std::cerr << "Linking in '" << InputFilenames[i] << "'\n";

    if (LinkModules(Composite.get(), M.get(), &ErrorMessage)) {
      std::cerr << argv[0] << ": link error in '" << InputFilenames[i]
                << "': " << ErrorMessage << "\n";
      return 1;
    }
  }

  // Get the list of dependent libraries from the composite module
  const Module::LibraryListType& libs = Composite.get()->getLibraries();

  // Iterate over the list of dependent libraries, linking them in as we
  // find them
  Module::LibraryListType::const_iterator I = libs.begin();
  while (I != libs.end()) {
    Module* Mod = 0;
    if (LoadLibrary(*I,Mod)) {
      if (Mod != 0) {
        std::auto_ptr<Module> M(Mod);
        if (LinkModules(Composite.get(), M.get(), &ErrorMessage)) {
          std::cerr << argv[0] << ": link error in '" << *I
                << "': " << ErrorMessage << "\n";
          return 1;
        }
      } else {
        std::cerr << argv[0] << ": confused loading library '" << *I
          << "'. Aborting\n";
        return 2;
      }
    }
    ++I;
  }

  // TODO: Iterate over the -l list and link in any modules containing
  // global symbols that have not been resolved so far.

  if (DumpAsm) std::cerr << "Here's the assembly:\n" << Composite.get();

  std::ostream *Out = &std::cout;  // Default to printing to stdout...
  if (OutputFilename != "-") {
    if (!Force && std::ifstream(OutputFilename.c_str())) {
      // If force is not specified, make sure not to overwrite a file!
      std::cerr << argv[0] << ": error opening '" << OutputFilename
                << "': file exists!\n"
                << "Use -f command line argument to force output\n";
      return 1;
    }
    Out = new std::ofstream(OutputFilename.c_str());
    if (!Out->good()) {
      std::cerr << argv[0] << ": error opening '" << OutputFilename << "'!\n";
      return 1;
    }

    // Make sure that the Out file gets unlinked from the disk if we get a
    // SIGINT
    sys::RemoveFileOnSignal(OutputFilename);
  }

  if (verifyModule(*Composite.get())) {
    std::cerr << argv[0] << ": linked module is broken!\n";
    return 1;
  }

  if (Verbose) std::cerr << "Writing bytecode...\n";
  WriteBytecodeToFile(Composite.get(), *Out);

  if (Out != &std::cout) delete Out;
  return 0;
}
