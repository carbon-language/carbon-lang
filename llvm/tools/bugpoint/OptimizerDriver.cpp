//===- OptimizerDriver.cpp - Allow BugPoint to run passes safely ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an interface that allows bugpoint to run various passes
// without the threat of a buggy pass corrupting bugpoint (of course, bugpoint
// may have its own bugs, but that's another story...).  It achieves this by
// forking a copy of itself and having the child process do the optimizations.
// If this client dies, we can always fork a new one.  :)
//
//===----------------------------------------------------------------------===//

// Note: as a short term hack, the old Unix-specific code and platform-
// independent code co-exist via conditional compilation until it is verified
// that the new code works correctly on Unix.

#ifdef _MSC_VER
#define PLATFORMINDEPENDENT
#endif

#include "BugDriver.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bytecode/WriteBytecodePass.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/System/Path.h"
#include <fstream>
#ifndef PLATFORMINDEPENDENT
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#endif
using namespace llvm;

/// writeProgramToFile - This writes the current "Program" to the named bytecode
/// file.  If an error occurs, true is returned.
///
bool BugDriver::writeProgramToFile(const std::string &Filename,
                                   Module *M) const {
  std::ios::openmode io_mode = std::ios::out | std::ios::trunc |
                               std::ios::binary;
  std::ofstream Out(Filename.c_str(), io_mode);
  if (!Out.good()) return true;
  WriteBytecodeToFile(M ? M : Program, Out, /*compression=*/true);
  return false;
}


/// EmitProgressBytecode - This function is used to output the current Program
/// to a file named "bugpoint-ID.bc".
///
void BugDriver::EmitProgressBytecode(const std::string &ID, bool NoFlyer) {
  // Output the input to the current pass to a bytecode file, emit a message
  // telling the user how to reproduce it: opt -foo blah.bc
  //
  std::string Filename = "bugpoint-" + ID + ".bc";
  if (writeProgramToFile(Filename)) {
    std::cerr <<  "Error opening file '" << Filename << "' for writing!\n";
    return;
  }

  std::cout << "Emitted bytecode to '" << Filename << "'\n";
  if (NoFlyer || PassesToRun.empty()) return;
  std::cout << "\n*** You can reproduce the problem with: ";

  unsigned PassType = PassesToRun[0]->getPassType();
  for (unsigned i = 1, e = PassesToRun.size(); i != e; ++i)
    PassType &= PassesToRun[i]->getPassType();

  if (PassType & PassInfo::Analysis)
    std::cout << "analyze";
  else if (PassType & PassInfo::Optimization)
    std::cout << "opt";
  else if (PassType & PassInfo::LLC)
    std::cout << "llc";
  else
    std::cout << "bugpoint";
  std::cout << " " << Filename << " ";
  std::cout << getPassesString(PassesToRun) << "\n";
}

static void RunChild(Module *Program,const std::vector<const PassInfo*> &Passes,
                     const std::string &OutFilename) {
  std::ios::openmode io_mode = std::ios::out | std::ios::trunc |
                               std::ios::binary;
  std::ofstream OutFile(OutFilename.c_str(), io_mode);
  if (!OutFile.good()) {
    std::cerr << "Error opening bytecode file: " << OutFilename << "\n";
    exit(1);
  }

  PassManager PM;
  // Make sure that the appropriate target data is always used...
  PM.add(new TargetData("bugpoint", Program));

  for (unsigned i = 0, e = Passes.size(); i != e; ++i) {
    if (Passes[i]->getNormalCtor())
      PM.add(Passes[i]->getNormalCtor()());
    else
      std::cerr << "Cannot create pass yet: " << Passes[i]->getPassName()
                << "\n";
  }
  // Check that the module is well formed on completion of optimization
  PM.add(createVerifierPass());

  // Write bytecode out to disk as the last step...
  PM.add(new WriteBytecodePass(&OutFile));

  // Run all queued passes.
  PM.run(*Program);
}

/// runPasses - Run the specified passes on Program, outputting a bytecode file
/// and writing the filename into OutputFile if successful.  If the
/// optimizations fail for some reason (optimizer crashes), return true,
/// otherwise return false.  If DeleteOutput is set to true, the bytecode is
/// deleted on success, and the filename string is undefined.  This prints to
/// cout a single line message indicating whether compilation was successful or
/// failed.
///
bool BugDriver::runPasses(const std::vector<const PassInfo*> &Passes,
                          std::string &OutputFilename, bool DeleteOutput,
                          bool Quiet) const{
  std::cout << std::flush;
  sys::Path uniqueFilename("bugpoint-output.bc");
  uniqueFilename.makeUnique();
  OutputFilename = uniqueFilename.toString();

#ifndef PLATFORMINDEPENDENT
  pid_t child_pid;
  switch (child_pid = fork()) {
  case -1:    // Error occurred
    std::cerr << ToolName << ": Error forking!\n";
    exit(1);
  case 0:     // Child process runs passes.
    RunChild(Program, Passes, OutputFilename);
    exit(0);  // If we finish successfully, return 0!
  default:    // Parent continues...
    break;
  }

  // Wait for the child process to get done.
  int Status;
  if (wait(&Status) != child_pid) {
    std::cerr << "Error waiting for child process!\n";
    exit(1);
  }

  bool ExitedOK = WIFEXITED(Status) && WEXITSTATUS(Status) == 0;
#else
  bool ExitedOK = false;
#endif

  // If we are supposed to delete the bytecode file or if the passes crashed,
  // remove it now.  This may fail if the file was never created, but that's ok.
  if (DeleteOutput || !ExitedOK)
    sys::Path(OutputFilename).destroy();

#ifndef PLATFORMINDEPENDENT
  if (!Quiet) {
    if (ExitedOK)
      std::cout << "Success!\n";
    else if (WIFEXITED(Status))
      std::cout << "Exited with error code '" << WEXITSTATUS(Status) << "'\n";
    else if (WIFSIGNALED(Status))
      std::cout << "Crashed with signal #" << WTERMSIG(Status) << "\n";
#ifdef WCOREDUMP
    else if (WCOREDUMP(Status))
      std::cout << "Dumped core\n";
#endif
    else
      std::cout << "Failed for unknown reason!\n";
  }
#endif

  // Was the child successful?
  return !ExitedOK;
}


/// runPassesOn - Carefully run the specified set of pass on the specified
/// module, returning the transformed module on success, or a null pointer on
/// failure.
Module *BugDriver::runPassesOn(Module *M,
                               const std::vector<const PassInfo*> &Passes,
                               bool AutoDebugCrashes) {
  Module *OldProgram = swapProgramIn(M);
  std::string BytecodeResult;
  if (runPasses(Passes, BytecodeResult, false/*delete*/, true/*quiet*/)) {
    if (AutoDebugCrashes) {
      std::cerr << " Error running this sequence of passes"
                << " on the input program!\n";
      delete OldProgram;
      EmitProgressBytecode("pass-error",  false);
      exit(debugOptimizerCrash());
    }
    swapProgramIn(OldProgram);
    return 0;
  }

  // Restore the current program.
  swapProgramIn(OldProgram);

  Module *Ret = ParseInputFile(BytecodeResult);
  if (Ret == 0) {
    std::cerr << getToolName() << ": Error reading bytecode file '"
              << BytecodeResult << "'!\n";
    exit(1);
  }
  sys::Path(BytecodeResult).destroy();  // No longer need the file on disk
  return Ret;
}
