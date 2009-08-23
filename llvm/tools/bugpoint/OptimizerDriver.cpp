//===- OptimizerDriver.cpp - Allow BugPoint to run passes safely ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#include "BugDriver.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Path.h"
#include "llvm/System/Program.h"

#define DONT_GET_PLUGIN_LOADER_OPTION
#include "llvm/Support/PluginLoader.h"

#include <fstream>
using namespace llvm;


namespace {
  // ChildOutput - This option captures the name of the child output file that
  // is set up by the parent bugpoint process
  cl::opt<std::string> ChildOutput("child-output", cl::ReallyHidden);
  cl::opt<bool> UseValgrind("enable-valgrind",
                            cl::desc("Run optimizations through valgrind"));
}

/// writeProgramToFile - This writes the current "Program" to the named bitcode
/// file.  If an error occurs, true is returned.
///
bool BugDriver::writeProgramToFile(const std::string &Filename,
                                   Module *M) const {
  std::string ErrInfo;
  raw_fd_ostream Out(Filename.c_str(), ErrInfo,
                     raw_fd_ostream::F_Force|raw_fd_ostream::F_Binary);
  if (!ErrInfo.empty()) return true;
  
  WriteBitcodeToFile(M ? M : Program, Out);
  return false;
}


/// EmitProgressBitcode - This function is used to output the current Program
/// to a file named "bugpoint-ID.bc".
///
void BugDriver::EmitProgressBitcode(const std::string &ID, bool NoFlyer) {
  // Output the input to the current pass to a bitcode file, emit a message
  // telling the user how to reproduce it: opt -foo blah.bc
  //
  std::string Filename = "bugpoint-" + ID + ".bc";
  if (writeProgramToFile(Filename)) {
    errs() <<  "Error opening file '" << Filename << "' for writing!\n";
    return;
  }

  outs() << "Emitted bitcode to '" << Filename << "'\n";
  if (NoFlyer || PassesToRun.empty()) return;
  outs() << "\n*** You can reproduce the problem with: ";
  if (UseValgrind) outs() << "valgrind ";
  outs() << "opt " << Filename << " ";
  outs() << getPassesString(PassesToRun) << "\n";
}

int BugDriver::runPassesAsChild(const std::vector<const PassInfo*> &Passes) {
  std::string ErrInfo;
  raw_fd_ostream OutFile(ChildOutput.c_str(), ErrInfo,
                         raw_fd_ostream::F_Force|raw_fd_ostream::F_Binary);
  if (!ErrInfo.empty()) {
    errs() << "Error opening bitcode file: " << ChildOutput << "\n";
    return 1;
  }

  PassManager PM;
  // Make sure that the appropriate target data is always used...
  PM.add(new TargetData(Program));

  for (unsigned i = 0, e = Passes.size(); i != e; ++i) {
    if (Passes[i]->getNormalCtor())
      PM.add(Passes[i]->getNormalCtor()());
    else
      errs() << "Cannot create pass yet: " << Passes[i]->getPassName() << "\n";
  }
  // Check that the module is well formed on completion of optimization
  PM.add(createVerifierPass());

  // Write bitcode out to disk as the last step...
  PM.add(createBitcodeWriterPass(OutFile));

  // Run all queued passes.
  PM.run(*Program);

  return 0;
}

cl::opt<bool> SilencePasses("silence-passes", cl::desc("Suppress output of running passes (both stdout and stderr)"));

/// runPasses - Run the specified passes on Program, outputting a bitcode file
/// and writing the filename into OutputFile if successful.  If the
/// optimizations fail for some reason (optimizer crashes), return true,
/// otherwise return false.  If DeleteOutput is set to true, the bitcode is
/// deleted on success, and the filename string is undefined.  This prints to
/// outs() a single line message indicating whether compilation was successful
/// or failed.
///
bool BugDriver::runPasses(const std::vector<const PassInfo*> &Passes,
                          std::string &OutputFilename, bool DeleteOutput,
                          bool Quiet, unsigned NumExtraArgs,
                          const char * const *ExtraArgs) const {
  // setup the output file name
  outs().flush();
  sys::Path uniqueFilename("bugpoint-output.bc");
  std::string ErrMsg;
  if (uniqueFilename.makeUnique(true, &ErrMsg)) {
    errs() << getToolName() << ": Error making unique filename: "
           << ErrMsg << "\n";
    return(1);
  }
  OutputFilename = uniqueFilename.str();

  // set up the input file name
  sys::Path inputFilename("bugpoint-input.bc");
  if (inputFilename.makeUnique(true, &ErrMsg)) {
    errs() << getToolName() << ": Error making unique filename: "
           << ErrMsg << "\n";
    return(1);
  }
  
  std::string ErrInfo;
  raw_fd_ostream InFile(inputFilename.c_str(), ErrInfo,
                        raw_fd_ostream::F_Force|raw_fd_ostream::F_Binary);
  
  
  if (!ErrInfo.empty()) {
    errs() << "Error opening bitcode file: " << inputFilename.str() << "\n";
    return 1;
  }
  WriteBitcodeToFile(Program, InFile);
  InFile.close();

  // setup the child process' arguments
  SmallVector<const char*, 8> Args;
  sys::Path tool = sys::Program::FindProgramByName(ToolName);
  if (UseValgrind) {
    Args.push_back("valgrind");
    Args.push_back("--error-exitcode=1");
    Args.push_back("-q");
    Args.push_back(tool.c_str());
  } else
    Args.push_back(ToolName);

  Args.push_back("-as-child");
  Args.push_back("-child-output");
  Args.push_back(OutputFilename.c_str());
  std::vector<std::string> pass_args;
  for (unsigned i = 0, e = PluginLoader::getNumPlugins(); i != e; ++i) {
    pass_args.push_back( std::string("-load"));
    pass_args.push_back( PluginLoader::getPlugin(i));
  }
  for (std::vector<const PassInfo*>::const_iterator I = Passes.begin(),
       E = Passes.end(); I != E; ++I )
    pass_args.push_back( std::string("-") + (*I)->getPassArgument() );
  for (std::vector<std::string>::const_iterator I = pass_args.begin(),
       E = pass_args.end(); I != E; ++I )
    Args.push_back(I->c_str());
  Args.push_back(inputFilename.c_str());
  for (unsigned i = 0; i < NumExtraArgs; ++i)
    Args.push_back(*ExtraArgs);
  Args.push_back(0);

  sys::Path prog;
  if (UseValgrind)
    prog = sys::Program::FindProgramByName("valgrind");
  else
    prog = tool;
  
  // Redirect stdout and stderr to nowhere if SilencePasses is given
  sys::Path Nowhere;
  const sys::Path *Redirects[3] = {0, &Nowhere, &Nowhere};

  int result = sys::Program::ExecuteAndWait(prog, Args.data(), 0,
                                            (SilencePasses ? Redirects : 0),
                                            Timeout, MemoryLimit, &ErrMsg);

  // If we are supposed to delete the bitcode file or if the passes crashed,
  // remove it now.  This may fail if the file was never created, but that's ok.
  if (DeleteOutput || result != 0)
    sys::Path(OutputFilename).eraseFromDisk();

  // Remove the temporary input file as well
  inputFilename.eraseFromDisk();

  if (!Quiet) {
    if (result == 0)
      outs() << "Success!\n";
    else if (result > 0)
      outs() << "Exited with error code '" << result << "'\n";
    else if (result < 0) {
      if (result == -1)
        outs() << "Execute failed: " << ErrMsg << "\n";
      else
        outs() << "Crashed with signal #" << abs(result) << "\n";
    }
    if (result & 0x01000000)
      outs() << "Dumped core\n";
  }

  // Was the child successful?
  return result != 0;
}


/// runPassesOn - Carefully run the specified set of pass on the specified
/// module, returning the transformed module on success, or a null pointer on
/// failure.
Module *BugDriver::runPassesOn(Module *M,
                               const std::vector<const PassInfo*> &Passes,
                               bool AutoDebugCrashes, unsigned NumExtraArgs,
                               const char * const *ExtraArgs) {
  Module *OldProgram = swapProgramIn(M);
  std::string BitcodeResult;
  if (runPasses(Passes, BitcodeResult, false/*delete*/, true/*quiet*/,
                NumExtraArgs, ExtraArgs)) {
    if (AutoDebugCrashes) {
      errs() << " Error running this sequence of passes"
             << " on the input program!\n";
      delete OldProgram;
      EmitProgressBitcode("pass-error",  false);
      exit(debugOptimizerCrash());
    }
    swapProgramIn(OldProgram);
    return 0;
  }

  // Restore the current program.
  swapProgramIn(OldProgram);

  Module *Ret = ParseInputFile(BitcodeResult, Context);
  if (Ret == 0) {
    errs() << getToolName() << ": Error reading bitcode file '"
           << BitcodeResult << "'!\n";
    exit(1);
  }
  sys::Path(BitcodeResult).eraseFromDisk();  // No longer need the file on disk
  return Ret;
}
