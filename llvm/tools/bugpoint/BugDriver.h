//===- BugDriver.h - Top-Level BugPoint class -------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This class contains all of the shared state and information that is used by
// the BugPoint tool to track down errors in optimizations.  This class is the
// main driver class that invokes all sub-functionality.
//
//===----------------------------------------------------------------------===//

#ifndef BUGDRIVER_H
#define BUGDRIVER_H

#include <vector>
#include <string>

namespace llvm {

class PassInfo;
class Module;
class Function;
class AbstractInterpreter;
class Instruction;

class DebugCrashes;

class CBE;
class GCC;

extern bool DisableSimplifyCFG;

class BugDriver {
  const std::string ToolName;  // Name of bugpoint
  std::string ReferenceOutputFile; // Name of `good' output file
  Module *Program;             // The raw program, linked together
  std::vector<const PassInfo*> PassesToRun;
  AbstractInterpreter *Interpreter;   // How to run the program
  CBE *cbe;
  GCC *gcc;

  // FIXME: sort out public/private distinctions...
  friend class ReducePassList;
  friend class ReduceMisCodegenFunctions;

public:
  BugDriver(const char *toolname);

  const std::string &getToolName() const { return ToolName; }

  // Set up methods... these methods are used to copy information about the
  // command line arguments into instance variables of BugDriver.
  //
  bool addSources(const std::vector<std::string> &FileNames);
  template<class It>
  void addPasses(It I, It E) { PassesToRun.insert(PassesToRun.end(), I, E); }
  void setPassesToRun(const std::vector<const PassInfo*> &PTR) {
    PassesToRun = PTR;
  }
  const std::vector<const PassInfo*> &getPassesToRun() const {
    return PassesToRun;
  }

  /// run - The top level method that is invoked after all of the instance
  /// variables are set up from command line arguments.
  ///
  bool run();

  /// debugOptimizerCrash - This method is called when some optimizer pass
  /// crashes on input.  It attempts to prune down the testcase to something
  /// reasonable, and figure out exactly which pass is crashing.
  ///
  bool debugOptimizerCrash();
  
  /// debugCodeGeneratorCrash - This method is called when the code generator
  /// crashes on an input.  It attempts to reduce the input as much as possible
  /// while still causing the code generator to crash.
  bool debugCodeGeneratorCrash();

  /// debugMiscompilation - This method is used when the passes selected are not
  /// crashing, but the generated output is semantically different from the
  /// input.
  bool debugMiscompilation();

  /// debugPassMiscompilation - This method is called when the specified pass
  /// miscompiles Program as input.  It tries to reduce the testcase to
  /// something that smaller that still miscompiles the program.
  /// ReferenceOutput contains the filename of the file containing the output we
  /// are to match.
  ///
  bool debugPassMiscompilation(const PassInfo *ThePass,
			       const std::string &ReferenceOutput);

  /// compileSharedObject - This method creates a SharedObject from a given
  /// BytecodeFile for debugging a code generator.
  ///
  std::string compileSharedObject(const std::string &BytecodeFile);

  /// debugCodeGenerator - This method narrows down a module to a function or
  /// set of functions, using the CBE as a ``safe'' code generator for other
  /// functions that are not under consideration.
  bool debugCodeGenerator();

  /// isExecutingJIT - Returns true if bugpoint is currently testing the JIT
  ///
  bool isExecutingJIT();

  /// runPasses - Run all of the passes in the "PassesToRun" list, discard the
  /// output, and return true if any of the passes crashed.
  bool runPasses(Module *M = 0) {
    if (M == 0) M = Program;
    std::swap(M, Program);
    bool Result = runPasses(PassesToRun);
    std::swap(M, Program);
    return Result;
  }

  Module *getProgram() const { return Program; }

  /// swapProgramIn - Set the current module to the specified module, returning
  /// the old one.
  Module *swapProgramIn(Module *M) {
    Module *OldProgram = Program;
    Program = M;
    return OldProgram;
  }

  /// setNewProgram - If we reduce or update the program somehow, call this
  /// method to update bugdriver with it.  This deletes the old module and sets
  /// the specified one as the current program.
  void setNewProgram(Module *M);

  /// compileProgram - Try to compile the specified module, throwing an
  /// exception if an error occurs, or returning normally if not.  This is used
  /// for code generation crash testing.
  ///
  void compileProgram(Module *M);

  /// executeProgram - This method runs "Program", capturing the output of the
  /// program to a file, returning the filename of the file.  A recommended
  /// filename may be optionally specified.  If there is a problem with the code
  /// generator (e.g., llc crashes), this will throw an exception.
  ///
  std::string executeProgram(std::string RequestedOutputFilename = "",
                             std::string Bytecode = "",
                             const std::string &SharedObjects = "",
                             AbstractInterpreter *AI = 0,
                             bool *ProgramExitedNonzero = 0);

  /// executeProgramWithCBE - Used to create reference output with the C
  /// backend, if reference output is not provided.  If there is a problem with
  /// the code generator (e.g., llc crashes), this will throw an exception.
  ///
  std::string executeProgramWithCBE(std::string OutputFile = "");

  /// diffProgram - This method executes the specified module and diffs the
  /// output against the file specified by ReferenceOutputFile.  If the output
  /// is different, true is returned.  If there is a problem with the code
  /// generator (e.g., llc crashes), this will throw an exception.
  ///
  bool diffProgram(const std::string &BytecodeFile = "",
                   const std::string &SharedObj = "",
                   bool RemoveBytecode = false);
  /// EmitProgressBytecode - This function is used to output the current Program
  /// to a file named "bugpoint-ID.bc".
  ///
  void EmitProgressBytecode(const std::string &ID, bool NoFlyer = false);

  /// deleteInstructionFromProgram - This method clones the current Program and
  /// deletes the specified instruction from the cloned module.  It then runs a
  /// series of cleanup passes (ADCE and SimplifyCFG) to eliminate any code
  /// which depends on the value.  The modified module is then returned.
  ///
  Module *deleteInstructionFromProgram(const Instruction *I, unsigned Simp)
    const;

  /// performFinalCleanups - This method clones the current Program and performs
  /// a series of cleanups intended to get rid of extra cruft on the module.  If
  /// the MayModifySemantics argument is true, then the cleanups is allowed to
  /// modify how the code behaves.
  ///
  Module *performFinalCleanups(Module *M, bool MayModifySemantics = false);

  /// ExtractLoop - Given a module, extract up to one loop from it into a new
  /// function.  This returns null if there are no extractable loops in the
  /// program or if the loop extractor crashes.
  Module *ExtractLoop(Module *M);

  /// runPassesOn - Carefully run the specified set of pass on the specified
  /// module, returning the transformed module on success, or a null pointer on
  /// failure.  If AutoDebugCrashes is set to true, then bugpoint will
  /// automatically attempt to track down a crashing pass if one exists, and
  /// this method will never return null.
  Module *runPassesOn(Module *M, const std::vector<const PassInfo*> &Passes,
                      bool AutoDebugCrashes = false);

  /// runPasses - Run the specified passes on Program, outputting a bytecode
  /// file and writting the filename into OutputFile if successful.  If the
  /// optimizations fail for some reason (optimizer crashes), return true,
  /// otherwise return false.  If DeleteOutput is set to true, the bytecode is
  /// deleted on success, and the filename string is undefined.  This prints to
  /// cout a single line message indicating whether compilation was successful
  /// or failed, unless Quiet is set.
  ///
  bool runPasses(const std::vector<const PassInfo*> &PassesToRun,
                 std::string &OutputFilename, bool DeleteOutput = false,
		 bool Quiet = false) const;
private:
  /// writeProgramToFile - This writes the current "Program" to the named
  /// bytecode file.  If an error occurs, true is returned.
  ///
  bool writeProgramToFile(const std::string &Filename, Module *M = 0) const;

  /// runPasses - Just like the method above, but this just returns true or
  /// false indicating whether or not the optimizer crashed on the specified
  /// input (true = crashed).
  ///
  bool runPasses(const std::vector<const PassInfo*> &PassesToRun,
                 bool DeleteOutput = true) const {
    std::string Filename;
    return runPasses(PassesToRun, Filename, DeleteOutput);
  }

  /// initializeExecutionEnvironment - This method is used to set up the
  /// environment for executing LLVM programs.
  ///
  bool initializeExecutionEnvironment();
};

/// ParseInputFile - Given a bytecode or assembly input filename, parse and
/// return it, or return null if not possible.
///
Module *ParseInputFile(const std::string &InputFilename);


/// getPassesString - Turn a list of passes into a string which indicates the
/// command line options that must be passed to add the passes.
///
std::string getPassesString(const std::vector<const PassInfo*> &Passes);

/// PrintFunctionList - prints out list of problematic functions
///
void PrintFunctionList(const std::vector<Function*> &Funcs);

// DeleteFunctionBody - "Remove" the function by deleting all of it's basic
// blocks, making it external.
//
void DeleteFunctionBody(Function *F);

/// SplitFunctionsOutOfModule - Given a module and a list of functions in the
/// module, split the functions OUT of the specified module, and place them in
/// the new module.
Module *SplitFunctionsOutOfModule(Module *M, const std::vector<Function*> &F);

} // End llvm namespace

#endif
