//===-- tools/bugpoint/ToolRunner.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes an abstraction around a platform C compiler, used to
// compile C and assembly code.  It also exposes an "AbstractIntepreter"
// interface, which is used to execute code using one of the LLVM execution
// engines.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_BUGPOINT_TOOLRUNNER_H
#define LLVM_TOOLS_BUGPOINT_TOOLRUNNER_H

#include "llvm/ADT/Triple.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SystemUtils.h"
#include <exception>
#include <vector>

namespace llvm {

extern cl::opt<bool> SaveTemps;
extern Triple TargetTriple;

class LLC;

//===---------------------------------------------------------------------===//
// GCC abstraction
//
class GCC {
  std::string GCCPath;                // The path to the gcc executable.
  std::string RemoteClientPath;       // The path to the rsh / ssh executable.
  std::vector<std::string> gccArgs; // GCC-specific arguments.
  GCC(StringRef gccPath, StringRef RemotePath,
      const std::vector<std::string> *GCCArgs)
    : GCCPath(gccPath), RemoteClientPath(RemotePath) {
    if (GCCArgs) gccArgs = *GCCArgs;
  }
public:
  enum FileType { AsmFile, ObjectFile, CFile };

  static GCC *create(std::string &Message,
                     const std::string &GCCBinary,
                     const std::vector<std::string> *Args);

  /// ExecuteProgram - Execute the program specified by "ProgramFile" (which is
  /// either a .s file, or a .c file, specified by FileType), with the specified
  /// arguments.  Standard input is specified with InputFile, and standard
  /// Output is captured to the specified OutputFile location.  The SharedLibs
  /// option specifies optional native shared objects that can be loaded into
  /// the program for execution.
  ///
  int ExecuteProgram(const std::string &ProgramFile,
                     const std::vector<std::string> &Args,
                     FileType fileType,
                     const std::string &InputFile,
                     const std::string &OutputFile,
                     std::string *Error = nullptr,
                     const std::vector<std::string> &GCCArgs =
                         std::vector<std::string>(),
                     unsigned Timeout = 0,
                     unsigned MemoryLimit = 0);

  /// MakeSharedObject - This compiles the specified file (which is either a .c
  /// file or a .s file) into a shared object.
  ///
  int MakeSharedObject(const std::string &InputFile, FileType fileType,
                       std::string &OutputFile,
                       const std::vector<std::string> &ArgsForGCC,
                       std::string &Error);
};


//===---------------------------------------------------------------------===//
/// AbstractInterpreter Class - Subclasses of this class are used to execute
/// LLVM bitcode in a variety of ways.  This abstract interface hides this
/// complexity behind a simple interface.
///
class AbstractInterpreter {
  virtual void anchor();
public:
  static LLC *createLLC(const char *Argv0, std::string &Message,
                        const std::string              &GCCBinary,
                        const std::vector<std::string> *Args = nullptr,
                        const std::vector<std::string> *GCCArgs = nullptr,
                        bool UseIntegratedAssembler = false);

  static AbstractInterpreter*
  createLLI(const char *Argv0, std::string &Message,
            const std::vector<std::string> *Args = nullptr);

  static AbstractInterpreter*
  createJIT(const char *Argv0, std::string &Message,
            const std::vector<std::string> *Args = nullptr);

  static AbstractInterpreter*
  createCustomCompiler(std::string &Message,
                       const std::string &CompileCommandLine);

  static AbstractInterpreter*
  createCustomExecutor(std::string &Message,
                       const std::string &ExecCommandLine);


  virtual ~AbstractInterpreter() {}

  /// compileProgram - Compile the specified program from bitcode to executable
  /// code.  This does not produce any output, it is only used when debugging
  /// the code generator.  It returns false if the code generator fails.
  virtual void compileProgram(const std::string &Bitcode, std::string *Error,
                              unsigned Timeout = 0, unsigned MemoryLimit = 0) {}

  /// OutputCode - Compile the specified program from bitcode to code
  /// understood by the GCC driver (either C or asm).  If the code generator
  /// fails, it sets Error, otherwise, this function returns the type of code
  /// emitted.
  virtual GCC::FileType OutputCode(const std::string &Bitcode,
                                   std::string &OutFile, std::string &Error,
                                   unsigned Timeout = 0,
                                   unsigned MemoryLimit = 0) {
    Error = "OutputCode not supported by this AbstractInterpreter!";
    return GCC::AsmFile;
  }

  /// ExecuteProgram - Run the specified bitcode file, emitting output to the
  /// specified filename.  This sets RetVal to the exit code of the program or
  /// returns false if a problem was encountered that prevented execution of
  /// the program.
  ///
  virtual int ExecuteProgram(const std::string &Bitcode,
                             const std::vector<std::string> &Args,
                             const std::string &InputFile,
                             const std::string &OutputFile,
                             std::string *Error,
                             const std::vector<std::string> &GCCArgs =
                               std::vector<std::string>(),
                             const std::vector<std::string> &SharedLibs =
                               std::vector<std::string>(),
                             unsigned Timeout = 0,
                             unsigned MemoryLimit = 0) = 0;
};

//===---------------------------------------------------------------------===//
// LLC Implementation of AbstractIntepreter interface
//
class LLC : public AbstractInterpreter {
  std::string LLCPath;               // The path to the LLC executable.
  std::vector<std::string> ToolArgs; // Extra args to pass to LLC.
  GCC *gcc;
  bool UseIntegratedAssembler;
public:
  LLC(const std::string &llcPath, GCC *Gcc,
      const std::vector<std::string> *Args,
      bool useIntegratedAssembler)
    : LLCPath(llcPath), gcc(Gcc),
      UseIntegratedAssembler(useIntegratedAssembler) {
    ToolArgs.clear();
    if (Args) ToolArgs = *Args;
  }
  ~LLC() { delete gcc; }

  /// compileProgram - Compile the specified program from bitcode to executable
  /// code.  This does not produce any output, it is only used when debugging
  /// the code generator.  Returns false if the code generator fails.
  void compileProgram(const std::string &Bitcode, std::string *Error,
                      unsigned Timeout = 0, unsigned MemoryLimit = 0) override;

  int ExecuteProgram(const std::string &Bitcode,
                     const std::vector<std::string> &Args,
                     const std::string &InputFile,
                     const std::string &OutputFile,
                     std::string *Error,
                     const std::vector<std::string> &GCCArgs =
                       std::vector<std::string>(),
                     const std::vector<std::string> &SharedLibs =
                        std::vector<std::string>(),
                     unsigned Timeout = 0,
                     unsigned MemoryLimit = 0) override;

  /// OutputCode - Compile the specified program from bitcode to code
  /// understood by the GCC driver (either C or asm).  If the code generator
  /// fails, it sets Error, otherwise, this function returns the type of code
  /// emitted.
  GCC::FileType OutputCode(const std::string &Bitcode,
                           std::string &OutFile, std::string &Error,
                           unsigned Timeout = 0,
                           unsigned MemoryLimit = 0) override;
};

} // End llvm namespace

#endif
