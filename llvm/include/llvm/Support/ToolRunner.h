//===-- Support/ToolRunner.h ------------------------------------*- C++ -*-===//
//
// This file exposes an abstraction around a platform C compiler, used to
// compile C and assembly code.  It also exposes an "AbstractIntepreter"
// interface, which is used to execute code using one of the LLVM execution
// engines.
//
//===----------------------------------------------------------------------===//

#ifndef TOOLRUNNER_H
#define TOOLRUNNER_H

#include "Support/SystemUtils.h"
#include <vector>

class CBE;
class LLC;

//===---------------------------------------------------------------------===//
// GCC abstraction
//
class GCC {
  std::string GCCPath;          // The path to the gcc executable
  GCC(const std::string &gccPath) : GCCPath(gccPath) { }
public:
  enum FileType { AsmFile, CFile };

  static GCC* create(const std::string &ProgramPath, std::string &Message);

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
                     const std::vector<std::string> &SharedLibs = 
                         std::vector<std::string>());

  /// MakeSharedObject - This compiles the specified file (which is either a .c
  /// file or a .s file) into a shared object.
  ///
  int MakeSharedObject(const std::string &InputFile, FileType fileType,
                       std::string &OutputFile);
  
private:
  void ProcessFailure(const char **Args);
};


//===---------------------------------------------------------------------===//
/// AbstractInterpreter Class - Subclasses of this class are used to execute
/// LLVM bytecode in a variety of ways.  This abstract interface hides this
/// complexity behind a simple interface.
///
struct AbstractInterpreter {
  static CBE* createCBE(const std::string &ProgramPath, std::string &Message);
  static LLC *createLLC(const std::string &ProgramPath, std::string &Message);

  static AbstractInterpreter* createLLI(const std::string &ProgramPath,
                                        std::string &Message);

  static AbstractInterpreter* createJIT(const std::string &ProgramPath,
                                        std::string &Message);


  virtual ~AbstractInterpreter() {}

  /// ExecuteProgram - Run the specified bytecode file, emitting output to the
  /// specified filename.  This returns the exit code of the program.
  ///
  virtual int ExecuteProgram(const std::string &Bytecode,
                             const std::vector<std::string> &Args,
                             const std::string &InputFile,
                             const std::string &OutputFile,
                             const std::vector<std::string> &SharedLibs = 
                               std::vector<std::string>()) = 0;
};

//===---------------------------------------------------------------------===//
// CBE Implementation of AbstractIntepreter interface
//
class CBE : public AbstractInterpreter {
  std::string DISPath;          // The path to the `llvm-dis' executable
  GCC *gcc;
public:
  CBE(const std::string &disPath, GCC *Gcc) : DISPath(disPath), gcc(Gcc) { }
  ~CBE() { delete gcc; }

  virtual int ExecuteProgram(const std::string &Bytecode,
                             const std::vector<std::string> &Args,
                             const std::string &InputFile,
                             const std::string &OutputFile,
                             const std::vector<std::string> &SharedLibs = 
                               std::vector<std::string>());

  // Sometimes we just want to go half-way and only generate the .c file,
  // not necessarily compile it with GCC and run the program.
  //
  virtual int OutputC(const std::string &Bytecode, std::string &OutputCFile);
};


//===---------------------------------------------------------------------===//
// LLC Implementation of AbstractIntepreter interface
//
class LLC : public AbstractInterpreter {
  std::string LLCPath;          // The path to the LLC executable
  GCC *gcc;
public:
  LLC(const std::string &llcPath, GCC *Gcc)
    : LLCPath(llcPath), gcc(Gcc) { }
  ~LLC() { delete gcc; }

  virtual int ExecuteProgram(const std::string &Bytecode,
                             const std::vector<std::string> &Args,
                             const std::string &InputFile,
                             const std::string &OutputFile,
                             const std::vector<std::string> &SharedLibs = 
                                std::vector<std::string>());

  // Sometimes we just want to go half-way and only generate the .s file,
  // not necessarily compile it all the way and run the program.
  //
  int OutputAsm(const std::string &Bytecode, std::string &OutputAsmFile);
};

#endif
