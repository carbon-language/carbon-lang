#ifndef TOOLRUNNER_H
#define TOOLRUNNER_H

#include "Support/CommandLine.h"
#include "Support/SystemUtils.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

enum FileType { AsmFile, CFile };

//===---------------------------------------------------------------------===//
// GCC abstraction
//
// This is not a *real* AbstractInterpreter as it does not accept bytecode
// files, but only input acceptable to GCC, i.e. C, C++, and assembly files
//
class GCC {
  std::string GCCPath;          // The path to the gcc executable
public:
  GCC(const std::string &gccPath) : GCCPath(gccPath) { }
  virtual ~GCC() {}

  virtual int ExecuteProgram(const std::string &ProgramFile,
                             const cl::list<std::string> &Args,
                             FileType fileType,
                             const std::string &InputFile,
                             const std::string &OutputFile,
                             const std::string &SharedLib = "");

  int MakeSharedObject(const std::string &InputFile,
                       FileType fileType,
                       std::string &OutputFile);
  
  void ProcessFailure(const char **Args);
};

GCC* createGCCtool(const std::string &ProgramPath,
                   std::string &Message);

/// AbstractInterpreter Class - Subclasses of this class are used to execute
/// LLVM bytecode in a variety of ways.  This abstract interface hides this
/// complexity behind a simple interface.
///
struct AbstractInterpreter {

  virtual ~AbstractInterpreter() {}

  /// ExecuteProgram - Run the specified bytecode file, emitting output to the
  /// specified filename.  This returns the exit code of the program.
  ///
  virtual int ExecuteProgram(const std::string &Bytecode,
                             const cl::list<std::string> &Args,
                             const std::string &InputFile,
                             const std::string &OutputFile,
                             const std::string &SharedLib = "") = 0;
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
                             const cl::list<std::string> &Args,
                             const std::string &InputFile,
                             const std::string &OutputFile,
                             const std::string &SharedLib = "");

  // Sometimes we just want to go half-way and only generate the C file,
  // not necessarily compile it with GCC and run the program
  virtual int OutputC(const std::string &Bytecode,
                      std::string &OutputCFile);

};

CBE* createCBEtool(const std::string &ProgramPath, std::string &Message);

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
                             const cl::list<std::string> &Args,
                             const std::string &InputFile,
                             const std::string &OutputFile,
                             const std::string &SharedLib = "");

  int OutputAsm(const std::string &Bytecode,
                std::string &OutputAsmFile);
};

LLC* createLLCtool(const std::string &ProgramPath, std::string &Message);

AbstractInterpreter* createLLItool(const std::string &ProgramPath,
                                   std::string &Message);

AbstractInterpreter* createJITtool(const std::string &ProgramPath,
                                   std::string &Message);


#endif
