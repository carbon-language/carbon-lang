//===--- Action.cpp - The LLVM Compiler Driver ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Action class - implementation and auxiliary functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/CompilerDriver/Action.h"
#include "llvm/CompilerDriver/BuiltinOptions.h"
#include "llvm/CompilerDriver/Error.h"
#include "llvm/CompilerDriver/Main.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/System/Program.h"
#include "llvm/System/TimeValue.h"

#include <stdexcept>
#include <string>

using namespace llvm;
using namespace llvmc;

namespace llvmc {

extern const char* ProgramName;

}

namespace {

  void PrintString (const std::string& str) {
    errs() << str << ' ';
  }

  void PrintCommand (const std::string& Cmd, const StrVector& Args) {
    errs() << Cmd << ' ';
    std::for_each(Args.begin(), Args.end(), &PrintString);
    errs() << '\n';
  }

  bool IsSegmentationFault (int returnCode) {
#ifdef LLVM_ON_WIN32
    return (returnCode >= 0xc0000000UL)
#else
    return (returnCode < 0);
#endif
  }

  int ExecuteProgram (const std::string& name, const StrVector& args) {
    sys::Path prog(name);

    if (!prog.isAbsolute()) {
      prog = PrependMainExecutablePath(name, ProgramName,
                                       (void *)(intptr_t)&Main);

      if (!prog.canExecute()) {
        prog = sys::Program::FindProgramByName(name);
        if (prog.isEmpty()) {
          PrintError("Can't find program '" + name + "'");
          return -1;
        }
      }
    }
    if (!prog.canExecute()) {
      PrintError("Program '" + name + "' is not executable.");
      return -1;
    }

    // Build the command line vector and the redirects array.
    const sys::Path* redirects[3] = {0,0,0};
    sys::Path stdout_redirect;

    std::vector<const char*> argv;
    argv.reserve((args.size()+2));
    argv.push_back(name.c_str());

    for (StrVector::const_iterator B = args.begin(), E = args.end();
         B!=E; ++B) {
      if (*B == ">") {
        ++B;
        stdout_redirect.set(*B);
        redirects[1] = &stdout_redirect;
      }
      else {
        argv.push_back((*B).c_str());
      }
    }
    argv.push_back(0);  // null terminate list.

    // Invoke the program.
    int ret = sys::Program::ExecuteAndWait(prog, &argv[0], 0, &redirects[0]);

    if (IsSegmentationFault(ret)) {
      errs() << "Segmentation fault: ";
      PrintCommand(name, args);
    }

    return ret;
  }
}

namespace llvmc {
  void AppendToGlobalTimeLog (const std::string& cmd, double time);
}

int llvmc::Action::Execute () const {
  if (DryRun || VerboseMode)
    PrintCommand(Command_, Args_);

  if (!DryRun) {
    if (Time) {
      sys::TimeValue now = sys::TimeValue::now();
      int ret = ExecuteProgram(Command_, Args_);
      sys::TimeValue now2 = sys::TimeValue::now();
      now2 -= now;
      double elapsed = now2.seconds()  + now2.microseconds()  / 1000000.0;
      AppendToGlobalTimeLog(Command_, elapsed);

      return ret;
    }
    else {
      return ExecuteProgram(Command_, Args_);
    }
  }

  return 0;
}
