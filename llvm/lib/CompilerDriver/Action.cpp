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

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/System/Program.h"
#include "llvm/System/TimeValue.h"

#include <stdexcept>
#include <string>

using namespace llvm;
using namespace llvmc;

namespace llvmc {

extern int Main(int argc, char** argv);
extern const char* ProgramName;

}

namespace {
  int ExecuteProgram(const std::string& name,
                     const StrVector& args) {
    sys::Path prog = sys::Program::FindProgramByName(name);

    if (prog.isEmpty()) {
      prog = FindExecutable(name, ProgramName, (void *)(intptr_t)&Main);
      if (prog.isEmpty())
        throw std::runtime_error("Can't find program '" + name + "'");
    }
    if (!prog.canExecute())
      throw std::runtime_error("Program '" + name + "' is not executable.");

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
    return sys::Program::ExecuteAndWait(prog, &argv[0], 0, &redirects[0]);
  }

  void print_string (const std::string& str) {
    errs() << str << ' ';
  }
}

namespace llvmc {
  void AppendToGlobalTimeLog(const std::string& cmd, double time);
}

int llvmc::Action::Execute() const {
  if (DryRun || VerboseMode) {
    errs() << Command_ << " ";
    std::for_each(Args_.begin(), Args_.end(), print_string);
    errs() << '\n';
  }
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
