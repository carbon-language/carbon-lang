//===--- Driver.cpp - Clang GCC Compatible Driver -----------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Driver.h"

#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Option.h"
#include "clang/Driver/Options.h"

#include "llvm/Support/raw_ostream.h"
using namespace clang::driver;

Driver::Driver() : Opts(new OptTable()) {
  
}

Driver::~Driver() {
  delete Opts;
}

ArgList *Driver::ParseArgStrings(const char **ArgBegin, const char **ArgEnd) {
  ArgList *Args = new ArgList(ArgBegin, ArgEnd);
  
  unsigned Index = 0, End = ArgEnd - ArgBegin;
  while (Index < End) {
    unsigned Prev = Index;
    Arg *A = getOpts().ParseOneArg(*Args, Index, End);
    if (A)
      Args->append(A);

    assert(Index > Prev && "Parser failed to consume argument.");
  }

  return Args;
}

Compilation *Driver::BuildCompilation(int argc, const char **argv) {
  ArgList *Args = ParseArgStrings(argv + 1, argv + argc);

  // Hard coded to print-options behavior.
  unsigned i = 0;
  for (ArgList::iterator it = Args->begin(), ie = Args->end(); 
       it != ie; ++it, ++i) {
    Arg *A = *it;
    llvm::errs() << "Option " << i << " - "
                 << "Name: \"" << A->getOption().getName() << "\", "
                 << "Values: {";
    for (unsigned j = 0; j < A->getNumValues(); ++j) {
      if (j)
        llvm::errs() << ", ";
      llvm::errs() << '"' << A->getValue(*Args, j) << '"';
    }
    llvm::errs() << "}\n";
  }

  return new Compilation();
}
