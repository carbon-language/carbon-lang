//===--- Tools.cpp - Tools Implementations ------------------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Tools.h"

#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/HostInfo.h"
#include "clang/Driver/Option.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Driver/Util.h"

#include "llvm/ADT/SmallVector.h"

#include "InputInfo.h"

using namespace clang::driver;
using namespace clang::driver::tools;

void Clang::ConstructJob(Compilation &C, const JobAction &JA,
                         Job &Dest,
                         const InputInfo &Output,
                         const InputInfoList &Inputs,
                         const ArgList &TCArgs,
                         const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  for (InputInfoList::const_iterator
         it = Inputs.begin(), ie = Inputs.end(); it != ie; ++it) {
    const InputInfo &II = *it;
    if (II.isPipe())
      CmdArgs.push_back("-");
    else
      CmdArgs.push_back(II.getInputFilename());
  }

  if (Output.isPipe()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back("-");
  } else if (const char *N = Output.getInputFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(N);
  }

  Dest.addCommand(new Command("clang", CmdArgs));
}

void gcc::Common::ConstructJob(Compilation &C, const JobAction &JA,
                               Job &Dest,
                               const InputInfo &Output,
                               const InputInfoList &Inputs,
                               const ArgList &TCArgs,
                               const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  for (ArgList::const_iterator 
         it = TCArgs.begin(), ie = TCArgs.end(); it != ie; ++it) {
    Arg *A = *it;
    if (A->getOption().hasForwardToGCC())
      A->render(TCArgs, CmdArgs);
  }
  
  RenderExtraToolArgs(CmdArgs);

  // If using a driver driver, force the arch.
  if (getToolChain().getHost().useDriverDriver()) {
    CmdArgs.push_back("-arch");
    CmdArgs.push_back(getToolChain().getArchName().c_str());
  }

  if (Output.isPipe()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back("-");
  } else if (const char *N = Output.getInputFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(N);
  } else
    CmdArgs.push_back("-fsyntax-only");


  // Only pass -x if gcc will understand it; otherwise hope gcc
  // understands the suffix correctly. The main use case this would go
  // wrong in is for linker inputs if they happened to have an odd
  // suffix; really the only way to get this to happen is a command
  // like '-x foobar a.c' which will treat a.c like a linker input.
  //
  // FIXME: For the linker case specifically, can we safely convert
  // inputs into '-Wl,' options?
  for (InputInfoList::const_iterator
         it = Inputs.begin(), ie = Inputs.end(); it != ie; ++it) {
    const InputInfo &II = *it;
    if (types::canTypeBeUserSpecified(II.getType())) {
      CmdArgs.push_back("-x");
      CmdArgs.push_back(types::getTypeName(II.getType()));
    }

    if (II.isPipe())
      CmdArgs.push_back("-");
    else
      // FIXME: Linker inputs
      CmdArgs.push_back(II.getInputFilename());
  }

  Dest.addCommand(new Command("gcc", CmdArgs));
}

void gcc::Preprocess::RenderExtraToolArgs(ArgStringList &CmdArgs) const {
  CmdArgs.push_back("-E");
}

void gcc::Precompile::RenderExtraToolArgs(ArgStringList &CmdArgs) const {
  // The type is good enough.
}

void gcc::Compile::RenderExtraToolArgs(ArgStringList &CmdArgs) const {
  CmdArgs.push_back("-S");
}

void gcc::Assemble::RenderExtraToolArgs(ArgStringList &CmdArgs) const {
  CmdArgs.push_back("-c");
}

void gcc::Link::RenderExtraToolArgs(ArgStringList &CmdArgs) const {
  // The types are (hopefully) good enough.
}

