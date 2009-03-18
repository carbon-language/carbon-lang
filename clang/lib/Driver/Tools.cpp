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
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Job.h"
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
}

void gcc::Preprocess::ConstructJob(Compilation &C, const JobAction &JA,
                                   Job &Dest,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &TCArgs,
                                   const char *LinkingOutput) const {

}

void gcc::Precompile::ConstructJob(Compilation &C, const JobAction &JA,
                                   Job &Dest,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &TCArgs,
                                   const char *LinkingOutput) const {

}

void gcc::Compile::ConstructJob(Compilation &C, const JobAction &JA,
                                Job &Dest,
                                const InputInfo &Output,
                                const InputInfoList &Inputs,
                                const ArgList &TCArgs,
                                const char *LinkingOutput) const {

}

void gcc::Assemble::ConstructJob(Compilation &C, const JobAction &JA,
                                 Job &Dest,
                                 const InputInfo &Output,
                                 const InputInfoList &Inputs,
                                 const ArgList &TCArgs,
                                 const char *LinkingOutput) const {

}

void gcc::Link::ConstructJob(Compilation &C, const JobAction &JA,
                             Job &Dest,
                             const InputInfo &Output,
                             const InputInfoList &Inputs,
                             const ArgList &TCArgs,
                             const char *LinkingOutput) const {

}
