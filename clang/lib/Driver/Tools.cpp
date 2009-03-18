//===--- Tools.cpp - Tools Implementations ------------------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Tools.h"

using namespace clang::driver;
using namespace clang::driver::tools;

void Clang::ConstructJob(Compilation &C, const JobAction &JA,
                         const InputInfo &Output, 
                         const InputInfoList &Inputs,
                         const ArgList &TCArgs,
                         const char *LinkingOutput) const {
}

void gcc::Preprocess::ConstructJob(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output, 
                                   const InputInfoList &Inputs,
                                   const ArgList &TCArgs,
                                   const char *LinkingOutput) const {

}

void gcc::Precompile::ConstructJob(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output, 
                                   const InputInfoList &Inputs,
                                   const ArgList &TCArgs,
                                   const char *LinkingOutput) const {

}

void gcc::Compile::ConstructJob(Compilation &C, const JobAction &JA,
                                const InputInfo &Output, 
                                const InputInfoList &Inputs,
                                const ArgList &TCArgs,
                                const char *LinkingOutput) const {

}

void gcc::Assemble::ConstructJob(Compilation &C, const JobAction &JA,
                                 const InputInfo &Output, 
                                 const InputInfoList &Inputs,
                                 const ArgList &TCArgs,
                                 const char *LinkingOutput) const {

}

void gcc::Link::ConstructJob(Compilation &C, const JobAction &JA,
                             const InputInfo &Output, 
                             const InputInfoList &Inputs,
                             const ArgList &TCArgs,
                             const char *LinkingOutput) const {

}
