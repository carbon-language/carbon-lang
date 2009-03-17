//===--- ToolChain.cpp - Collections of tools for one platform ----------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/ToolChain.h"

#include "clang/Driver/Action.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/HostInfo.h"

using namespace clang::driver;

ToolChain::ToolChain(const HostInfo &_Host, const char *_Arch, 
                     const char *_Platform, const char *_OS) 
  : Host(_Host), Arch(_Arch), Platform(_Platform), OS(_OS) {
}

ToolChain::~ToolChain() {
}

llvm::sys::Path ToolChain::GetFilePath(const Compilation &C, 
                                       const char *Name) const {
  return Host.getDriver().GetFilePath(Name, this);
  
}

llvm::sys::Path ToolChain::GetProgramPath(const Compilation &C, 
                                          const char *Name) const {
  return Host.getDriver().GetProgramPath(Name, this);
}

bool ToolChain::ShouldUseClangCompiler(const Compilation &C, 
                                       const JobAction &JA) const {
  // Check if user requested no clang, or clang doesn't understand
  // this type (we only handle single inputs for now).
  if (Host.getDriver().CCCNoClang || JA.size() != 1 || 
      !types::isAcceptedByClang((*JA.begin())->getType()))
    return false;

  // Otherwise make sure this is an action clang undertands.
  if (isa<PreprocessJobAction>(JA)) {
    if (Host.getDriver().CCCNoClangCPP)
      return false;
  } else if (!isa<PrecompileJobAction>(JA) && !isa<CompileJobAction>(JA))
    return false;

  // Avoid CXX if the user requested.
  if (Host.getDriver().CCCNoClangCXX && types::isCXX((*JA.begin())->getType()))
    return false;

  // Finally, don't use clang if this isn't one of the user specified
  // archs to build.
  if (!Host.getDriver().CCCClangArchs.empty() && 
      Host.getDriver().CCCClangArchs.count(getArchName()))
    return false;

  return true;
}
