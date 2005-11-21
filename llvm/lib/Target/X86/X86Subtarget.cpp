//===-- X86Subtarget.cpp - X86 Subtarget Information ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Nate Begeman and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the X86 specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "X86Subtarget.h"
#include "llvm/Module.h"
using namespace llvm;

X86Subtarget::X86Subtarget(const Module &M, const std::string &FS)
  : TargetSubtarget(), stackAlignment(8),
    indirectExternAndWeakGlobals(false), asmDarwinLinkerStubs(false),
    asmLeadingUnderscore(false), asmAlignmentIsInBytes(false),
    asmPrintDotLocalConstants(false), asmPrintDotLCommConstants(false),
    asmPrintConstantAlignment(false) {
      
  // Default to ELF unless otherwise specified.
  TargetType = isELF;
      
  // Set the boolean corresponding to the current target triple, or the default
  // if one cannot be determined, to true.
  const std::string& TT = M.getTargetTriple();
  if (TT.length() > 5) {
    if (TT.find("cygwin") != std::string::npos ||
        TT.find("mingw")  != std::string::npos)
      TargetType = isCygwin;
    else if (TT.find("darwin") != std::string::npos)
      TargetType = isDarwin;
    else if (TT.find("win32") != std::string::npos)
      TargetType = isWindows;
  } else if (TT.empty()) {
#if defined(__CYGWIN__) || defined(__MINGW32__)
    TargetType = isCygwin;
#elif defined(__APPLE__)
    TargetType = isDarwin;
#elif defined(_WIN32)
    TargetType = isWindows;
#endif
  }

  switch (TargetType) {
  case isCygwin:
    asmLeadingUnderscore = true;
    break;
  case isDarwin:
    stackAlignment = 16;
    indirectExternAndWeakGlobals = true;
    asmDarwinLinkerStubs = true;
    asmLeadingUnderscore = true;
    asmPrintDotLCommConstants = true;
    break;
  default: break;
  }
}
