//===- X86Subtarget.cpp - X86 Instruction Information -----------*- C++ -*-===//
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

X86Subtarget::X86Subtarget(const Module &M) 
  : TargetSubtarget(), stackAlignment(8), 
    indirectExternAndWeakGlobals(false), asmDarwinLinkerStubs(false),
    asmLeadingUnderscore(false), asmAlignmentIsInBytes(false),
    asmPrintDotLocalConstants(false), asmPrintDotLCommConstants(false),
    asmPrintConstantAlignment(false) {
  // Declare a boolean for each major platform.
  bool forCygwin = false;
  bool forDarwin = false;
  bool forWindows = false;
  
  // Set the boolean corresponding to the current target triple, or the default
  // if one cannot be determined, to true.
  const std::string& TT = M.getTargetTriple();
  if (TT.length() > 5) {
    forCygwin = TT.find("cygwin") != std::string::npos ||
                TT.find("mingw")  != std::string::npos;
    forDarwin = TT.find("darwin") != std::string::npos;
    forWindows = TT.find("win32") != std::string::npos;
  } else if (TT.empty()) {
#if defined(__CYGWIN__) || defined(__MINGW32__)
    forCygwin = true;
#elif defined(__APPLE__)
    forDarwin = true;
#elif defined(_WIN32)
    forWindows = true;
#endif
  }

  if (forCygwin) {
    asmLeadingUnderscore = true;
  } else if (forDarwin) {
    stackAlignment = 16;
    indirectExternAndWeakGlobals = true;
    asmDarwinLinkerStubs = true;
    asmLeadingUnderscore = true;
    asmPrintDotLCommConstants = true;
  } else if (forWindows) {
  }
}
