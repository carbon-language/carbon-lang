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
#include "X86GenSubtarget.inc"
using namespace llvm;

static void GetCpuIDAndInfo(unsigned value, unsigned *EAX, unsigned *EBX,
                            unsigned *ECX, unsigned *EDX) {
#if defined(i386) || defined(__i386__) || defined(__x86__) || defined(_M_IX86)
#if defined(__GNUC__)
  asm ("pushl\t%%ebx\n\t"
       "cpuid\n\t"
       "movl\t%%ebx, %%esi\n\t"
       "popl\t%%ebx"
       : "=a" (*EAX),
         "=S" (*EBX),
         "=c" (*ECX),
         "=d" (*EDX)
       :  "a" (value));
#endif
#endif
}

static const char *GetCurrentX86CPU() {
  unsigned EAX = 0, EBX = 0, ECX = 0, EDX = 0;
  GetCpuIDAndInfo(0x1, &EAX, &EBX, &ECX, &EDX);
  unsigned Family  = (EAX & (0xffffffff >> (32 - 4)) << 8) >> 8; // Bits 8 - 11
  unsigned Model   = (EAX & (0xffffffff >> (32 - 4)) << 4) >> 4; // Bits 4 - 7
  GetCpuIDAndInfo(0x80000001, &EAX, &EBX, &ECX, &EDX);
  bool Em64T = EDX & (1 << 29);

  switch (Family) {
    case 3:
      return "i386";
    case 4:
      return "i486";
    case 5:
      switch (Model) {
      case 4:  return "pentium-mmx";
      default: return "pentium";
      }
      break;
    case 6:
      switch (Model) {
      case 1:  return "pentiumpro";
      case 3:
      case 5:
      case 6:  return "pentium2";
      case 7:
      case 8:
      case 10:
      case 11: return "pentium3";
      case 9:
      case 13: return "pentium-m";
      case 14: return "yonah";
      default: return "i686";
      }
    case 15: {
      switch (Model) {
      case 3:  
      case 4:
        return (Em64T) ? "nocona" : "prescott";
      default:
        return (Em64T) ? "x86-64" : "pentium4";
      }
    }
      
  default:
    return "generic";
  }
}

X86Subtarget::X86Subtarget(const Module &M, const std::string &FS)
  : stackAlignment(8), indirectExternAndWeakGlobals(false) {
      
  // Determine default and user specified characteristics
  std::string CPU = GetCurrentX86CPU();

  // Parse features string.
  ParseSubtargetFeatures(FS, CPU);

  // Default to ELF unless otherwise specified.
  TargetType = isELF;
  
  // FIXME: Force these off until they work.  An llc-beta option should turn
  // them back on.
  X86SSELevel = NoMMXSSE;
  X863DNowLevel = NoThreeDNow;
      
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

  if (TargetType == isDarwin) {
    stackAlignment = 16;
    indirectExternAndWeakGlobals = true;
  }
}
