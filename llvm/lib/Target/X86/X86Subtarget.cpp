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

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/host_info.h>
#include <mach/machine.h>

/// GetCurrentX86CPU - Returns the current CPUs features.
static const char *GetCurrentX86CPU() {
  host_basic_info_data_t hostInfo;
  mach_msg_type_number_t infoCount;

  infoCount = HOST_BASIC_INFO_COUNT;
  host_info(mach_host_self(), HOST_BASIC_INFO, (host_info_t)&hostInfo, 
            &infoCount);
            
  if (hostInfo.cpu_type != CPU_TYPE_I386) return "generic";

  switch(hostInfo.cpu_subtype) {
  case CPU_SUBTYPE_386:            return "i386";
  case CPU_SUBTYPE_486:
  case CPU_SUBTYPE_486SX:          return "i486";
  case CPU_SUBTYPE_PENT:           return "pentium";
  case CPU_SUBTYPE_PENTPRO:        return "pentiumpro";
  case CPU_SUBTYPE_PENTII_M3:      return "pentium2";
  case CPU_SUBTYPE_PENTII_M5:      return "pentium2";
  case CPU_SUBTYPE_CELERON:
  case CPU_SUBTYPE_CELERON_MOBILE: return "celeron";
  case CPU_SUBTYPE_PENTIUM_3:      return "pentium3";
  case CPU_SUBTYPE_PENTIUM_3_M:    return "pentium3m";
  case CPU_SUBTYPE_PENTIUM_3_XEON: return "pentium3";   // FIXME: not sure.
  case CPU_SUBTYPE_PENTIUM_M:      return "pentium-m";
  case CPU_SUBTYPE_PENTIUM_4:      return "pentium4";
  case CPU_SUBTYPE_PENTIUM_4_M:    return "pentium4m";
  // FIXME: prescott, yonah? Check CPU_THREADTYPE_INTEL_HTT?
  case CPU_SUBTYPE_XEON:
  case CPU_SUBTYPE_XEON_MP:        return "nocona";
  default: ;
  }
  
  return "generic";
}
#endif

X86Subtarget::X86Subtarget(const Module &M, const std::string &FS)
  : stackAlignment(8), indirectExternAndWeakGlobals(false) {
      
  // Determine default and user specified characteristics
  std::string CPU = "generic";
#if defined(__APPLE__)
  CPU = GetCurrentX86CPU();
#endif

  // Parse features string.
  ParseSubtargetFeatures(FS, CPU);

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

  if (TargetType == isDarwin) {
    stackAlignment = 16;
    indirectExternAndWeakGlobals = true;
  }
}
