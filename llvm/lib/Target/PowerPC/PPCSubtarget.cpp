//===- PowerPCSubtarget.cpp - PPC Subtarget Information -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Nate Begeman and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PPC specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "PPCSubtarget.h"
#include "PPC.h"
#include "llvm/Module.h"
#include "llvm/Support/CommandLine.h"
#include "PPCGenSubtarget.inc"

using namespace llvm;
PPCTargetEnum llvm::PPCTarget = TargetDefault;
bool llvm::PPCGenerateStaticCode = false;

namespace llvm {
  cl::opt<PPCTargetEnum, true>
  PPCTargetArg(cl::desc("Force generation of code for a specific PPC target:"),
               cl::values(
                          clEnumValN(TargetAIX,  "aix", "  Enable AIX codegen"),
                          clEnumValN(TargetDarwin,"darwin",
                                     "  Enable Darwin codegen"),
                          clEnumValEnd),
               cl::location(PPCTarget), cl::init(TargetDefault));
  
  cl::opt<bool, true>
  PPCStaticCode("ppc-static",
                cl::desc("PowerPC: generate completely non-pic code"),
                cl::location(PPCGenerateStaticCode));
} 
 
#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/host_info.h>
#include <mach/machine.h>

/// GetCurrentPowerPCFeatures - Returns the current CPUs features.
static const char *GetCurrentPowerPCCPU() {
  host_basic_info_data_t hostInfo;
  mach_msg_type_number_t infoCount;

  infoCount = HOST_BASIC_INFO_COUNT;
  host_info(mach_host_self(), HOST_BASIC_INFO, (host_info_t)&hostInfo, 
            &infoCount);
            
  if (hostInfo.cpu_type != CPU_TYPE_POWERPC) return "generic";

  switch(hostInfo.cpu_subtype) {
  case CPU_SUBTYPE_POWERPC_601:   return "601";
  case CPU_SUBTYPE_POWERPC_602:   return "602";
  case CPU_SUBTYPE_POWERPC_603:   return "603";
  case CPU_SUBTYPE_POWERPC_603e:  return "603e";
  case CPU_SUBTYPE_POWERPC_603ev: return "603ev";
  case CPU_SUBTYPE_POWERPC_604:   return "604";
  case CPU_SUBTYPE_POWERPC_604e:  return "604e";
  case CPU_SUBTYPE_POWERPC_620:   return "620";
  case CPU_SUBTYPE_POWERPC_750:   return "750";
  case CPU_SUBTYPE_POWERPC_7400:  return "7400";
  case CPU_SUBTYPE_POWERPC_7450:  return "7450";
  case CPU_SUBTYPE_POWERPC_970:   return "970";
  default: ;
  }
  
  return "generic";
}
#endif


PPCSubtarget::PPCSubtarget(const Module &M, const std::string &FS)
  : StackAlignment(16)
  , InstrItins()
  , IsGigaProcessor(false)
  , Is64Bit(false)
  , Has64BitRegs(false)
  , HasAltivec(false)
  , HasFSQRT(false)
  , IsAIX(false)
  , IsDarwin(false) {

  // Determine default and user specified characteristics
  std::string CPU = "generic";
#if defined(__APPLE__)
  CPU = GetCurrentPowerPCCPU();
#endif

  // Parse features string.
  ParseSubtargetFeatures(FS, CPU);

  // Set the boolean corresponding to the current target triple, or the default
  // if one cannot be determined, to true.
  const std::string& TT = M.getTargetTriple();
  if (TT.length() > 5) {
    IsDarwin = TT.find("darwin") != std::string::npos;
  } else if (TT.empty()) {
#if defined(_POWER)
    IsAIX = true;
#elif defined(__APPLE__)
    IsDarwin = true;
#endif
  }
}
