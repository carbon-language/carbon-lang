//===- PowerPCSubtarget.cpp - PPC Subtarget Information ---------*- C++ -*-===//
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
#include "llvm/Target/SubtargetFeature.h"

using namespace llvm;
PPCTargetEnum llvm::PPCTarget = TargetDefault;

namespace llvm {
  cl::opt<PPCTargetEnum, true>
  PPCTargetArg(cl::desc("Force generation of code for a specific PPC target:"),
               cl::values(
                          clEnumValN(TargetAIX,  "aix", "  Enable AIX codegen"),
                          clEnumValN(TargetDarwin,"darwin",
                                     "  Enable Darwin codegen"),
                          clEnumValEnd),
               cl::location(PPCTarget), cl::init(TargetDefault));
}

enum PowerPCFeature {
  PowerPCFeature64Bit   = 1 << 0,
  PowerPCFeatureAltivec = 1 << 1,
  PowerPCFeatureFSqrt   = 1 << 2,
  PowerPCFeatureGPUL    = 1 << 3,
  PowerPCFeature64BRegs = 1 << 4
};

/// Sorted (by key) array of values for CPU subtype.
static const SubtargetFeatureKV PowerPCSubTypeKV[] = {
  { "601"    , "Select the PowerPC 601 processor", 0 },
  { "602"    , "Select the PowerPC 602 processor", 0 },
  { "603"    , "Select the PowerPC 603 processor", 0 },
  { "603e"   , "Select the PowerPC 603e processor", 0 },
  { "603ev"  , "Select the PowerPC 603ev processor", 0 },
  { "604"    , "Select the PowerPC 604 processor", 0 },
  { "604e"   , "Select the PowerPC 604e processor", 0 },
  { "620"    , "Select the PowerPC 620 processor", 0 },
  { "7400"   , "Select the PowerPC 7400 (G4) processor",
               PowerPCFeatureAltivec },
  { "7450"   , "Select the PowerPC 7450 (G4+) processor",
               PowerPCFeatureAltivec },
  { "750"    , "Select the PowerPC 750 (G3) processor", 0 },
  { "970"    , "Select the PowerPC 970 (G5 - GPUL) processor",
               PowerPCFeature64Bit | PowerPCFeatureAltivec |
               PowerPCFeatureFSqrt | PowerPCFeatureGPUL },
  { "g3"     , "Select the PowerPC G3 (750) processor", 0 },
  { "g4"     , "Select the PowerPC G4 (7400) processor",
               PowerPCFeatureAltivec },
  { "g4+"    , "Select the PowerPC G4+ (7450) processor",
               PowerPCFeatureAltivec },
  { "g5"     , "Select the PowerPC g5 (970 - GPUL)  processor",
               PowerPCFeature64Bit | PowerPCFeatureAltivec |
               PowerPCFeatureFSqrt | PowerPCFeatureGPUL },
  { "generic", "Select instructions for a generic PowerPC processor", 0 }
};
/// Length of PowerPCSubTypeKV.
static const unsigned PowerPCSubTypeKVSize = sizeof(PowerPCSubTypeKV)
                                             / sizeof(SubtargetFeatureKV);

/// Sorted (by key) array of values for CPU features.
static SubtargetFeatureKV PowerPCFeatureKV[] = {
  { "64bit"  , "Should 64 bit instructions be used"  , PowerPCFeature64Bit   },
  { "64bitregs", "Should 64 bit registers be used"   , PowerPCFeature64BRegs },
  { "altivec", "Should Altivec instructions be used" , PowerPCFeatureAltivec },
  { "fsqrt"  , "Should the fsqrt instruction be used", PowerPCFeatureFSqrt   },
  { "gpul"   , "Should GPUL instructions be used"    , PowerPCFeatureGPUL    }
 };
/// Length of PowerPCFeatureKV.
static const unsigned PowerPCFeatureKVSize = sizeof(PowerPCFeatureKV)
                                          / sizeof(SubtargetFeatureKV);


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
  : StackAlignment(16), IsGigaProcessor(false), IsAIX(false), IsDarwin(false) {

  // Determine default and user specified characteristics
  std::string CPU = "generic";
#if defined(__APPLE__)
  CPU = GetCurrentPowerPCCPU();
#endif
  uint32_t Bits =
  SubtargetFeatures::Parse(FS, CPU,
                           PowerPCSubTypeKV, PowerPCSubTypeKVSize,
                           PowerPCFeatureKV, PowerPCFeatureKVSize);
  IsGigaProcessor = (Bits & PowerPCFeatureGPUL ) != 0;
  Is64Bit         = (Bits & PowerPCFeature64Bit) != 0;
  HasFSQRT        = (Bits & PowerPCFeatureFSqrt) != 0;
  Has64BitRegs    = (Bits & PowerPCFeature64BRegs) != 0;

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
