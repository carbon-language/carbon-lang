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

#include "PowerPCSubtarget.h"
#include "PowerPC.h"
#include "llvm/Module.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;
PPCTargetEnum llvm::PPCTarget = TargetDefault;

namespace llvm {
  cl::opt<PPCTargetEnum, true>
  PPCTargetArg(cl::desc("Force generation of code for a specific PPC target:"),
               cl::values(
                          clEnumValN(TargetAIX,  "aix", "  Enable AIX codegen"),
                          clEnumValN(TargetDarwin,"darwin","  Enable Darwin codegen"),
                          clEnumValEnd),
               cl::location(PPCTarget), cl::init(TargetDefault));
  cl::opt<bool> EnableGPOPT("enable-gpopt", cl::Hidden,
                             cl::desc("Enable optimizations for GP cpus"));
}

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/host_info.h>
#include <mach/machine.h>

static boolean_t IsGP() {
  host_basic_info_data_t hostInfo;
  mach_msg_type_number_t infoCount;

  infoCount = HOST_BASIC_INFO_COUNT;
  host_info(mach_host_self(), HOST_BASIC_INFO, (host_info_t)&hostInfo, 
            &infoCount);
  
  return ((hostInfo.cpu_type == CPU_TYPE_POWERPC) &&
          (hostInfo.cpu_subtype == CPU_SUBTYPE_POWERPC_970));
} 
#endif

PPCSubtarget::PPCSubtarget(const Module &M)
  : StackAlignment(16), IsGigaProcessor(false), IsAIX(false), IsDarwin(false) {

    // Set the boolean corresponding to the current target triple, or the default
  // if one cannot be determined, to true.
  const std::string& TT = M.getTargetTriple();
  if (TT.length() > 5) {
    IsDarwin = TT.find("darwin") != std::string::npos;
#if defined(__APPLE__)
    IsGigaProcessor = IsGP();
#endif
  } else if (TT.empty()) {
#if defined(_POWER)
    IsAIX = true;
#elif defined(__APPLE__)
    IsDarwin = true;
    IsGigaProcessor = IsGP();
#endif
  }
  
  // If GP opts are forced on by the commandline, do so now.
  if (EnableGPOPT) IsGigaProcessor = true;
}
