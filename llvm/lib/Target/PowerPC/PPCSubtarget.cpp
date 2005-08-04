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
#include "llvm/Module.h"

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

using namespace llvm;

PPCSubtarget::PPCSubtarget(const Module &M)
  : TargetSubtarget(), stackAlignment(16), isGigaProcessor(false), isAIX(false),
    isDarwin(false) {
  // Set the boolean corresponding to the current target triple, or the default
  // if one cannot be determined, to true.
  const std::string& TT = M.getTargetTriple();
  if (TT.length() > 5) {
    isDarwin = TT.find("darwin") != std::string::npos;
  } else if (TT.empty()) {
#if defined(_POWER)
    isAIX = true;
#elif defined(__APPLE__)
    isDarwin = true;
    isGigaProcessor = IsGP();
#endif
  }
}
