//===-- PowerPCSubtarget.cpp - PPC Subtarget Information ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PPC specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "PPCSubtarget.h"
#include "PPCRegisterInfo.h"
#include "PPC.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/GlobalValue.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/DataStream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetRegistry.h"
#include <cstdlib>

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "PPCGenSubtargetInfo.inc"

using namespace llvm;

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
#elif defined(__linux__) && (defined(__ppc__) || defined(__powerpc__))
static const char *GetCurrentPowerPCCPU() {
  // Access to the Processor Version Register (PVR) on PowerPC is privileged,
  // and so we must use an operating-system interface to determine the current
  // processor type. On Linux, this is exposed through the /proc/cpuinfo file.
  const char *generic = "generic";

  // Note: We cannot mmap /proc/cpuinfo here and then process the resulting
  // memory buffer because the 'file' has 0 size (it can be read from only
  // as a stream).

  std::string Err;
  DataStreamer *DS = getDataFileStreamer("/proc/cpuinfo", &Err);
  if (!DS) {
    DEBUG(dbgs() << "Unable to open /proc/cpuinfo: " << Err << "\n");
    return generic;
  }

  // The cpu line is second (after the 'processor: 0' line), so if this
  // buffer is too small then something has changed (or is wrong).
  char buffer[1024];
  size_t CPUInfoSize = DS->GetBytes((unsigned char*) buffer, sizeof(buffer));
  delete DS;

  const char *CPUInfoStart = buffer;
  const char *CPUInfoEnd = buffer + CPUInfoSize;

  const char *CIP = CPUInfoStart;

  const char *CPUStart = 0;
  size_t CPULen = 0;

  // We need to find the first line which starts with cpu, spaces, and a colon.
  // After the colon, there may be some additional spaces and then the cpu type.
  while (CIP < CPUInfoEnd && CPUStart == 0) {
    if (CIP < CPUInfoEnd && *CIP == '\n')
      ++CIP;

    if (CIP < CPUInfoEnd && *CIP == 'c') {
      ++CIP;
      if (CIP < CPUInfoEnd && *CIP == 'p') {
        ++CIP;
        if (CIP < CPUInfoEnd && *CIP == 'u') {
          ++CIP;
          while (CIP < CPUInfoEnd && (*CIP == ' ' || *CIP == '\t'))
            ++CIP;
  
          if (CIP < CPUInfoEnd && *CIP == ':') {
            ++CIP;
            while (CIP < CPUInfoEnd && (*CIP == ' ' || *CIP == '\t'))
              ++CIP;
  
            if (CIP < CPUInfoEnd) {
              CPUStart = CIP;
              while (CIP < CPUInfoEnd && (*CIP != ' ' && *CIP != '\t' &&
                                          *CIP != ',' && *CIP != '\n'))
                ++CIP;
              CPULen = CIP - CPUStart;
            }
          }
        }
      }
    }

    if (CPUStart == 0)
      while (CIP < CPUInfoEnd && *CIP != '\n')
        ++CIP;
  }

  if (CPUStart == 0)
    return generic;

  return StringSwitch<const char *>(StringRef(CPUStart, CPULen))
    .Case("604e", "604e")
    .Case("604", "604")
    .Case("7400", "7400")
    .Case("7410", "7400")
    .Case("7447", "7400")
    .Case("7455", "7450")
    .Case("G4", "g4")
    .Case("POWER4", "g4")
    .Case("PPC970FX", "970")
    .Case("PPC970MP", "970")
    .Case("G5", "g5")
    .Case("POWER5", "g5")
    .Case("POWER6", "pwr6")
    .Case("POWER7", "pwr7")
    .Default(generic);
}
#endif


PPCSubtarget::PPCSubtarget(const std::string &TT, const std::string &CPU,
                           const std::string &FS, bool is64Bit)
  : PPCGenSubtargetInfo(TT, CPU, FS)
  , StackAlignment(16)
  , DarwinDirective(PPC::DIR_NONE)
  , IsGigaProcessor(false)
  , Has64BitSupport(false)
  , Use64BitRegs(false)
  , IsPPC64(is64Bit)
  , HasAltivec(false)
  , HasFSQRT(false)
  , HasSTFIWX(false)
  , IsBookE(false)
  , HasLazyResolverStubs(false)
  , IsJITCodeModel(false)
  , TargetTriple(TT) {

  // Determine default and user specified characteristics
  std::string CPUName = CPU;
  if (CPUName.empty())
    CPUName = "generic";
#if defined(__APPLE__) || \
      (defined(__linux__) && (defined(__ppc__) || defined(__powerpc__)))
  if (CPUName == "generic")
    CPUName = GetCurrentPowerPCCPU();
#endif

  // Parse features string.
  ParseSubtargetFeatures(CPUName, FS);

  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(CPUName);

  // If we are generating code for ppc64, verify that options make sense.
  if (is64Bit) {
    Has64BitSupport = true;
    // Silently force 64-bit register use on ppc64.
    Use64BitRegs = true;
  }
  
  // If the user requested use of 64-bit regs, but the cpu selected doesn't
  // support it, ignore.
  if (use64BitRegs() && !has64BitSupport())
    Use64BitRegs = false;

  // Set up darwin-specific properties.
  if (isDarwin())
    HasLazyResolverStubs = true;
}

/// SetJITMode - This is called to inform the subtarget info that we are
/// producing code for the JIT.
void PPCSubtarget::SetJITMode() {
  // JIT mode doesn't want lazy resolver stubs, it knows exactly where
  // everything is.  This matters for PPC64, which codegens in PIC mode without
  // stubs.
  HasLazyResolverStubs = false;

  // Calls to external functions need to use indirect calls
  IsJITCodeModel = true;
}


/// hasLazyResolverStub - Return true if accesses to the specified global have
/// to go through a dyld lazy resolution stub.  This means that an extra load
/// is required to get the address of the global.
bool PPCSubtarget::hasLazyResolverStub(const GlobalValue *GV,
                                       const TargetMachine &TM) const {
  // We never have stubs if HasLazyResolverStubs=false or if in static mode.
  if (!HasLazyResolverStubs || TM.getRelocationModel() == Reloc::Static)
    return false;
  // If symbol visibility is hidden, the extra load is not needed if
  // the symbol is definitely defined in the current translation unit.
  bool isDecl = GV->isDeclaration() && !GV->isMaterializable();
  if (GV->hasHiddenVisibility() && !isDecl && !GV->hasCommonLinkage())
    return false;
  return GV->hasWeakLinkage() || GV->hasLinkOnceLinkage() ||
         GV->hasCommonLinkage() || isDecl;
}

bool PPCSubtarget::enablePostRAScheduler(
           CodeGenOpt::Level OptLevel,
           TargetSubtargetInfo::AntiDepBreakMode& Mode,
           RegClassVector& CriticalPathRCs) const {
  // FIXME: It would be best to use TargetSubtargetInfo::ANTIDEP_ALL here,
  // but we can't because we can't reassign the cr registers. There is a
  // dependence between the cr register and the RLWINM instruction used
  // to extract its value which the anti-dependency breaker can't currently
  // see. Maybe we should make a late-expanded pseudo to encode this dependency.
  // (the relevant code is in PPCDAGToDAGISel::SelectSETCC)

  Mode = TargetSubtargetInfo::ANTIDEP_CRITICAL;

  CriticalPathRCs.clear();

  if (isPPC64())
    CriticalPathRCs.push_back(&PPC::G8RCRegClass);
  else
    CriticalPathRCs.push_back(&PPC::GPRCRegClass);
    
  CriticalPathRCs.push_back(&PPC::F8RCRegClass);
  CriticalPathRCs.push_back(&PPC::VRRCRegClass);

  return OptLevel >= CodeGenOpt::Default;
}

