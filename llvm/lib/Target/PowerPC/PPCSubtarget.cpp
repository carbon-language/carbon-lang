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
#include "PPC.h"
#include "PPCRegisterInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include <cstdlib>

using namespace llvm;

#define DEBUG_TYPE "ppc-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "PPCGenSubtargetInfo.inc"

/// Return the datalayout string of a subtarget.
static std::string getDataLayoutString(const PPCSubtarget &ST) {
  const Triple &T = ST.getTargetTriple();

  std::string Ret;

  // Most PPC* platforms are big endian, PPC64LE is little endian.
  if (ST.isLittleEndian())
    Ret = "e";
  else
    Ret = "E";

  Ret += DataLayout::getManglingComponent(T);

  // PPC32 has 32 bit pointers. The PS3 (OS Lv2) is a PPC64 machine with 32 bit
  // pointers.
  if (!ST.isPPC64() || T.getOS() == Triple::Lv2)
    Ret += "-p:32:32";

  // Note, the alignment values for f64 and i64 on ppc64 in Darwin
  // documentation are wrong; these are correct (i.e. "what gcc does").
  if (ST.isPPC64() || ST.isSVR4ABI())
    Ret += "-i64:64";
  else
    Ret += "-f64:32:64";

  // PPC64 has 32 and 64 bit registers, PPC32 has only 32 bit ones.
  if (ST.isPPC64())
    Ret += "-n32:64";
  else
    Ret += "-n32";

  return Ret;
}

PPCSubtarget &PPCSubtarget::initializeSubtargetDependencies(StringRef CPU,
                                                            StringRef FS) {
  initializeEnvironment();
  resetSubtargetFeatures(CPU, FS);
  return *this;
}

PPCSubtarget::PPCSubtarget(const std::string &TT, const std::string &CPU,
                           const std::string &FS, PPCTargetMachine &TM,
                           bool is64Bit, CodeGenOpt::Level OptLevel)
    : PPCGenSubtargetInfo(TT, CPU, FS), IsPPC64(is64Bit), TargetTriple(TT),
      OptLevel(OptLevel),
      FrameLowering(initializeSubtargetDependencies(CPU, FS)),
      DL(getDataLayoutString(*this)), InstrInfo(*this), JITInfo(*this),
      TLInfo(TM), TSInfo(&DL) {}

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

void PPCSubtarget::resetSubtargetFeatures(const MachineFunction *MF) {
  AttributeSet FnAttrs = MF->getFunction()->getAttributes();
  Attribute CPUAttr = FnAttrs.getAttribute(AttributeSet::FunctionIndex,
                                           "target-cpu");
  Attribute FSAttr = FnAttrs.getAttribute(AttributeSet::FunctionIndex,
                                          "target-features");
  std::string CPU =
    !CPUAttr.hasAttribute(Attribute::None) ? CPUAttr.getValueAsString() : "";
  std::string FS =
    !FSAttr.hasAttribute(Attribute::None) ? FSAttr.getValueAsString() : "";
  if (!FS.empty()) {
    initializeEnvironment();
    resetSubtargetFeatures(CPU, FS);
  }
}

void PPCSubtarget::initializeEnvironment() {
  StackAlignment = 16;
  DarwinDirective = PPC::DIR_NONE;
  HasMFOCRF = false;
  Has64BitSupport = false;
  Use64BitRegs = false;
  UseCRBits = false;
  HasAltivec = false;
  HasQPX = false;
  HasVSX = false;
  HasFCPSGN = false;
  HasFSQRT = false;
  HasFRE = false;
  HasFRES = false;
  HasFRSQRTE = false;
  HasFRSQRTES = false;
  HasRecipPrec = false;
  HasSTFIWX = false;
  HasLFIWAX = false;
  HasFPRND = false;
  HasFPCVT = false;
  HasISEL = false;
  HasPOPCNTD = false;
  HasLDBRX = false;
  IsBookE = false;
  DeprecatedMFTB = false;
  DeprecatedDST = false;
  HasLazyResolverStubs = false;
  IsJITCodeModel = false;
}

void PPCSubtarget::resetSubtargetFeatures(StringRef CPU, StringRef FS) {
  // Determine default and user specified characteristics
  std::string CPUName = CPU;
  if (CPUName.empty())
    CPUName = "generic";
#if (defined(__APPLE__) || defined(__linux__)) && \
    (defined(__ppc__) || defined(__powerpc__))
  if (CPUName == "generic")
    CPUName = sys::getHostCPUName();
#endif

  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(CPUName);

  // Make sure 64-bit features are available when CPUname is generic
  std::string FullFS = FS;

  // If we are generating code for ppc64, verify that options make sense.
  if (IsPPC64) {
    Has64BitSupport = true;
    // Silently force 64-bit register use on ppc64.
    Use64BitRegs = true;
    if (!FullFS.empty())
      FullFS = "+64bit," + FullFS;
    else
      FullFS = "+64bit";
  }

  // At -O2 and above, track CR bits as individual registers.
  if (OptLevel >= CodeGenOpt::Default) {
    if (!FullFS.empty())
      FullFS = "+crbits," + FullFS;
    else
      FullFS = "+crbits";
  }

  // Parse features string.
  ParseSubtargetFeatures(CPUName, FullFS);

  // If the user requested use of 64-bit regs, but the cpu selected doesn't
  // support it, ignore.
  if (use64BitRegs() && !has64BitSupport())
    Use64BitRegs = false;

  // Set up darwin-specific properties.
  if (isDarwin())
    HasLazyResolverStubs = true;

  // QPX requires a 32-byte aligned stack. Note that we need to do this if
  // we're compiling for a BG/Q system regardless of whether or not QPX
  // is enabled because external functions will assume this alignment.
  if (hasQPX() || isBGQ())
    StackAlignment = 32;

  // Determine endianness.
  IsLittleEndian = (TargetTriple.getArch() == Triple::ppc64le);

  // FIXME: For now, we disable VSX in little-endian mode until endian
  // issues in those instructions can be addressed.
  if (IsLittleEndian)
    HasVSX = false;
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

// Embedded cores need aggressive scheduling (and some others also benefit).
static bool needsAggressiveScheduling(unsigned Directive) {
  switch (Directive) {
  default: return false;
  case PPC::DIR_440:
  case PPC::DIR_A2:
  case PPC::DIR_E500mc:
  case PPC::DIR_E5500:
  case PPC::DIR_PWR7:
  case PPC::DIR_PWR8:
    return true;
  }
}

bool PPCSubtarget::enableMachineScheduler() const {
  // Enable MI scheduling for the embedded cores.
  // FIXME: Enable this for all cores (some additional modeling
  // may be necessary).
  return needsAggressiveScheduling(DarwinDirective);
}

// This overrides the PostRAScheduler bit in the SchedModel for each CPU.
bool PPCSubtarget::enablePostMachineScheduler() const { return true; }

PPCGenSubtargetInfo::AntiDepBreakMode PPCSubtarget::getAntiDepBreakMode() const {
  return TargetSubtargetInfo::ANTIDEP_ALL;
}

void PPCSubtarget::getCriticalPathRCs(RegClassVector &CriticalPathRCs) const {
  CriticalPathRCs.clear();
  CriticalPathRCs.push_back(isPPC64() ?
                            &PPC::G8RCRegClass : &PPC::GPRCRegClass);
}

void PPCSubtarget::overrideSchedPolicy(MachineSchedPolicy &Policy,
                                       MachineInstr *begin,
                                       MachineInstr *end,
                                       unsigned NumRegionInstrs) const {
  if (needsAggressiveScheduling(DarwinDirective)) {
    Policy.OnlyTopDown = false;
    Policy.OnlyBottomUp = false;
  }

  // Spilling is generally expensive on all PPC cores, so always enable
  // register-pressure tracking.
  Policy.ShouldTrackPressure = true;
}

bool PPCSubtarget::useAA() const {
  // Use AA during code generation for the embedded cores.
  return needsAggressiveScheduling(DarwinDirective);
}

