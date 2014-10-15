//===-- X86MCTargetDesc.cpp - X86 Target Descriptions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides X86 specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "X86MCTargetDesc.h"
#include "InstPrinter/X86ATTInstPrinter.h"
#include "InstPrinter/X86IntelInstPrinter.h"
#include "X86MCAsmInfo.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"

#if _MSC_VER
#include <intrin.h>
#endif

using namespace llvm;

#define GET_REGINFO_MC_DESC
#include "X86GenRegisterInfo.inc"

#define GET_INSTRINFO_MC_DESC
#include "X86GenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "X86GenSubtargetInfo.inc"

std::string X86_MC::ParseX86Triple(StringRef TT) {
  Triple TheTriple(TT);
  std::string FS;
  if (TheTriple.getArch() == Triple::x86_64)
    FS = "+64bit-mode,-32bit-mode,-16bit-mode";
  else if (TheTriple.getEnvironment() != Triple::CODE16)
    FS = "-64bit-mode,+32bit-mode,-16bit-mode";
  else
    FS = "-64bit-mode,-32bit-mode,+16bit-mode";

  return FS;
}

/// GetCpuIDAndInfo - Execute the specified cpuid and return the 4 values in the
/// specified arguments.  If we can't run cpuid on the host, return true.
bool X86_MC::GetCpuIDAndInfo(unsigned value, unsigned *rEAX,
                             unsigned *rEBX, unsigned *rECX, unsigned *rEDX) {
#if defined(__x86_64__) || defined(_M_AMD64) || defined (_M_X64)
  #if defined(__GNUC__)
    // gcc doesn't know cpuid would clobber ebx/rbx. Preseve it manually.
    asm ("movq\t%%rbx, %%rsi\n\t"
         "cpuid\n\t"
         "xchgq\t%%rbx, %%rsi\n\t"
         : "=a" (*rEAX),
           "=S" (*rEBX),
           "=c" (*rECX),
           "=d" (*rEDX)
         :  "a" (value));
    return false;
  #elif defined(_MSC_VER)
    int registers[4];
    __cpuid(registers, value);
    *rEAX = registers[0];
    *rEBX = registers[1];
    *rECX = registers[2];
    *rEDX = registers[3];
    return false;
  #else
    return true;
  #endif
#elif defined(i386) || defined(__i386__) || defined(__x86__) || defined(_M_IX86)
  #if defined(__GNUC__)
    asm ("movl\t%%ebx, %%esi\n\t"
         "cpuid\n\t"
         "xchgl\t%%ebx, %%esi\n\t"
         : "=a" (*rEAX),
           "=S" (*rEBX),
           "=c" (*rECX),
           "=d" (*rEDX)
         :  "a" (value));
    return false;
  #elif defined(_MSC_VER)
    __asm {
      mov   eax,value
      cpuid
      mov   esi,rEAX
      mov   dword ptr [esi],eax
      mov   esi,rEBX
      mov   dword ptr [esi],ebx
      mov   esi,rECX
      mov   dword ptr [esi],ecx
      mov   esi,rEDX
      mov   dword ptr [esi],edx
    }
    return false;
  #else
    return true;
  #endif
#else
  return true;
#endif
}

/// GetCpuIDAndInfoEx - Execute the specified cpuid with subleaf and return the
/// 4 values in the specified arguments.  If we can't run cpuid on the host,
/// return true.
bool X86_MC::GetCpuIDAndInfoEx(unsigned value, unsigned subleaf, unsigned *rEAX,
                               unsigned *rEBX, unsigned *rECX, unsigned *rEDX) {
#if defined(__x86_64__) || defined(_M_AMD64) || defined (_M_X64)
  #if defined(__GNUC__)
    // gcc desn't know cpuid would clobber ebx/rbx. Preseve it manually.
    asm ("movq\t%%rbx, %%rsi\n\t"
         "cpuid\n\t"
         "xchgq\t%%rbx, %%rsi\n\t"
         : "=a" (*rEAX),
           "=S" (*rEBX),
           "=c" (*rECX),
           "=d" (*rEDX)
         :  "a" (value),
            "c" (subleaf));
    return false;
  #elif defined(_MSC_VER)
    // __cpuidex was added in MSVC++ 9.0 SP1
    #if (_MSC_VER > 1500) || (_MSC_VER == 1500 && _MSC_FULL_VER >= 150030729)
      int registers[4];
      __cpuidex(registers, value, subleaf);
      *rEAX = registers[0];
      *rEBX = registers[1];
      *rECX = registers[2];
      *rEDX = registers[3];
      return false;
    #else
      return true;
    #endif
  #else
    return true;
  #endif
#elif defined(i386) || defined(__i386__) || defined(__x86__) || defined(_M_IX86)
  #if defined(__GNUC__)
    asm ("movl\t%%ebx, %%esi\n\t"
         "cpuid\n\t"
         "xchgl\t%%ebx, %%esi\n\t"
         : "=a" (*rEAX),
           "=S" (*rEBX),
           "=c" (*rECX),
           "=d" (*rEDX)
         :  "a" (value),
            "c" (subleaf));
    return false;
  #elif defined(_MSC_VER)
    __asm {
      mov   eax,value
      mov   ecx,subleaf
      cpuid
      mov   esi,rEAX
      mov   dword ptr [esi],eax
      mov   esi,rEBX
      mov   dword ptr [esi],ebx
      mov   esi,rECX
      mov   dword ptr [esi],ecx
      mov   esi,rEDX
      mov   dword ptr [esi],edx
    }
    return false;
  #else
    return true;
  #endif
#else
  return true;
#endif
}

void X86_MC::DetectFamilyModel(unsigned EAX, unsigned &Family,
                               unsigned &Model) {
  Family = (EAX >> 8) & 0xf; // Bits 8 - 11
  Model  = (EAX >> 4) & 0xf; // Bits 4 - 7
  if (Family == 6 || Family == 0xf) {
    if (Family == 0xf)
      // Examine extended family ID if family ID is F.
      Family += (EAX >> 20) & 0xff;    // Bits 20 - 27
    // Examine extended model ID if family ID is 6 or F.
    Model += ((EAX >> 16) & 0xf) << 4; // Bits 16 - 19
  }
}

unsigned X86_MC::getDwarfRegFlavour(Triple TT, bool isEH) {
  if (TT.getArch() == Triple::x86_64)
    return DWARFFlavour::X86_64;

  if (TT.isOSDarwin())
    return isEH ? DWARFFlavour::X86_32_DarwinEH : DWARFFlavour::X86_32_Generic;
  if (TT.isOSCygMing())
    // Unsupported by now, just quick fallback
    return DWARFFlavour::X86_32_Generic;
  return DWARFFlavour::X86_32_Generic;
}

void X86_MC::InitLLVM2SEHRegisterMapping(MCRegisterInfo *MRI) {
  // FIXME: TableGen these.
  for (unsigned Reg = X86::NoRegister+1; Reg < X86::NUM_TARGET_REGS; ++Reg) {
    unsigned SEH = MRI->getEncodingValue(Reg);
    MRI->mapLLVMRegToSEHReg(Reg, SEH);
  }
}

MCSubtargetInfo *X86_MC::createX86MCSubtargetInfo(StringRef TT, StringRef CPU,
                                                  StringRef FS) {
  std::string ArchFS = X86_MC::ParseX86Triple(TT);
  if (!FS.empty()) {
    if (!ArchFS.empty())
      ArchFS = ArchFS + "," + FS.str();
    else
      ArchFS = FS;
  }

  std::string CPUName = CPU;
  if (CPUName.empty())
    CPUName = "generic";

  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitX86MCSubtargetInfo(X, TT, CPUName, ArchFS);
  return X;
}

static MCInstrInfo *createX86MCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitX86MCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createX86MCRegisterInfo(StringRef TT) {
  Triple TheTriple(TT);
  unsigned RA = (TheTriple.getArch() == Triple::x86_64)
    ? X86::RIP     // Should have dwarf #16.
    : X86::EIP;    // Should have dwarf #8.

  MCRegisterInfo *X = new MCRegisterInfo();
  InitX86MCRegisterInfo(X, RA,
                        X86_MC::getDwarfRegFlavour(TheTriple, false),
                        X86_MC::getDwarfRegFlavour(TheTriple, true),
                        RA);
  X86_MC::InitLLVM2SEHRegisterMapping(X);
  return X;
}

static MCAsmInfo *createX86MCAsmInfo(const MCRegisterInfo &MRI, StringRef TT) {
  Triple TheTriple(TT);
  bool is64Bit = TheTriple.getArch() == Triple::x86_64;

  MCAsmInfo *MAI;
  if (TheTriple.isOSBinFormatMachO()) {
    if (is64Bit)
      MAI = new X86_64MCAsmInfoDarwin(TheTriple);
    else
      MAI = new X86MCAsmInfoDarwin(TheTriple);
  } else if (TheTriple.isOSBinFormatELF()) {
    // Force the use of an ELF container.
    MAI = new X86ELFMCAsmInfo(TheTriple);
  } else if (TheTriple.isWindowsMSVCEnvironment()) {
    MAI = new X86MCAsmInfoMicrosoft(TheTriple);
  } else if (TheTriple.isOSCygMing() ||
             TheTriple.isWindowsItaniumEnvironment()) {
    MAI = new X86MCAsmInfoGNUCOFF(TheTriple);
  } else {
    // The default is ELF.
    MAI = new X86ELFMCAsmInfo(TheTriple);
  }

  // Initialize initial frame state.
  // Calculate amount of bytes used for return address storing
  int stackGrowth = is64Bit ? -8 : -4;

  // Initial state of the frame pointer is esp+stackGrowth.
  unsigned StackPtr = is64Bit ? X86::RSP : X86::ESP;
  MCCFIInstruction Inst = MCCFIInstruction::createDefCfa(
      nullptr, MRI.getDwarfRegNum(StackPtr, true), -stackGrowth);
  MAI->addInitialFrameState(Inst);

  // Add return address to move list
  unsigned InstPtr = is64Bit ? X86::RIP : X86::EIP;
  MCCFIInstruction Inst2 = MCCFIInstruction::createOffset(
      nullptr, MRI.getDwarfRegNum(InstPtr, true), stackGrowth);
  MAI->addInitialFrameState(Inst2);

  return MAI;
}

static MCCodeGenInfo *createX86MCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                             CodeModel::Model CM,
                                             CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();

  Triple T(TT);
  bool is64Bit = T.getArch() == Triple::x86_64;

  if (RM == Reloc::Default) {
    // Darwin defaults to PIC in 64 bit mode and dynamic-no-pic in 32 bit mode.
    // Win64 requires rip-rel addressing, thus we force it to PIC. Otherwise we
    // use static relocation model by default.
    if (T.isOSDarwin()) {
      if (is64Bit)
        RM = Reloc::PIC_;
      else
        RM = Reloc::DynamicNoPIC;
    } else if (T.isOSWindows() && is64Bit)
      RM = Reloc::PIC_;
    else
      RM = Reloc::Static;
  }

  // ELF and X86-64 don't have a distinct DynamicNoPIC model.  DynamicNoPIC
  // is defined as a model for code which may be used in static or dynamic
  // executables but not necessarily a shared library. On X86-32 we just
  // compile in -static mode, in x86-64 we use PIC.
  if (RM == Reloc::DynamicNoPIC) {
    if (is64Bit)
      RM = Reloc::PIC_;
    else if (!T.isOSDarwin())
      RM = Reloc::Static;
  }

  // If we are on Darwin, disallow static relocation model in X86-64 mode, since
  // the Mach-O file format doesn't support it.
  if (RM == Reloc::Static && T.isOSDarwin() && is64Bit)
    RM = Reloc::PIC_;

  // For static codegen, if we're not already set, use Small codegen.
  if (CM == CodeModel::Default)
    CM = CodeModel::Small;
  else if (CM == CodeModel::JITDefault)
    // 64-bit JIT places everything in the same buffer except external funcs.
    CM = is64Bit ? CodeModel::Large : CodeModel::Small;

  X->InitMCCodeGenInfo(RM, CM, OL);
  return X;
}

static MCStreamer *createMCStreamer(const Target &T, StringRef TT,
                                    MCContext &Ctx, MCAsmBackend &MAB,
                                    raw_ostream &_OS, MCCodeEmitter *_Emitter,
                                    const MCSubtargetInfo &STI, bool RelaxAll) {
  Triple TheTriple(TT);

  switch (TheTriple.getObjectFormat()) {
  default: llvm_unreachable("unsupported object format");
  case Triple::MachO:
    return createMachOStreamer(Ctx, MAB, _OS, _Emitter, RelaxAll);
  case Triple::COFF:
    assert(TheTriple.isOSWindows() && "only Windows COFF is supported");
    return createX86WinCOFFStreamer(Ctx, MAB, _Emitter, _OS, RelaxAll);
  case Triple::ELF:
    return createELFStreamer(Ctx, MAB, _OS, _Emitter, RelaxAll);
  }
}

static MCInstPrinter *createX86MCInstPrinter(const Target &T,
                                             unsigned SyntaxVariant,
                                             const MCAsmInfo &MAI,
                                             const MCInstrInfo &MII,
                                             const MCRegisterInfo &MRI,
                                             const MCSubtargetInfo &STI) {
  if (SyntaxVariant == 0)
    return new X86ATTInstPrinter(MAI, MII, MRI, STI);
  if (SyntaxVariant == 1)
    return new X86IntelInstPrinter(MAI, MII, MRI);
  return nullptr;
}

static MCRelocationInfo *createX86MCRelocationInfo(StringRef TT,
                                                   MCContext &Ctx) {
  Triple TheTriple(TT);
  if (TheTriple.isOSBinFormatMachO() && TheTriple.getArch() == Triple::x86_64)
    return createX86_64MachORelocationInfo(Ctx);
  else if (TheTriple.isOSBinFormatELF())
    return createX86_64ELFRelocationInfo(Ctx);
  // Default to the stock relocation info.
  return llvm::createMCRelocationInfo(TT, Ctx);
}

static MCInstrAnalysis *createX86MCInstrAnalysis(const MCInstrInfo *Info) {
  return new MCInstrAnalysis(Info);
}

// Force static initialization.
extern "C" void LLVMInitializeX86TargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfoFn A(TheX86_32Target, createX86MCAsmInfo);
  RegisterMCAsmInfoFn B(TheX86_64Target, createX86MCAsmInfo);

  // Register the MC codegen info.
  RegisterMCCodeGenInfoFn C(TheX86_32Target, createX86MCCodeGenInfo);
  RegisterMCCodeGenInfoFn D(TheX86_64Target, createX86MCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheX86_32Target, createX86MCInstrInfo);
  TargetRegistry::RegisterMCInstrInfo(TheX86_64Target, createX86MCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheX86_32Target, createX86MCRegisterInfo);
  TargetRegistry::RegisterMCRegInfo(TheX86_64Target, createX86MCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheX86_32Target,
                                          X86_MC::createX86MCSubtargetInfo);
  TargetRegistry::RegisterMCSubtargetInfo(TheX86_64Target,
                                          X86_MC::createX86MCSubtargetInfo);

  // Register the MC instruction analyzer.
  TargetRegistry::RegisterMCInstrAnalysis(TheX86_32Target,
                                          createX86MCInstrAnalysis);
  TargetRegistry::RegisterMCInstrAnalysis(TheX86_64Target,
                                          createX86MCInstrAnalysis);

  // Register the code emitter.
  TargetRegistry::RegisterMCCodeEmitter(TheX86_32Target,
                                        createX86MCCodeEmitter);
  TargetRegistry::RegisterMCCodeEmitter(TheX86_64Target,
                                        createX86MCCodeEmitter);

  // Register the asm backend.
  TargetRegistry::RegisterMCAsmBackend(TheX86_32Target,
                                       createX86_32AsmBackend);
  TargetRegistry::RegisterMCAsmBackend(TheX86_64Target,
                                       createX86_64AsmBackend);

  // Register the object streamer.
  TargetRegistry::RegisterMCObjectStreamer(TheX86_32Target,
                                           createMCStreamer);
  TargetRegistry::RegisterMCObjectStreamer(TheX86_64Target,
                                           createMCStreamer);

  // Register the MCInstPrinter.
  TargetRegistry::RegisterMCInstPrinter(TheX86_32Target,
                                        createX86MCInstPrinter);
  TargetRegistry::RegisterMCInstPrinter(TheX86_64Target,
                                        createX86MCInstPrinter);

  // Register the MC relocation info.
  TargetRegistry::RegisterMCRelocationInfo(TheX86_32Target,
                                           createX86MCRelocationInfo);
  TargetRegistry::RegisterMCRelocationInfo(TheX86_64Target,
                                           createX86MCRelocationInfo);
}
