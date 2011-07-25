//===-- X86MCTargetDesc.cpp - X86 Target Descriptions -----------*- C++ -*-===//
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
#include "X86MCAsmInfo.h"
#include "InstPrinter/X86ATTInstPrinter.h"
#include "InstPrinter/X86IntelInstPrinter.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Host.h"

#define GET_REGINFO_MC_DESC
#include "X86GenRegisterInfo.inc"

#define GET_INSTRINFO_MC_DESC
#include "X86GenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "X86GenSubtargetInfo.inc"

using namespace llvm;


std::string X86_MC::ParseX86Triple(StringRef TT) {
  Triple TheTriple(TT);
  if (TheTriple.getArch() == Triple::x86_64)
    return "+64bit-mode";
  return "-64bit-mode";
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
  #endif
#endif
  return true;
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

unsigned X86_MC::getDwarfRegFlavour(StringRef TT, bool isEH) {
  Triple TheTriple(TT);
  if (TheTriple.getArch() == Triple::x86_64)
    return DWARFFlavour::X86_64;

  if (TheTriple.isOSDarwin())
    return isEH ? DWARFFlavour::X86_32_DarwinEH : DWARFFlavour::X86_32_Generic;
  if (TheTriple.getOS() == Triple::MinGW32 ||
      TheTriple.getOS() == Triple::Cygwin)
    // Unsupported by now, just quick fallback
    return DWARFFlavour::X86_32_Generic;
  return DWARFFlavour::X86_32_Generic;
}

/// getX86RegNum - This function maps LLVM register identifiers to their X86
/// specific numbering, which is used in various places encoding instructions.
unsigned X86_MC::getX86RegNum(unsigned RegNo) {
  switch(RegNo) {
  case X86::RAX: case X86::EAX: case X86::AX: case X86::AL: return N86::EAX;
  case X86::RCX: case X86::ECX: case X86::CX: case X86::CL: return N86::ECX;
  case X86::RDX: case X86::EDX: case X86::DX: case X86::DL: return N86::EDX;
  case X86::RBX: case X86::EBX: case X86::BX: case X86::BL: return N86::EBX;
  case X86::RSP: case X86::ESP: case X86::SP: case X86::SPL: case X86::AH:
    return N86::ESP;
  case X86::RBP: case X86::EBP: case X86::BP: case X86::BPL: case X86::CH:
    return N86::EBP;
  case X86::RSI: case X86::ESI: case X86::SI: case X86::SIL: case X86::DH:
    return N86::ESI;
  case X86::RDI: case X86::EDI: case X86::DI: case X86::DIL: case X86::BH:
    return N86::EDI;

  case X86::R8:  case X86::R8D:  case X86::R8W:  case X86::R8B:
    return N86::EAX;
  case X86::R9:  case X86::R9D:  case X86::R9W:  case X86::R9B:
    return N86::ECX;
  case X86::R10: case X86::R10D: case X86::R10W: case X86::R10B:
    return N86::EDX;
  case X86::R11: case X86::R11D: case X86::R11W: case X86::R11B:
    return N86::EBX;
  case X86::R12: case X86::R12D: case X86::R12W: case X86::R12B:
    return N86::ESP;
  case X86::R13: case X86::R13D: case X86::R13W: case X86::R13B:
    return N86::EBP;
  case X86::R14: case X86::R14D: case X86::R14W: case X86::R14B:
    return N86::ESI;
  case X86::R15: case X86::R15D: case X86::R15W: case X86::R15B:
    return N86::EDI;

  case X86::ST0: case X86::ST1: case X86::ST2: case X86::ST3:
  case X86::ST4: case X86::ST5: case X86::ST6: case X86::ST7:
    return RegNo-X86::ST0;

  case X86::XMM0: case X86::XMM8:
  case X86::YMM0: case X86::YMM8: case X86::MM0:
    return 0;
  case X86::XMM1: case X86::XMM9:
  case X86::YMM1: case X86::YMM9: case X86::MM1:
    return 1;
  case X86::XMM2: case X86::XMM10:
  case X86::YMM2: case X86::YMM10: case X86::MM2:
    return 2;
  case X86::XMM3: case X86::XMM11:
  case X86::YMM3: case X86::YMM11: case X86::MM3:
    return 3;
  case X86::XMM4: case X86::XMM12:
  case X86::YMM4: case X86::YMM12: case X86::MM4:
    return 4;
  case X86::XMM5: case X86::XMM13:
  case X86::YMM5: case X86::YMM13: case X86::MM5:
    return 5;
  case X86::XMM6: case X86::XMM14:
  case X86::YMM6: case X86::YMM14: case X86::MM6:
    return 6;
  case X86::XMM7: case X86::XMM15:
  case X86::YMM7: case X86::YMM15: case X86::MM7:
    return 7;

  case X86::ES: return 0;
  case X86::CS: return 1;
  case X86::SS: return 2;
  case X86::DS: return 3;
  case X86::FS: return 4;
  case X86::GS: return 5;

  case X86::CR0: case X86::CR8 : case X86::DR0: return 0;
  case X86::CR1: case X86::CR9 : case X86::DR1: return 1;
  case X86::CR2: case X86::CR10: case X86::DR2: return 2;
  case X86::CR3: case X86::CR11: case X86::DR3: return 3;
  case X86::CR4: case X86::CR12: case X86::DR4: return 4;
  case X86::CR5: case X86::CR13: case X86::DR5: return 5;
  case X86::CR6: case X86::CR14: case X86::DR6: return 6;
  case X86::CR7: case X86::CR15: case X86::DR7: return 7;

  // Pseudo index registers are equivalent to a "none"
  // scaled index (See Intel Manual 2A, table 2-3)
  case X86::EIZ:
  case X86::RIZ:
    return 4;

  default:
    assert((int(RegNo) > 0) && "Unknown physical register!");
    return 0;
  }
}

void X86_MC::InitLLVM2SEHRegisterMapping(MCRegisterInfo *MRI) {
  // FIXME: TableGen these.
  for (unsigned Reg = X86::NoRegister+1; Reg < X86::NUM_TARGET_REGS; ++Reg) {
    int SEH = X86_MC::getX86RegNum(Reg);
    switch (Reg) {
    case X86::R8:  case X86::R8D:  case X86::R8W:  case X86::R8B:
    case X86::R9:  case X86::R9D:  case X86::R9W:  case X86::R9B:
    case X86::R10: case X86::R10D: case X86::R10W: case X86::R10B:
    case X86::R11: case X86::R11D: case X86::R11W: case X86::R11B:
    case X86::R12: case X86::R12D: case X86::R12W: case X86::R12B:
    case X86::R13: case X86::R13D: case X86::R13W: case X86::R13B:
    case X86::R14: case X86::R14D: case X86::R14W: case X86::R14B:
    case X86::R15: case X86::R15D: case X86::R15W: case X86::R15B:
    case X86::XMM8:  case X86::XMM9:  case X86::XMM10: case X86::XMM11:
    case X86::XMM12: case X86::XMM13: case X86::XMM14: case X86::XMM15:
    case X86::YMM8:  case X86::YMM9:  case X86::YMM10: case X86::YMM11:
    case X86::YMM12: case X86::YMM13: case X86::YMM14: case X86::YMM15:
      SEH += 8;
      break;
    }
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
  if (CPUName.empty()) {
#if defined (__x86_64__) || defined(__i386__)
    CPUName = sys::getHostCPUName();
#else
    CPUName = "generic";
#endif
  }

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
                        X86_MC::getDwarfRegFlavour(TT, false),
                        X86_MC::getDwarfRegFlavour(TT, true));
  X86_MC::InitLLVM2SEHRegisterMapping(X);
  return X;
}

static MCAsmInfo *createX86MCAsmInfo(const Target &T, StringRef TT) {
  Triple TheTriple(TT);
  bool is64Bit = TheTriple.getArch() == Triple::x86_64;

  MCAsmInfo *MAI;
  if (TheTriple.isOSDarwin() || TheTriple.getEnvironment() == Triple::MachO) {
    if (is64Bit)
      MAI = new X86_64MCAsmInfoDarwin(TheTriple);
    else
      MAI = new X86MCAsmInfoDarwin(TheTriple);
  } else if (TheTriple.isOSWindows()) {
    MAI = new X86MCAsmInfoCOFF(TheTriple);
  } else {
    MAI = new X86ELFMCAsmInfo(TheTriple);
  }

  // Initialize initial frame state.
  // Calculate amount of bytes used for return address storing
  int stackGrowth = is64Bit ? -8 : -4;

  // Initial state of the frame pointer is esp+stackGrowth.
  MachineLocation Dst(MachineLocation::VirtualFP);
  MachineLocation Src(is64Bit ? X86::RSP : X86::ESP, stackGrowth);
  MAI->addInitialFrameState(0, Dst, Src);

  // Add return address to move list
  MachineLocation CSDst(is64Bit ? X86::RSP : X86::ESP, stackGrowth);
  MachineLocation CSSrc(is64Bit ? X86::RIP : X86::EIP);
  MAI->addInitialFrameState(0, CSDst, CSSrc);

  return MAI;
}

static MCCodeGenInfo *createX86MCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                             CodeModel::Model CM) {
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

  X->InitMCCodeGenInfo(RM, CM);
  return X;
}

static MCStreamer *createMCStreamer(const Target &T, const std::string &TT,
                                    MCContext &Ctx, TargetAsmBackend &TAB,
                                    raw_ostream &_OS,
                                    MCCodeEmitter *_Emitter,
                                    bool RelaxAll,
                                    bool NoExecStack) {
  Triple TheTriple(TT);

  if (TheTriple.isOSDarwin() || TheTriple.getEnvironment() == Triple::MachO)
    return createMachOStreamer(Ctx, TAB, _OS, _Emitter, RelaxAll);

  if (TheTriple.isOSWindows())
    return createWinCOFFStreamer(Ctx, TAB, *_Emitter, _OS, RelaxAll);

  return createELFStreamer(Ctx, TAB, _OS, _Emitter, RelaxAll, NoExecStack);
}

static MCInstPrinter *createX86MCInstPrinter(const Target &T,
                                             unsigned SyntaxVariant,
                                             const MCAsmInfo &MAI) {
  if (SyntaxVariant == 0)
    return new X86ATTInstPrinter(MAI);
  if (SyntaxVariant == 1)
    return new X86IntelInstPrinter(MAI);
  return 0;
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

  // Register the code emitter.
  TargetRegistry::RegisterCodeEmitter(TheX86_32Target,
                                      createX86MCCodeEmitter);
  TargetRegistry::RegisterCodeEmitter(TheX86_64Target,
                                      createX86MCCodeEmitter);

  // Register the asm backend.
  TargetRegistry::RegisterAsmBackend(TheX86_32Target,
                                     createX86_32AsmBackend);
  TargetRegistry::RegisterAsmBackend(TheX86_64Target,
                                     createX86_64AsmBackend);

  // Register the object streamer.
  TargetRegistry::RegisterObjectStreamer(TheX86_32Target,
                                         createMCStreamer);
  TargetRegistry::RegisterObjectStreamer(TheX86_64Target,
                                         createMCStreamer);

  // Register the MCInstPrinter.
  TargetRegistry::RegisterMCInstPrinter(TheX86_32Target,
                                        createX86MCInstPrinter);
  TargetRegistry::RegisterMCInstPrinter(TheX86_64Target,
                                        createX86MCInstPrinter);
}
