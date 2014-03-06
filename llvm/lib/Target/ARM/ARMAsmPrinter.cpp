//===-- ARMAsmPrinter.cpp - Print machine code to an ARM .s file ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format ARM assembly language.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "ARMAsmPrinter.h"
#include "ARM.h"
#include "ARMConstantPoolValue.h"
#include "ARMFPUName.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMTargetMachine.h"
#include "ARMTargetObjectFile.h"
#include "InstPrinter/ARMInstPrinter.h"
#include "MCTargetDesc/ARMAddressingModes.h"
#include "MCTargetDesc/ARMMCExpr.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/ARMBuildAttributes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <cctype>
using namespace llvm;

void ARMAsmPrinter::EmitFunctionBodyEnd() {
  // Make sure to terminate any constant pools that were at the end
  // of the function.
  if (!InConstantPool)
    return;
  InConstantPool = false;
  OutStreamer.EmitDataRegion(MCDR_DataRegionEnd);
}

void ARMAsmPrinter::EmitFunctionEntryLabel() {
  if (AFI->isThumbFunction()) {
    OutStreamer.EmitAssemblerFlag(MCAF_Code16);
    OutStreamer.EmitThumbFunc(CurrentFnSym);
  }

  OutStreamer.EmitLabel(CurrentFnSym);
}

void ARMAsmPrinter::EmitXXStructor(const Constant *CV) {
  uint64_t Size = TM.getDataLayout()->getTypeAllocSize(CV->getType());
  assert(Size && "C++ constructor pointer had zero size!");

  const GlobalValue *GV = dyn_cast<GlobalValue>(CV->stripPointerCasts());
  assert(GV && "C++ constructor pointer was not a GlobalValue!");

  const MCExpr *E = MCSymbolRefExpr::Create(getSymbol(GV),
                                            (Subtarget->isTargetELF()
                                             ? MCSymbolRefExpr::VK_ARM_TARGET1
                                             : MCSymbolRefExpr::VK_None),
                                            OutContext);
  
  OutStreamer.EmitValue(E, Size);
}

/// runOnMachineFunction - This uses the EmitInstruction()
/// method to print assembly for each instruction.
///
bool ARMAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  AFI = MF.getInfo<ARMFunctionInfo>();
  MCP = MF.getConstantPool();

  return AsmPrinter::runOnMachineFunction(MF);
}

void ARMAsmPrinter::printOperand(const MachineInstr *MI, int OpNum,
                                 raw_ostream &O, const char *Modifier) {
  const MachineOperand &MO = MI->getOperand(OpNum);
  unsigned TF = MO.getTargetFlags();

  switch (MO.getType()) {
  default: llvm_unreachable("<unknown operand type>");
  case MachineOperand::MO_Register: {
    unsigned Reg = MO.getReg();
    assert(TargetRegisterInfo::isPhysicalRegister(Reg));
    assert(!MO.getSubReg() && "Subregs should be eliminated!");
    if(ARM::GPRPairRegClass.contains(Reg)) {
      const MachineFunction &MF = *MI->getParent()->getParent();
      const TargetRegisterInfo *TRI = MF.getTarget().getRegisterInfo();
      Reg = TRI->getSubReg(Reg, ARM::gsub_0);
    }
    O << ARMInstPrinter::getRegisterName(Reg);
    break;
  }
  case MachineOperand::MO_Immediate: {
    int64_t Imm = MO.getImm();
    O << '#';
    if ((Modifier && strcmp(Modifier, "lo16") == 0) ||
        (TF == ARMII::MO_LO16))
      O << ":lower16:";
    else if ((Modifier && strcmp(Modifier, "hi16") == 0) ||
             (TF == ARMII::MO_HI16))
      O << ":upper16:";
    O << Imm;
    break;
  }
  case MachineOperand::MO_MachineBasicBlock:
    O << *MO.getMBB()->getSymbol();
    return;
  case MachineOperand::MO_GlobalAddress: {
    const GlobalValue *GV = MO.getGlobal();
    if ((Modifier && strcmp(Modifier, "lo16") == 0) ||
        (TF & ARMII::MO_LO16))
      O << ":lower16:";
    else if ((Modifier && strcmp(Modifier, "hi16") == 0) ||
             (TF & ARMII::MO_HI16))
      O << ":upper16:";
    O << *getSymbol(GV);

    printOffset(MO.getOffset(), O);
    if (TF == ARMII::MO_PLT)
      O << "(PLT)";
    break;
  }
  case MachineOperand::MO_ConstantPoolIndex:
    O << *GetCPISymbol(MO.getIndex());
    break;
  }
}

//===--------------------------------------------------------------------===//

MCSymbol *ARMAsmPrinter::
GetARMJTIPICJumpTableLabel2(unsigned uid, unsigned uid2) const {
  const DataLayout *DL = TM.getDataLayout();
  SmallString<60> Name;
  raw_svector_ostream(Name) << DL->getPrivateGlobalPrefix() << "JTI"
    << getFunctionNumber() << '_' << uid << '_' << uid2;
  return OutContext.GetOrCreateSymbol(Name.str());
}


MCSymbol *ARMAsmPrinter::GetARMSJLJEHLabel() const {
  const DataLayout *DL = TM.getDataLayout();
  SmallString<60> Name;
  raw_svector_ostream(Name) << DL->getPrivateGlobalPrefix() << "SJLJEH"
    << getFunctionNumber();
  return OutContext.GetOrCreateSymbol(Name.str());
}

bool ARMAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNum,
                                    unsigned AsmVariant, const char *ExtraCode,
                                    raw_ostream &O) {
  // Does this asm operand have a single letter operand modifier?
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default:
      // See if this is a generic print operand
      return AsmPrinter::PrintAsmOperand(MI, OpNum, AsmVariant, ExtraCode, O);
    case 'a': // Print as a memory address.
      if (MI->getOperand(OpNum).isReg()) {
        O << "["
          << ARMInstPrinter::getRegisterName(MI->getOperand(OpNum).getReg())
          << "]";
        return false;
      }
      // Fallthrough
    case 'c': // Don't print "#" before an immediate operand.
      if (!MI->getOperand(OpNum).isImm())
        return true;
      O << MI->getOperand(OpNum).getImm();
      return false;
    case 'P': // Print a VFP double precision register.
    case 'q': // Print a NEON quad precision register.
      printOperand(MI, OpNum, O);
      return false;
    case 'y': // Print a VFP single precision register as indexed double.
      if (MI->getOperand(OpNum).isReg()) {
        unsigned Reg = MI->getOperand(OpNum).getReg();
        const TargetRegisterInfo *TRI = MF->getTarget().getRegisterInfo();
        // Find the 'd' register that has this 's' register as a sub-register,
        // and determine the lane number.
        for (MCSuperRegIterator SR(Reg, TRI); SR.isValid(); ++SR) {
          if (!ARM::DPRRegClass.contains(*SR))
            continue;
          bool Lane0 = TRI->getSubReg(*SR, ARM::ssub_0) == Reg;
          O << ARMInstPrinter::getRegisterName(*SR) << (Lane0 ? "[0]" : "[1]");
          return false;
        }
      }
      return true;
    case 'B': // Bitwise inverse of integer or symbol without a preceding #.
      if (!MI->getOperand(OpNum).isImm())
        return true;
      O << ~(MI->getOperand(OpNum).getImm());
      return false;
    case 'L': // The low 16 bits of an immediate constant.
      if (!MI->getOperand(OpNum).isImm())
        return true;
      O << (MI->getOperand(OpNum).getImm() & 0xffff);
      return false;
    case 'M': { // A register range suitable for LDM/STM.
      if (!MI->getOperand(OpNum).isReg())
        return true;
      const MachineOperand &MO = MI->getOperand(OpNum);
      unsigned RegBegin = MO.getReg();
      // This takes advantage of the 2 operand-ness of ldm/stm and that we've
      // already got the operands in registers that are operands to the
      // inline asm statement.
      O << "{";
      if (ARM::GPRPairRegClass.contains(RegBegin)) {
        const TargetRegisterInfo *TRI = MF->getTarget().getRegisterInfo();
        unsigned Reg0 = TRI->getSubReg(RegBegin, ARM::gsub_0);
        O << ARMInstPrinter::getRegisterName(Reg0) << ", ";;
        RegBegin = TRI->getSubReg(RegBegin, ARM::gsub_1);
      }
      O << ARMInstPrinter::getRegisterName(RegBegin);

      // FIXME: The register allocator not only may not have given us the
      // registers in sequence, but may not be in ascending registers. This
      // will require changes in the register allocator that'll need to be
      // propagated down here if the operands change.
      unsigned RegOps = OpNum + 1;
      while (MI->getOperand(RegOps).isReg()) {
        O << ", "
          << ARMInstPrinter::getRegisterName(MI->getOperand(RegOps).getReg());
        RegOps++;
      }

      O << "}";

      return false;
    }
    case 'R': // The most significant register of a pair.
    case 'Q': { // The least significant register of a pair.
      if (OpNum == 0)
        return true;
      const MachineOperand &FlagsOP = MI->getOperand(OpNum - 1);
      if (!FlagsOP.isImm())
        return true;
      unsigned Flags = FlagsOP.getImm();

      // This operand may not be the one that actually provides the register. If
      // it's tied to a previous one then we should refer instead to that one
      // for registers and their classes.
      unsigned TiedIdx;
      if (InlineAsm::isUseOperandTiedToDef(Flags, TiedIdx)) {
        for (OpNum = InlineAsm::MIOp_FirstOperand; TiedIdx; --TiedIdx) {
          unsigned OpFlags = MI->getOperand(OpNum).getImm();
          OpNum += InlineAsm::getNumOperandRegisters(OpFlags) + 1;
        }
        Flags = MI->getOperand(OpNum).getImm();

        // Later code expects OpNum to be pointing at the register rather than
        // the flags.
        OpNum += 1;
      }

      unsigned NumVals = InlineAsm::getNumOperandRegisters(Flags);
      unsigned RC;
      InlineAsm::hasRegClassConstraint(Flags, RC);
      if (RC == ARM::GPRPairRegClassID) {
        if (NumVals != 1)
          return true;
        const MachineOperand &MO = MI->getOperand(OpNum);
        if (!MO.isReg())
          return true;
        const TargetRegisterInfo *TRI = MF->getTarget().getRegisterInfo();
        unsigned Reg = TRI->getSubReg(MO.getReg(), ExtraCode[0] == 'Q' ?
            ARM::gsub_0 : ARM::gsub_1);
        O << ARMInstPrinter::getRegisterName(Reg);
        return false;
      }
      if (NumVals != 2)
        return true;
      unsigned RegOp = ExtraCode[0] == 'Q' ? OpNum : OpNum + 1;
      if (RegOp >= MI->getNumOperands())
        return true;
      const MachineOperand &MO = MI->getOperand(RegOp);
      if (!MO.isReg())
        return true;
      unsigned Reg = MO.getReg();
      O << ARMInstPrinter::getRegisterName(Reg);
      return false;
    }

    case 'e': // The low doubleword register of a NEON quad register.
    case 'f': { // The high doubleword register of a NEON quad register.
      if (!MI->getOperand(OpNum).isReg())
        return true;
      unsigned Reg = MI->getOperand(OpNum).getReg();
      if (!ARM::QPRRegClass.contains(Reg))
        return true;
      const TargetRegisterInfo *TRI = MF->getTarget().getRegisterInfo();
      unsigned SubReg = TRI->getSubReg(Reg, ExtraCode[0] == 'e' ?
                                       ARM::dsub_0 : ARM::dsub_1);
      O << ARMInstPrinter::getRegisterName(SubReg);
      return false;
    }

    // This modifier is not yet supported.
    case 'h': // A range of VFP/NEON registers suitable for VLD1/VST1.
      return true;
    case 'H': { // The highest-numbered register of a pair.
      const MachineOperand &MO = MI->getOperand(OpNum);
      if (!MO.isReg())
        return true;
      const MachineFunction &MF = *MI->getParent()->getParent();
      const TargetRegisterInfo *TRI = MF.getTarget().getRegisterInfo();
      unsigned Reg = MO.getReg();
      if(!ARM::GPRPairRegClass.contains(Reg))
        return false;
      Reg = TRI->getSubReg(Reg, ARM::gsub_1);
      O << ARMInstPrinter::getRegisterName(Reg);
      return false;
    }
    }
  }

  printOperand(MI, OpNum, O);
  return false;
}

bool ARMAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                          unsigned OpNum, unsigned AsmVariant,
                                          const char *ExtraCode,
                                          raw_ostream &O) {
  // Does this asm operand have a single letter operand modifier?
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.

    switch (ExtraCode[0]) {
      case 'A': // A memory operand for a VLD1/VST1 instruction.
      default: return true;  // Unknown modifier.
      case 'm': // The base register of a memory operand.
        if (!MI->getOperand(OpNum).isReg())
          return true;
        O << ARMInstPrinter::getRegisterName(MI->getOperand(OpNum).getReg());
        return false;
    }
  }

  const MachineOperand &MO = MI->getOperand(OpNum);
  assert(MO.isReg() && "unexpected inline asm memory operand");
  O << "[" << ARMInstPrinter::getRegisterName(MO.getReg()) << "]";
  return false;
}

static bool isThumb(const MCSubtargetInfo& STI) {
  return (STI.getFeatureBits() & ARM::ModeThumb) != 0;
}

void ARMAsmPrinter::emitInlineAsmEnd(const MCSubtargetInfo &StartInfo,
                                     const MCSubtargetInfo *EndInfo) const {
  // If either end mode is unknown (EndInfo == NULL) or different than
  // the start mode, then restore the start mode.
  const bool WasThumb = isThumb(StartInfo);
  if (EndInfo == NULL || WasThumb != isThumb(*EndInfo)) {
    OutStreamer.EmitAssemblerFlag(WasThumb ? MCAF_Code16 : MCAF_Code32);
  }
}

void ARMAsmPrinter::EmitStartOfAsmFile(Module &M) {
  if (Subtarget->isTargetMachO()) {
    Reloc::Model RelocM = TM.getRelocationModel();
    if (RelocM == Reloc::PIC_ || RelocM == Reloc::DynamicNoPIC) {
      // Declare all the text sections up front (before the DWARF sections
      // emitted by AsmPrinter::doInitialization) so the assembler will keep
      // them together at the beginning of the object file.  This helps
      // avoid out-of-range branches that are due a fundamental limitation of
      // the way symbol offsets are encoded with the current Darwin ARM
      // relocations.
      const TargetLoweringObjectFileMachO &TLOFMacho =
        static_cast<const TargetLoweringObjectFileMachO &>(
          getObjFileLowering());

      // Collect the set of sections our functions will go into.
      SetVector<const MCSection *, SmallVector<const MCSection *, 8>,
        SmallPtrSet<const MCSection *, 8> > TextSections;
      // Default text section comes first.
      TextSections.insert(TLOFMacho.getTextSection());
      // Now any user defined text sections from function attributes.
      for (Module::iterator F = M.begin(), e = M.end(); F != e; ++F)
        if (!F->isDeclaration() && !F->hasAvailableExternallyLinkage())
          TextSections.insert(TLOFMacho.SectionForGlobal(F, *Mang, TM));
      // Now the coalescable sections.
      TextSections.insert(TLOFMacho.getTextCoalSection());
      TextSections.insert(TLOFMacho.getConstTextCoalSection());

      // Emit the sections in the .s file header to fix the order.
      for (unsigned i = 0, e = TextSections.size(); i != e; ++i)
        OutStreamer.SwitchSection(TextSections[i]);

      if (RelocM == Reloc::DynamicNoPIC) {
        const MCSection *sect =
          OutContext.getMachOSection("__TEXT", "__symbol_stub4",
                                     MCSectionMachO::S_SYMBOL_STUBS,
                                     12, SectionKind::getText());
        OutStreamer.SwitchSection(sect);
      } else {
        const MCSection *sect =
          OutContext.getMachOSection("__TEXT", "__picsymbolstub4",
                                     MCSectionMachO::S_SYMBOL_STUBS,
                                     16, SectionKind::getText());
        OutStreamer.SwitchSection(sect);
      }
      const MCSection *StaticInitSect =
        OutContext.getMachOSection("__TEXT", "__StaticInit",
                                   MCSectionMachO::S_REGULAR |
                                   MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                                   SectionKind::getText());
      OutStreamer.SwitchSection(StaticInitSect);
    }

    // Compiling with debug info should not affect the code
    // generation.  Ensure the cstring section comes before the
    // optional __DWARF secion. Otherwise, PC-relative loads would
    // have to use different instruction sequences at "-g" in order to
    // reach global data in the same object file.
    OutStreamer.SwitchSection(getObjFileLowering().getCStringSection());
  }

  // Use unified assembler syntax.
  OutStreamer.EmitAssemblerFlag(MCAF_SyntaxUnified);

  // Emit ARM Build Attributes
  if (Subtarget->isTargetELF())
    emitAttributes();
}


void ARMAsmPrinter::EmitEndOfAsmFile(Module &M) {
  if (Subtarget->isTargetMachO()) {
    // All darwin targets use mach-o.
    const TargetLoweringObjectFileMachO &TLOFMacho =
      static_cast<const TargetLoweringObjectFileMachO &>(getObjFileLowering());
    MachineModuleInfoMachO &MMIMacho =
      MMI->getObjFileInfo<MachineModuleInfoMachO>();

    // Output non-lazy-pointers for external and common global variables.
    MachineModuleInfoMachO::SymbolListTy Stubs = MMIMacho.GetGVStubList();

    if (!Stubs.empty()) {
      // Switch with ".non_lazy_symbol_pointer" directive.
      OutStreamer.SwitchSection(TLOFMacho.getNonLazySymbolPointerSection());
      EmitAlignment(2);
      for (unsigned i = 0, e = Stubs.size(); i != e; ++i) {
        // L_foo$stub:
        OutStreamer.EmitLabel(Stubs[i].first);
        //   .indirect_symbol _foo
        MachineModuleInfoImpl::StubValueTy &MCSym = Stubs[i].second;
        OutStreamer.EmitSymbolAttribute(MCSym.getPointer(),MCSA_IndirectSymbol);

        if (MCSym.getInt())
          // External to current translation unit.
          OutStreamer.EmitIntValue(0, 4/*size*/);
        else
          // Internal to current translation unit.
          //
          // When we place the LSDA into the TEXT section, the type info
          // pointers need to be indirect and pc-rel. We accomplish this by
          // using NLPs; however, sometimes the types are local to the file.
          // We need to fill in the value for the NLP in those cases.
          OutStreamer.EmitValue(MCSymbolRefExpr::Create(MCSym.getPointer(),
                                                        OutContext),
                                4/*size*/);
      }

      Stubs.clear();
      OutStreamer.AddBlankLine();
    }

    Stubs = MMIMacho.GetHiddenGVStubList();
    if (!Stubs.empty()) {
      OutStreamer.SwitchSection(getObjFileLowering().getDataSection());
      EmitAlignment(2);
      for (unsigned i = 0, e = Stubs.size(); i != e; ++i) {
        // L_foo$stub:
        OutStreamer.EmitLabel(Stubs[i].first);
        //   .long _foo
        OutStreamer.EmitValue(MCSymbolRefExpr::
                              Create(Stubs[i].second.getPointer(),
                                     OutContext),
                              4/*size*/);
      }

      Stubs.clear();
      OutStreamer.AddBlankLine();
    }

    // Funny Darwin hack: This flag tells the linker that no global symbols
    // contain code that falls through to other global symbols (e.g. the obvious
    // implementation of multiple entry points).  If this doesn't occur, the
    // linker can safely perform dead code stripping.  Since LLVM never
    // generates code that does this, it is always safe to set.
    OutStreamer.EmitAssemblerFlag(MCAF_SubsectionsViaSymbols);
  }
}

//===----------------------------------------------------------------------===//
// Helper routines for EmitStartOfAsmFile() and EmitEndOfAsmFile()
// FIXME:
// The following seem like one-off assembler flags, but they actually need
// to appear in the .ARM.attributes section in ELF.
// Instead of subclassing the MCELFStreamer, we do the work here.

static ARMBuildAttrs::CPUArch getArchForCPU(StringRef CPU,
                                            const ARMSubtarget *Subtarget) {
  if (CPU == "xscale")
    return ARMBuildAttrs::v5TEJ;

  if (Subtarget->hasV8Ops())
    return ARMBuildAttrs::v8;
  else if (Subtarget->hasV7Ops()) {
    if (Subtarget->isMClass() && Subtarget->hasThumb2DSP())
      return ARMBuildAttrs::v7E_M;
    return ARMBuildAttrs::v7;
  } else if (Subtarget->hasV6T2Ops())
    return ARMBuildAttrs::v6T2;
  else if (Subtarget->hasV6MOps())
    return ARMBuildAttrs::v6S_M;
  else if (Subtarget->hasV6Ops())
    return ARMBuildAttrs::v6;
  else if (Subtarget->hasV5TEOps())
    return ARMBuildAttrs::v5TE;
  else if (Subtarget->hasV5TOps())
    return ARMBuildAttrs::v5T;
  else if (Subtarget->hasV4TOps())
    return ARMBuildAttrs::v4T;
  else
    return ARMBuildAttrs::v4;
}

void ARMAsmPrinter::emitAttributes() {
  MCTargetStreamer &TS = *OutStreamer.getTargetStreamer();
  ARMTargetStreamer &ATS = static_cast<ARMTargetStreamer &>(TS);

  ATS.switchVendor("aeabi");

  std::string CPUString = Subtarget->getCPUString();

  // FIXME: remove krait check when GNU tools support krait cpu
  if (CPUString != "generic" && CPUString != "krait")
    ATS.emitTextAttribute(ARMBuildAttrs::CPU_name, CPUString);

  ATS.emitAttribute(ARMBuildAttrs::CPU_arch,
                    getArchForCPU(CPUString, Subtarget));

  // Tag_CPU_arch_profile must have the default value of 0 when "Architecture
  // profile is not applicable (e.g. pre v7, or cross-profile code)". 
  if (Subtarget->hasV7Ops()) {
    if (Subtarget->isAClass()) {
      ATS.emitAttribute(ARMBuildAttrs::CPU_arch_profile,
                        ARMBuildAttrs::ApplicationProfile);
    } else if (Subtarget->isRClass()) {
      ATS.emitAttribute(ARMBuildAttrs::CPU_arch_profile,
                        ARMBuildAttrs::RealTimeProfile);
    } else if (Subtarget->isMClass()) {
      ATS.emitAttribute(ARMBuildAttrs::CPU_arch_profile,
                        ARMBuildAttrs::MicroControllerProfile);
    }
  }

  ATS.emitAttribute(ARMBuildAttrs::ARM_ISA_use, Subtarget->hasARMOps() ?
                      ARMBuildAttrs::Allowed : ARMBuildAttrs::Not_Allowed);
  if (Subtarget->isThumb1Only()) {
    ATS.emitAttribute(ARMBuildAttrs::THUMB_ISA_use,
                      ARMBuildAttrs::Allowed);
  } else if (Subtarget->hasThumb2()) {
    ATS.emitAttribute(ARMBuildAttrs::THUMB_ISA_use,
                      ARMBuildAttrs::AllowThumb32);
  }

  if (Subtarget->hasNEON()) {
    /* NEON is not exactly a VFP architecture, but GAS emit one of
     * neon/neon-fp-armv8/neon-vfpv4/vfpv3/vfpv2 for .fpu parameters */
    if (Subtarget->hasFPARMv8()) {
      if (Subtarget->hasCrypto())
        ATS.emitFPU(ARM::CRYPTO_NEON_FP_ARMV8);
      else
        ATS.emitFPU(ARM::NEON_FP_ARMV8);
    }
    else if (Subtarget->hasVFP4())
      ATS.emitFPU(ARM::NEON_VFPV4);
    else
      ATS.emitFPU(ARM::NEON);
    // Emit Tag_Advanced_SIMD_arch for ARMv8 architecture
    if (Subtarget->hasV8Ops())
      ATS.emitAttribute(ARMBuildAttrs::Advanced_SIMD_arch,
                        ARMBuildAttrs::AllowNeonARMv8);
  } else {
    if (Subtarget->hasFPARMv8())
      ATS.emitFPU(ARM::FP_ARMV8);
    else if (Subtarget->hasVFP4())
      ATS.emitFPU(Subtarget->hasD16() ? ARM::VFPV4_D16 : ARM::VFPV4);
    else if (Subtarget->hasVFP3())
      ATS.emitFPU(Subtarget->hasD16() ? ARM::VFPV3_D16 : ARM::VFPV3);
    else if (Subtarget->hasVFP2())
      ATS.emitFPU(ARM::VFPV2);
  }

  // Signal various FP modes.
  if (!TM.Options.UnsafeFPMath) {
    ATS.emitAttribute(ARMBuildAttrs::ABI_FP_denormal, ARMBuildAttrs::Allowed);
    ATS.emitAttribute(ARMBuildAttrs::ABI_FP_exceptions,
                      ARMBuildAttrs::Allowed);
  }

  if (TM.Options.NoInfsFPMath && TM.Options.NoNaNsFPMath)
    ATS.emitAttribute(ARMBuildAttrs::ABI_FP_number_model,
                      ARMBuildAttrs::Allowed);
  else
    ATS.emitAttribute(ARMBuildAttrs::ABI_FP_number_model,
                      ARMBuildAttrs::AllowIEE754);

  // FIXME: add more flags to ARMBuildAttributes.h
  // 8-bytes alignment stuff.
  ATS.emitAttribute(ARMBuildAttrs::ABI_align_needed, 1);
  ATS.emitAttribute(ARMBuildAttrs::ABI_align_preserved, 1);

  // ABI_HardFP_use attribute to indicate single precision FP.
  if (Subtarget->isFPOnlySP())
    ATS.emitAttribute(ARMBuildAttrs::ABI_HardFP_use,
                      ARMBuildAttrs::HardFPSinglePrecision);

  // Hard float.  Use both S and D registers and conform to AAPCS-VFP.
  if (Subtarget->isAAPCS_ABI() && TM.Options.FloatABIType == FloatABI::Hard)
    ATS.emitAttribute(ARMBuildAttrs::ABI_VFP_args, ARMBuildAttrs::HardFPAAPCS);

  // FIXME: Should we signal R9 usage?

  if (Subtarget->hasFP16())
      ATS.emitAttribute(ARMBuildAttrs::FP_HP_extension, ARMBuildAttrs::AllowHPFP);

  if (Subtarget->hasMPExtension())
      ATS.emitAttribute(ARMBuildAttrs::MPextension_use, ARMBuildAttrs::AllowMP);

  // Hardware divide in ARM mode is part of base arch, starting from ARMv8.
  // If only Thumb hwdiv is present, it must also be in base arch (ARMv7-R/M).
  // It is not possible to produce DisallowDIV: if hwdiv is present in the base
  // arch, supplying -hwdiv downgrades the effective arch, via ClearImpliedBits.
  // AllowDIVExt is only emitted if hwdiv isn't available in the base arch;
  // otherwise, the default value (AllowDIVIfExists) applies.
  if (Subtarget->hasDivideInARMMode() && !Subtarget->hasV8Ops())
      ATS.emitAttribute(ARMBuildAttrs::DIV_use, ARMBuildAttrs::AllowDIVExt);

  if (Subtarget->hasTrustZone() && Subtarget->hasVirtualization())
      ATS.emitAttribute(ARMBuildAttrs::Virtualization_use,
                        ARMBuildAttrs::AllowTZVirtualization);
  else if (Subtarget->hasTrustZone())
      ATS.emitAttribute(ARMBuildAttrs::Virtualization_use,
                        ARMBuildAttrs::AllowTZ);
  else if (Subtarget->hasVirtualization())
      ATS.emitAttribute(ARMBuildAttrs::Virtualization_use,
                        ARMBuildAttrs::AllowVirtualization);

  ATS.finishAttributeSection();
}

//===----------------------------------------------------------------------===//

static MCSymbol *getPICLabel(const char *Prefix, unsigned FunctionNumber,
                             unsigned LabelId, MCContext &Ctx) {

  MCSymbol *Label = Ctx.GetOrCreateSymbol(Twine(Prefix)
                       + "PC" + Twine(FunctionNumber) + "_" + Twine(LabelId));
  return Label;
}

static MCSymbolRefExpr::VariantKind
getModifierVariantKind(ARMCP::ARMCPModifier Modifier) {
  switch (Modifier) {
  case ARMCP::no_modifier: return MCSymbolRefExpr::VK_None;
  case ARMCP::TLSGD:       return MCSymbolRefExpr::VK_TLSGD;
  case ARMCP::TPOFF:       return MCSymbolRefExpr::VK_TPOFF;
  case ARMCP::GOTTPOFF:    return MCSymbolRefExpr::VK_GOTTPOFF;
  case ARMCP::GOT:         return MCSymbolRefExpr::VK_GOT;
  case ARMCP::GOTOFF:      return MCSymbolRefExpr::VK_GOTOFF;
  }
  llvm_unreachable("Invalid ARMCPModifier!");
}

MCSymbol *ARMAsmPrinter::GetARMGVSymbol(const GlobalValue *GV,
                                        unsigned char TargetFlags) {
  bool isIndirect = Subtarget->isTargetMachO() &&
    (TargetFlags & ARMII::MO_NONLAZY) &&
    Subtarget->GVIsIndirectSymbol(GV, TM.getRelocationModel());
  if (!isIndirect)
    return getSymbol(GV);

  // FIXME: Remove this when Darwin transition to @GOT like syntax.
  MCSymbol *MCSym = getSymbolWithGlobalValueBase(GV, "$non_lazy_ptr");
  MachineModuleInfoMachO &MMIMachO =
    MMI->getObjFileInfo<MachineModuleInfoMachO>();
  MachineModuleInfoImpl::StubValueTy &StubSym =
    GV->hasHiddenVisibility() ? MMIMachO.getHiddenGVStubEntry(MCSym) :
    MMIMachO.getGVStubEntry(MCSym);
  if (StubSym.getPointer() == 0)
    StubSym = MachineModuleInfoImpl::
      StubValueTy(getSymbol(GV), !GV->hasInternalLinkage());
  return MCSym;
}

void ARMAsmPrinter::
EmitMachineConstantPoolValue(MachineConstantPoolValue *MCPV) {
  const DataLayout *DL = TM.getDataLayout();
  int Size = TM.getDataLayout()->getTypeAllocSize(MCPV->getType());

  ARMConstantPoolValue *ACPV = static_cast<ARMConstantPoolValue*>(MCPV);

  MCSymbol *MCSym;
  if (ACPV->isLSDA()) {
    SmallString<128> Str;
    raw_svector_ostream OS(Str);
    OS << DL->getPrivateGlobalPrefix() << "_LSDA_" << getFunctionNumber();
    MCSym = OutContext.GetOrCreateSymbol(OS.str());
  } else if (ACPV->isBlockAddress()) {
    const BlockAddress *BA =
      cast<ARMConstantPoolConstant>(ACPV)->getBlockAddress();
    MCSym = GetBlockAddressSymbol(BA);
  } else if (ACPV->isGlobalValue()) {
    const GlobalValue *GV = cast<ARMConstantPoolConstant>(ACPV)->getGV();

    // On Darwin, const-pool entries may get the "FOO$non_lazy_ptr" mangling, so
    // flag the global as MO_NONLAZY.
    unsigned char TF = Subtarget->isTargetMachO() ? ARMII::MO_NONLAZY : 0;
    MCSym = GetARMGVSymbol(GV, TF);
  } else if (ACPV->isMachineBasicBlock()) {
    const MachineBasicBlock *MBB = cast<ARMConstantPoolMBB>(ACPV)->getMBB();
    MCSym = MBB->getSymbol();
  } else {
    assert(ACPV->isExtSymbol() && "unrecognized constant pool value");
    const char *Sym = cast<ARMConstantPoolSymbol>(ACPV)->getSymbol();
    MCSym = GetExternalSymbolSymbol(Sym);
  }

  // Create an MCSymbol for the reference.
  const MCExpr *Expr =
    MCSymbolRefExpr::Create(MCSym, getModifierVariantKind(ACPV->getModifier()),
                            OutContext);

  if (ACPV->getPCAdjustment()) {
    MCSymbol *PCLabel = getPICLabel(DL->getPrivateGlobalPrefix(),
                                    getFunctionNumber(),
                                    ACPV->getLabelId(),
                                    OutContext);
    const MCExpr *PCRelExpr = MCSymbolRefExpr::Create(PCLabel, OutContext);
    PCRelExpr =
      MCBinaryExpr::CreateAdd(PCRelExpr,
                              MCConstantExpr::Create(ACPV->getPCAdjustment(),
                                                     OutContext),
                              OutContext);
    if (ACPV->mustAddCurrentAddress()) {
      // We want "(<expr> - .)", but MC doesn't have a concept of the '.'
      // label, so just emit a local label end reference that instead.
      MCSymbol *DotSym = OutContext.CreateTempSymbol();
      OutStreamer.EmitLabel(DotSym);
      const MCExpr *DotExpr = MCSymbolRefExpr::Create(DotSym, OutContext);
      PCRelExpr = MCBinaryExpr::CreateSub(PCRelExpr, DotExpr, OutContext);
    }
    Expr = MCBinaryExpr::CreateSub(Expr, PCRelExpr, OutContext);
  }
  OutStreamer.EmitValue(Expr, Size);
}

void ARMAsmPrinter::EmitJumpTable(const MachineInstr *MI) {
  unsigned Opcode = MI->getOpcode();
  int OpNum = 1;
  if (Opcode == ARM::BR_JTadd)
    OpNum = 2;
  else if (Opcode == ARM::BR_JTm)
    OpNum = 3;

  const MachineOperand &MO1 = MI->getOperand(OpNum);
  const MachineOperand &MO2 = MI->getOperand(OpNum+1); // Unique Id
  unsigned JTI = MO1.getIndex();

  // Emit a label for the jump table.
  MCSymbol *JTISymbol = GetARMJTIPICJumpTableLabel2(JTI, MO2.getImm());
  OutStreamer.EmitLabel(JTISymbol);

  // Mark the jump table as data-in-code.
  OutStreamer.EmitDataRegion(MCDR_DataRegionJT32);

  // Emit each entry of the table.
  const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  const std::vector<MachineBasicBlock*> &JTBBs = JT[JTI].MBBs;

  for (unsigned i = 0, e = JTBBs.size(); i != e; ++i) {
    MachineBasicBlock *MBB = JTBBs[i];
    // Construct an MCExpr for the entry. We want a value of the form:
    // (BasicBlockAddr - TableBeginAddr)
    //
    // For example, a table with entries jumping to basic blocks BB0 and BB1
    // would look like:
    // LJTI_0_0:
    //    .word (LBB0 - LJTI_0_0)
    //    .word (LBB1 - LJTI_0_0)
    const MCExpr *Expr = MCSymbolRefExpr::Create(MBB->getSymbol(), OutContext);

    if (TM.getRelocationModel() == Reloc::PIC_)
      Expr = MCBinaryExpr::CreateSub(Expr, MCSymbolRefExpr::Create(JTISymbol,
                                                                   OutContext),
                                     OutContext);
    // If we're generating a table of Thumb addresses in static relocation
    // model, we need to add one to keep interworking correctly.
    else if (AFI->isThumbFunction())
      Expr = MCBinaryExpr::CreateAdd(Expr, MCConstantExpr::Create(1,OutContext),
                                     OutContext);
    OutStreamer.EmitValue(Expr, 4);
  }
  // Mark the end of jump table data-in-code region.
  OutStreamer.EmitDataRegion(MCDR_DataRegionEnd);
}

void ARMAsmPrinter::EmitJump2Table(const MachineInstr *MI) {
  unsigned Opcode = MI->getOpcode();
  int OpNum = (Opcode == ARM::t2BR_JT) ? 2 : 1;
  const MachineOperand &MO1 = MI->getOperand(OpNum);
  const MachineOperand &MO2 = MI->getOperand(OpNum+1); // Unique Id
  unsigned JTI = MO1.getIndex();

  MCSymbol *JTISymbol = GetARMJTIPICJumpTableLabel2(JTI, MO2.getImm());
  OutStreamer.EmitLabel(JTISymbol);

  // Emit each entry of the table.
  const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  const std::vector<MachineBasicBlock*> &JTBBs = JT[JTI].MBBs;
  unsigned OffsetWidth = 4;
  if (MI->getOpcode() == ARM::t2TBB_JT) {
    OffsetWidth = 1;
    // Mark the jump table as data-in-code.
    OutStreamer.EmitDataRegion(MCDR_DataRegionJT8);
  } else if (MI->getOpcode() == ARM::t2TBH_JT) {
    OffsetWidth = 2;
    // Mark the jump table as data-in-code.
    OutStreamer.EmitDataRegion(MCDR_DataRegionJT16);
  }

  for (unsigned i = 0, e = JTBBs.size(); i != e; ++i) {
    MachineBasicBlock *MBB = JTBBs[i];
    const MCExpr *MBBSymbolExpr = MCSymbolRefExpr::Create(MBB->getSymbol(),
                                                      OutContext);
    // If this isn't a TBB or TBH, the entries are direct branch instructions.
    if (OffsetWidth == 4) {
      EmitToStreamer(OutStreamer, MCInstBuilder(ARM::t2B)
        .addExpr(MBBSymbolExpr)
        .addImm(ARMCC::AL)
        .addReg(0));
      continue;
    }
    // Otherwise it's an offset from the dispatch instruction. Construct an
    // MCExpr for the entry. We want a value of the form:
    // (BasicBlockAddr - TableBeginAddr) / 2
    //
    // For example, a TBB table with entries jumping to basic blocks BB0 and BB1
    // would look like:
    // LJTI_0_0:
    //    .byte (LBB0 - LJTI_0_0) / 2
    //    .byte (LBB1 - LJTI_0_0) / 2
    const MCExpr *Expr =
      MCBinaryExpr::CreateSub(MBBSymbolExpr,
                              MCSymbolRefExpr::Create(JTISymbol, OutContext),
                              OutContext);
    Expr = MCBinaryExpr::CreateDiv(Expr, MCConstantExpr::Create(2, OutContext),
                                   OutContext);
    OutStreamer.EmitValue(Expr, OffsetWidth);
  }
  // Mark the end of jump table data-in-code region. 32-bit offsets use
  // actual branch instructions here, so we don't mark those as a data-region
  // at all.
  if (OffsetWidth != 4)
    OutStreamer.EmitDataRegion(MCDR_DataRegionEnd);
}

void ARMAsmPrinter::EmitUnwindingInstruction(const MachineInstr *MI) {
  assert(MI->getFlag(MachineInstr::FrameSetup) &&
      "Only instruction which are involved into frame setup code are allowed");

  MCTargetStreamer &TS = *OutStreamer.getTargetStreamer();
  ARMTargetStreamer &ATS = static_cast<ARMTargetStreamer &>(TS);
  const MachineFunction &MF = *MI->getParent()->getParent();
  const TargetRegisterInfo *RegInfo = MF.getTarget().getRegisterInfo();
  const ARMFunctionInfo &AFI = *MF.getInfo<ARMFunctionInfo>();

  unsigned FramePtr = RegInfo->getFrameRegister(MF);
  unsigned Opc = MI->getOpcode();
  unsigned SrcReg, DstReg;

  if (Opc == ARM::tPUSH || Opc == ARM::tLDRpci) {
    // Two special cases:
    // 1) tPUSH does not have src/dst regs.
    // 2) for Thumb1 code we sometimes materialize the constant via constpool
    // load. Yes, this is pretty fragile, but for now I don't see better
    // way... :(
    SrcReg = DstReg = ARM::SP;
  } else {
    SrcReg = MI->getOperand(1).getReg();
    DstReg = MI->getOperand(0).getReg();
  }

  // Try to figure out the unwinding opcode out of src / dst regs.
  if (MI->mayStore()) {
    // Register saves.
    assert(DstReg == ARM::SP &&
           "Only stack pointer as a destination reg is supported");

    SmallVector<unsigned, 4> RegList;
    // Skip src & dst reg, and pred ops.
    unsigned StartOp = 2 + 2;
    // Use all the operands.
    unsigned NumOffset = 0;

    switch (Opc) {
    default:
      MI->dump();
      llvm_unreachable("Unsupported opcode for unwinding information");
    case ARM::tPUSH:
      // Special case here: no src & dst reg, but two extra imp ops.
      StartOp = 2; NumOffset = 2;
    case ARM::STMDB_UPD:
    case ARM::t2STMDB_UPD:
    case ARM::VSTMDDB_UPD:
      assert(SrcReg == ARM::SP &&
             "Only stack pointer as a source reg is supported");
      for (unsigned i = StartOp, NumOps = MI->getNumOperands() - NumOffset;
           i != NumOps; ++i) {
        const MachineOperand &MO = MI->getOperand(i);
        // Actually, there should never be any impdef stuff here. Skip it
        // temporary to workaround PR11902.
        if (MO.isImplicit())
          continue;
        RegList.push_back(MO.getReg());
      }
      break;
    case ARM::STR_PRE_IMM:
    case ARM::STR_PRE_REG:
    case ARM::t2STR_PRE:
      assert(MI->getOperand(2).getReg() == ARM::SP &&
             "Only stack pointer as a source reg is supported");
      RegList.push_back(SrcReg);
      break;
    }
    ATS.emitRegSave(RegList, Opc == ARM::VSTMDDB_UPD);
  } else {
    // Changes of stack / frame pointer.
    if (SrcReg == ARM::SP) {
      int64_t Offset = 0;
      switch (Opc) {
      default:
        MI->dump();
        llvm_unreachable("Unsupported opcode for unwinding information");
      case ARM::MOVr:
      case ARM::tMOVr:
        Offset = 0;
        break;
      case ARM::ADDri:
        Offset = -MI->getOperand(2).getImm();
        break;
      case ARM::SUBri:
      case ARM::t2SUBri:
        Offset = MI->getOperand(2).getImm();
        break;
      case ARM::tSUBspi:
        Offset = MI->getOperand(2).getImm()*4;
        break;
      case ARM::tADDspi:
      case ARM::tADDrSPi:
        Offset = -MI->getOperand(2).getImm()*4;
        break;
      case ARM::tLDRpci: {
        // Grab the constpool index and check, whether it corresponds to
        // original or cloned constpool entry.
        unsigned CPI = MI->getOperand(1).getIndex();
        const MachineConstantPool *MCP = MF.getConstantPool();
        if (CPI >= MCP->getConstants().size())
          CPI = AFI.getOriginalCPIdx(CPI);
        assert(CPI != -1U && "Invalid constpool index");

        // Derive the actual offset.
        const MachineConstantPoolEntry &CPE = MCP->getConstants()[CPI];
        assert(!CPE.isMachineConstantPoolEntry() && "Invalid constpool entry");
        // FIXME: Check for user, it should be "add" instruction!
        Offset = -cast<ConstantInt>(CPE.Val.ConstVal)->getSExtValue();
        break;
      }
      }

      if (DstReg == FramePtr && FramePtr != ARM::SP)
        // Set-up of the frame pointer. Positive values correspond to "add"
        // instruction.
        ATS.emitSetFP(FramePtr, ARM::SP, -Offset);
      else if (DstReg == ARM::SP) {
        // Change of SP by an offset. Positive values correspond to "sub"
        // instruction.
        ATS.emitPad(Offset);
      } else {
        // Move of SP to a register.  Positive values correspond to an "add"
        // instruction.
        ATS.emitMovSP(DstReg, -Offset);
      }
    } else if (DstReg == ARM::SP) {
      MI->dump();
      llvm_unreachable("Unsupported opcode for unwinding information");
    }
    else {
      MI->dump();
      llvm_unreachable("Unsupported opcode for unwinding information");
    }
  }
}

// Simple pseudo-instructions have their lowering (with expansion to real
// instructions) auto-generated.
#include "ARMGenMCPseudoLowering.inc"

void ARMAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  const DataLayout *DL = TM.getDataLayout();

  // If we just ended a constant pool, mark it as such.
  if (InConstantPool && MI->getOpcode() != ARM::CONSTPOOL_ENTRY) {
    OutStreamer.EmitDataRegion(MCDR_DataRegionEnd);
    InConstantPool = false;
  }

  // Emit unwinding stuff for frame-related instructions
  if (Subtarget->isTargetEHABICompatible() &&
       MI->getFlag(MachineInstr::FrameSetup))
    EmitUnwindingInstruction(MI);

  // Do any auto-generated pseudo lowerings.
  if (emitPseudoExpansionLowering(OutStreamer, MI))
    return;

  assert(!convertAddSubFlagsOpcode(MI->getOpcode()) &&
         "Pseudo flag setting opcode should be expanded early");

  // Check for manual lowerings.
  unsigned Opc = MI->getOpcode();
  switch (Opc) {
  case ARM::t2MOVi32imm: llvm_unreachable("Should be lowered by thumb2it pass");
  case ARM::DBG_VALUE: llvm_unreachable("Should be handled by generic printing");
  case ARM::LEApcrel:
  case ARM::tLEApcrel:
  case ARM::t2LEApcrel: {
    // FIXME: Need to also handle globals and externals
    MCSymbol *CPISymbol = GetCPISymbol(MI->getOperand(1).getIndex());
    EmitToStreamer(OutStreamer, MCInstBuilder(MI->getOpcode() ==
                                              ARM::t2LEApcrel ? ARM::t2ADR
                  : (MI->getOpcode() == ARM::tLEApcrel ? ARM::tADR
                     : ARM::ADR))
      .addReg(MI->getOperand(0).getReg())
      .addExpr(MCSymbolRefExpr::Create(CPISymbol, OutContext))
      // Add predicate operands.
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg()));
    return;
  }
  case ARM::LEApcrelJT:
  case ARM::tLEApcrelJT:
  case ARM::t2LEApcrelJT: {
    MCSymbol *JTIPICSymbol =
      GetARMJTIPICJumpTableLabel2(MI->getOperand(1).getIndex(),
                                  MI->getOperand(2).getImm());
    EmitToStreamer(OutStreamer, MCInstBuilder(MI->getOpcode() ==
                                              ARM::t2LEApcrelJT ? ARM::t2ADR
                  : (MI->getOpcode() == ARM::tLEApcrelJT ? ARM::tADR
                     : ARM::ADR))
      .addReg(MI->getOperand(0).getReg())
      .addExpr(MCSymbolRefExpr::Create(JTIPICSymbol, OutContext))
      // Add predicate operands.
      .addImm(MI->getOperand(3).getImm())
      .addReg(MI->getOperand(4).getReg()));
    return;
  }
  // Darwin call instructions are just normal call instructions with different
  // clobber semantics (they clobber R9).
  case ARM::BX_CALL: {
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::MOVr)
      .addReg(ARM::LR)
      .addReg(ARM::PC)
      // Add predicate operands.
      .addImm(ARMCC::AL)
      .addReg(0)
      // Add 's' bit operand (always reg0 for this)
      .addReg(0));

    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::BX)
      .addReg(MI->getOperand(0).getReg()));
    return;
  }
  case ARM::tBX_CALL: {
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::tMOVr)
      .addReg(ARM::LR)
      .addReg(ARM::PC)
      // Add predicate operands.
      .addImm(ARMCC::AL)
      .addReg(0));

    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::tBX)
      .addReg(MI->getOperand(0).getReg())
      // Add predicate operands.
      .addImm(ARMCC::AL)
      .addReg(0));
    return;
  }
  case ARM::BMOVPCRX_CALL: {
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::MOVr)
      .addReg(ARM::LR)
      .addReg(ARM::PC)
      // Add predicate operands.
      .addImm(ARMCC::AL)
      .addReg(0)
      // Add 's' bit operand (always reg0 for this)
      .addReg(0));

    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::MOVr)
      .addReg(ARM::PC)
      .addReg(MI->getOperand(0).getReg())
      // Add predicate operands.
      .addImm(ARMCC::AL)
      .addReg(0)
      // Add 's' bit operand (always reg0 for this)
      .addReg(0));
    return;
  }
  case ARM::BMOVPCB_CALL: {
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::MOVr)
      .addReg(ARM::LR)
      .addReg(ARM::PC)
      // Add predicate operands.
      .addImm(ARMCC::AL)
      .addReg(0)
      // Add 's' bit operand (always reg0 for this)
      .addReg(0));

    const GlobalValue *GV = MI->getOperand(0).getGlobal();
    MCSymbol *GVSym = getSymbol(GV);
    const MCExpr *GVSymExpr = MCSymbolRefExpr::Create(GVSym, OutContext);
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::Bcc)
      .addExpr(GVSymExpr)
      // Add predicate operands.
      .addImm(ARMCC::AL)
      .addReg(0));
    return;
  }
  case ARM::MOVi16_ga_pcrel:
  case ARM::t2MOVi16_ga_pcrel: {
    MCInst TmpInst;
    TmpInst.setOpcode(Opc == ARM::MOVi16_ga_pcrel? ARM::MOVi16 : ARM::t2MOVi16);
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));

    unsigned TF = MI->getOperand(1).getTargetFlags();
    const GlobalValue *GV = MI->getOperand(1).getGlobal();
    MCSymbol *GVSym = GetARMGVSymbol(GV, TF);
    const MCExpr *GVSymExpr = MCSymbolRefExpr::Create(GVSym, OutContext);

    MCSymbol *LabelSym = getPICLabel(DL->getPrivateGlobalPrefix(),
                                     getFunctionNumber(),
                                     MI->getOperand(2).getImm(), OutContext);
    const MCExpr *LabelSymExpr= MCSymbolRefExpr::Create(LabelSym, OutContext);
    unsigned PCAdj = (Opc == ARM::MOVi16_ga_pcrel) ? 8 : 4;
    const MCExpr *PCRelExpr =
      ARMMCExpr::CreateLower16(MCBinaryExpr::CreateSub(GVSymExpr,
                                      MCBinaryExpr::CreateAdd(LabelSymExpr,
                                      MCConstantExpr::Create(PCAdj, OutContext),
                                      OutContext), OutContext), OutContext);
      TmpInst.addOperand(MCOperand::CreateExpr(PCRelExpr));

    // Add predicate operands.
    TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    TmpInst.addOperand(MCOperand::CreateReg(0));
    // Add 's' bit operand (always reg0 for this)
    TmpInst.addOperand(MCOperand::CreateReg(0));
    EmitToStreamer(OutStreamer, TmpInst);
    return;
  }
  case ARM::MOVTi16_ga_pcrel:
  case ARM::t2MOVTi16_ga_pcrel: {
    MCInst TmpInst;
    TmpInst.setOpcode(Opc == ARM::MOVTi16_ga_pcrel
                      ? ARM::MOVTi16 : ARM::t2MOVTi16);
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(1).getReg()));

    unsigned TF = MI->getOperand(2).getTargetFlags();
    const GlobalValue *GV = MI->getOperand(2).getGlobal();
    MCSymbol *GVSym = GetARMGVSymbol(GV, TF);
    const MCExpr *GVSymExpr = MCSymbolRefExpr::Create(GVSym, OutContext);

    MCSymbol *LabelSym = getPICLabel(DL->getPrivateGlobalPrefix(),
                                     getFunctionNumber(),
                                     MI->getOperand(3).getImm(), OutContext);
    const MCExpr *LabelSymExpr= MCSymbolRefExpr::Create(LabelSym, OutContext);
    unsigned PCAdj = (Opc == ARM::MOVTi16_ga_pcrel) ? 8 : 4;
    const MCExpr *PCRelExpr =
        ARMMCExpr::CreateUpper16(MCBinaryExpr::CreateSub(GVSymExpr,
                                   MCBinaryExpr::CreateAdd(LabelSymExpr,
                                      MCConstantExpr::Create(PCAdj, OutContext),
                                          OutContext), OutContext), OutContext);
      TmpInst.addOperand(MCOperand::CreateExpr(PCRelExpr));
    // Add predicate operands.
    TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    TmpInst.addOperand(MCOperand::CreateReg(0));
    // Add 's' bit operand (always reg0 for this)
    TmpInst.addOperand(MCOperand::CreateReg(0));
    EmitToStreamer(OutStreamer, TmpInst);
    return;
  }
  case ARM::tPICADD: {
    // This is a pseudo op for a label + instruction sequence, which looks like:
    // LPC0:
    //     add r0, pc
    // This adds the address of LPC0 to r0.

    // Emit the label.
    OutStreamer.EmitLabel(getPICLabel(DL->getPrivateGlobalPrefix(),
                          getFunctionNumber(), MI->getOperand(2).getImm(),
                          OutContext));

    // Form and emit the add.
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::tADDhirr)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(0).getReg())
      .addReg(ARM::PC)
      // Add predicate operands.
      .addImm(ARMCC::AL)
      .addReg(0));
    return;
  }
  case ARM::PICADD: {
    // This is a pseudo op for a label + instruction sequence, which looks like:
    // LPC0:
    //     add r0, pc, r0
    // This adds the address of LPC0 to r0.

    // Emit the label.
    OutStreamer.EmitLabel(getPICLabel(DL->getPrivateGlobalPrefix(),
                          getFunctionNumber(), MI->getOperand(2).getImm(),
                          OutContext));

    // Form and emit the add.
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::ADDrr)
      .addReg(MI->getOperand(0).getReg())
      .addReg(ARM::PC)
      .addReg(MI->getOperand(1).getReg())
      // Add predicate operands.
      .addImm(MI->getOperand(3).getImm())
      .addReg(MI->getOperand(4).getReg())
      // Add 's' bit operand (always reg0 for this)
      .addReg(0));
    return;
  }
  case ARM::PICSTR:
  case ARM::PICSTRB:
  case ARM::PICSTRH:
  case ARM::PICLDR:
  case ARM::PICLDRB:
  case ARM::PICLDRH:
  case ARM::PICLDRSB:
  case ARM::PICLDRSH: {
    // This is a pseudo op for a label + instruction sequence, which looks like:
    // LPC0:
    //     OP r0, [pc, r0]
    // The LCP0 label is referenced by a constant pool entry in order to get
    // a PC-relative address at the ldr instruction.

    // Emit the label.
    OutStreamer.EmitLabel(getPICLabel(DL->getPrivateGlobalPrefix(),
                          getFunctionNumber(), MI->getOperand(2).getImm(),
                          OutContext));

    // Form and emit the load
    unsigned Opcode;
    switch (MI->getOpcode()) {
    default:
      llvm_unreachable("Unexpected opcode!");
    case ARM::PICSTR:   Opcode = ARM::STRrs; break;
    case ARM::PICSTRB:  Opcode = ARM::STRBrs; break;
    case ARM::PICSTRH:  Opcode = ARM::STRH; break;
    case ARM::PICLDR:   Opcode = ARM::LDRrs; break;
    case ARM::PICLDRB:  Opcode = ARM::LDRBrs; break;
    case ARM::PICLDRH:  Opcode = ARM::LDRH; break;
    case ARM::PICLDRSB: Opcode = ARM::LDRSB; break;
    case ARM::PICLDRSH: Opcode = ARM::LDRSH; break;
    }
    EmitToStreamer(OutStreamer, MCInstBuilder(Opcode)
      .addReg(MI->getOperand(0).getReg())
      .addReg(ARM::PC)
      .addReg(MI->getOperand(1).getReg())
      .addImm(0)
      // Add predicate operands.
      .addImm(MI->getOperand(3).getImm())
      .addReg(MI->getOperand(4).getReg()));

    return;
  }
  case ARM::CONSTPOOL_ENTRY: {
    /// CONSTPOOL_ENTRY - This instruction represents a floating constant pool
    /// in the function.  The first operand is the ID# for this instruction, the
    /// second is the index into the MachineConstantPool that this is, the third
    /// is the size in bytes of this constant pool entry.
    /// The required alignment is specified on the basic block holding this MI.
    unsigned LabelId = (unsigned)MI->getOperand(0).getImm();
    unsigned CPIdx   = (unsigned)MI->getOperand(1).getIndex();

    // If this is the first entry of the pool, mark it.
    if (!InConstantPool) {
      OutStreamer.EmitDataRegion(MCDR_DataRegion);
      InConstantPool = true;
    }

    OutStreamer.EmitLabel(GetCPISymbol(LabelId));

    const MachineConstantPoolEntry &MCPE = MCP->getConstants()[CPIdx];
    if (MCPE.isMachineConstantPoolEntry())
      EmitMachineConstantPoolValue(MCPE.Val.MachineCPVal);
    else
      EmitGlobalConstant(MCPE.Val.ConstVal);
    return;
  }
  case ARM::t2BR_JT: {
    // Lower and emit the instruction itself, then the jump table following it.
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::tMOVr)
      .addReg(ARM::PC)
      .addReg(MI->getOperand(0).getReg())
      // Add predicate operands.
      .addImm(ARMCC::AL)
      .addReg(0));

    // Output the data for the jump table itself
    EmitJump2Table(MI);
    return;
  }
  case ARM::t2TBB_JT: {
    // Lower and emit the instruction itself, then the jump table following it.
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::t2TBB)
      .addReg(ARM::PC)
      .addReg(MI->getOperand(0).getReg())
      // Add predicate operands.
      .addImm(ARMCC::AL)
      .addReg(0));

    // Output the data for the jump table itself
    EmitJump2Table(MI);
    // Make sure the next instruction is 2-byte aligned.
    EmitAlignment(1);
    return;
  }
  case ARM::t2TBH_JT: {
    // Lower and emit the instruction itself, then the jump table following it.
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::t2TBH)
      .addReg(ARM::PC)
      .addReg(MI->getOperand(0).getReg())
      // Add predicate operands.
      .addImm(ARMCC::AL)
      .addReg(0));

    // Output the data for the jump table itself
    EmitJump2Table(MI);
    return;
  }
  case ARM::tBR_JTr:
  case ARM::BR_JTr: {
    // Lower and emit the instruction itself, then the jump table following it.
    // mov pc, target
    MCInst TmpInst;
    unsigned Opc = MI->getOpcode() == ARM::BR_JTr ?
      ARM::MOVr : ARM::tMOVr;
    TmpInst.setOpcode(Opc);
    TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    // Add predicate operands.
    TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    TmpInst.addOperand(MCOperand::CreateReg(0));
    // Add 's' bit operand (always reg0 for this)
    if (Opc == ARM::MOVr)
      TmpInst.addOperand(MCOperand::CreateReg(0));
    EmitToStreamer(OutStreamer, TmpInst);

    // Make sure the Thumb jump table is 4-byte aligned.
    if (Opc == ARM::tMOVr)
      EmitAlignment(2);

    // Output the data for the jump table itself
    EmitJumpTable(MI);
    return;
  }
  case ARM::BR_JTm: {
    // Lower and emit the instruction itself, then the jump table following it.
    // ldr pc, target
    MCInst TmpInst;
    if (MI->getOperand(1).getReg() == 0) {
      // literal offset
      TmpInst.setOpcode(ARM::LDRi12);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
      TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
      TmpInst.addOperand(MCOperand::CreateImm(MI->getOperand(2).getImm()));
    } else {
      TmpInst.setOpcode(ARM::LDRrs);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
      TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
      TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(1).getReg()));
      TmpInst.addOperand(MCOperand::CreateImm(0));
    }
    // Add predicate operands.
    TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    TmpInst.addOperand(MCOperand::CreateReg(0));
    EmitToStreamer(OutStreamer, TmpInst);

    // Output the data for the jump table itself
    EmitJumpTable(MI);
    return;
  }
  case ARM::BR_JTadd: {
    // Lower and emit the instruction itself, then the jump table following it.
    // add pc, target, idx
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::ADDrr)
      .addReg(ARM::PC)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      // Add predicate operands.
      .addImm(ARMCC::AL)
      .addReg(0)
      // Add 's' bit operand (always reg0 for this)
      .addReg(0));

    // Output the data for the jump table itself
    EmitJumpTable(MI);
    return;
  }
  case ARM::TRAP: {
    // Non-Darwin binutils don't yet support the "trap" mnemonic.
    // FIXME: Remove this special case when they do.
    if (!Subtarget->isTargetMachO()) {
      //.long 0xe7ffdefe @ trap
      uint32_t Val = 0xe7ffdefeUL;
      OutStreamer.AddComment("trap");
      OutStreamer.EmitIntValue(Val, 4);
      return;
    }
    break;
  }
  case ARM::TRAPNaCl: {
    //.long 0xe7fedef0 @ trap
    uint32_t Val = 0xe7fedef0UL;
    OutStreamer.AddComment("trap");
    OutStreamer.EmitIntValue(Val, 4);
    return;
  }
  case ARM::tTRAP: {
    // Non-Darwin binutils don't yet support the "trap" mnemonic.
    // FIXME: Remove this special case when they do.
    if (!Subtarget->isTargetMachO()) {
      //.short 57086 @ trap
      uint16_t Val = 0xdefe;
      OutStreamer.AddComment("trap");
      OutStreamer.EmitIntValue(Val, 2);
      return;
    }
    break;
  }
  case ARM::t2Int_eh_sjlj_setjmp:
  case ARM::t2Int_eh_sjlj_setjmp_nofp:
  case ARM::tInt_eh_sjlj_setjmp: {
    // Two incoming args: GPR:$src, GPR:$val
    // mov $val, pc
    // adds $val, #7
    // str $val, [$src, #4]
    // movs r0, #0
    // b 1f
    // movs r0, #1
    // 1:
    unsigned SrcReg = MI->getOperand(0).getReg();
    unsigned ValReg = MI->getOperand(1).getReg();
    MCSymbol *Label = GetARMSJLJEHLabel();
    OutStreamer.AddComment("eh_setjmp begin");
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::tMOVr)
      .addReg(ValReg)
      .addReg(ARM::PC)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::tADDi3)
      .addReg(ValReg)
      // 's' bit operand
      .addReg(ARM::CPSR)
      .addReg(ValReg)
      .addImm(7)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::tSTRi)
      .addReg(ValReg)
      .addReg(SrcReg)
      // The offset immediate is #4. The operand value is scaled by 4 for the
      // tSTR instruction.
      .addImm(1)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::tMOVi8)
      .addReg(ARM::R0)
      .addReg(ARM::CPSR)
      .addImm(0)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    const MCExpr *SymbolExpr = MCSymbolRefExpr::Create(Label, OutContext);
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::tB)
      .addExpr(SymbolExpr)
      .addImm(ARMCC::AL)
      .addReg(0));

    OutStreamer.AddComment("eh_setjmp end");
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::tMOVi8)
      .addReg(ARM::R0)
      .addReg(ARM::CPSR)
      .addImm(1)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    OutStreamer.EmitLabel(Label);
    return;
  }

  case ARM::Int_eh_sjlj_setjmp_nofp:
  case ARM::Int_eh_sjlj_setjmp: {
    // Two incoming args: GPR:$src, GPR:$val
    // add $val, pc, #8
    // str $val, [$src, #+4]
    // mov r0, #0
    // add pc, pc, #0
    // mov r0, #1
    unsigned SrcReg = MI->getOperand(0).getReg();
    unsigned ValReg = MI->getOperand(1).getReg();

    OutStreamer.AddComment("eh_setjmp begin");
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::ADDri)
      .addReg(ValReg)
      .addReg(ARM::PC)
      .addImm(8)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0)
      // 's' bit operand (always reg0 for this).
      .addReg(0));

    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::STRi12)
      .addReg(ValReg)
      .addReg(SrcReg)
      .addImm(4)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::MOVi)
      .addReg(ARM::R0)
      .addImm(0)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0)
      // 's' bit operand (always reg0 for this).
      .addReg(0));

    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::ADDri)
      .addReg(ARM::PC)
      .addReg(ARM::PC)
      .addImm(0)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0)
      // 's' bit operand (always reg0 for this).
      .addReg(0));

    OutStreamer.AddComment("eh_setjmp end");
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::MOVi)
      .addReg(ARM::R0)
      .addImm(1)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0)
      // 's' bit operand (always reg0 for this).
      .addReg(0));
    return;
  }
  case ARM::Int_eh_sjlj_longjmp: {
    // ldr sp, [$src, #8]
    // ldr $scratch, [$src, #4]
    // ldr r7, [$src]
    // bx $scratch
    unsigned SrcReg = MI->getOperand(0).getReg();
    unsigned ScratchReg = MI->getOperand(1).getReg();
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::LDRi12)
      .addReg(ARM::SP)
      .addReg(SrcReg)
      .addImm(8)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::LDRi12)
      .addReg(ScratchReg)
      .addReg(SrcReg)
      .addImm(4)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::LDRi12)
      .addReg(ARM::R7)
      .addReg(SrcReg)
      .addImm(0)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::BX)
      .addReg(ScratchReg)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));
    return;
  }
  case ARM::tInt_eh_sjlj_longjmp: {
    // ldr $scratch, [$src, #8]
    // mov sp, $scratch
    // ldr $scratch, [$src, #4]
    // ldr r7, [$src]
    // bx $scratch
    unsigned SrcReg = MI->getOperand(0).getReg();
    unsigned ScratchReg = MI->getOperand(1).getReg();
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::tLDRi)
      .addReg(ScratchReg)
      .addReg(SrcReg)
      // The offset immediate is #8. The operand value is scaled by 4 for the
      // tLDR instruction.
      .addImm(2)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::tMOVr)
      .addReg(ARM::SP)
      .addReg(ScratchReg)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::tLDRi)
      .addReg(ScratchReg)
      .addReg(SrcReg)
      .addImm(1)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::tLDRi)
      .addReg(ARM::R7)
      .addReg(SrcReg)
      .addImm(0)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    EmitToStreamer(OutStreamer, MCInstBuilder(ARM::tBX)
      .addReg(ScratchReg)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));
    return;
  }
  }

  MCInst TmpInst;
  LowerARMMachineInstrToMCInst(MI, TmpInst, *this);

  EmitToStreamer(OutStreamer, TmpInst);
}

//===----------------------------------------------------------------------===//
// Target Registry Stuff
//===----------------------------------------------------------------------===//

// Force static initialization.
extern "C" void LLVMInitializeARMAsmPrinter() {
  RegisterAsmPrinter<ARMAsmPrinter> X(TheARMTarget);
  RegisterAsmPrinter<ARMAsmPrinter> Y(TheThumbTarget);
}
