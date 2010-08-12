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
#include "ARM.h"
#include "ARMBuildAttrs.h"
#include "ARMAddressingModes.h"
#include "ARMConstantPoolValue.h"
#include "AsmPrinter/ARMInstPrinter.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMMCInstLower.h"
#include "ARMTargetMachine.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
using namespace llvm;

static cl::opt<bool>
EnableMCInst("enable-arm-mcinst-printer", cl::Hidden,
            cl::desc("enable experimental asmprinter gunk in the arm backend"));

namespace llvm {
  namespace ARM {
    enum DW_ISA {
      DW_ISA_ARM_thumb = 1,
      DW_ISA_ARM_arm = 2
    };
  }
}

namespace {
  class ARMAsmPrinter : public AsmPrinter {

    /// Subtarget - Keep a pointer to the ARMSubtarget around so that we can
    /// make the right decision when printing asm code for different targets.
    const ARMSubtarget *Subtarget;

    /// AFI - Keep a pointer to ARMFunctionInfo for the current
    /// MachineFunction.
    ARMFunctionInfo *AFI;

    /// MCP - Keep a pointer to constantpool entries of the current
    /// MachineFunction.
    const MachineConstantPool *MCP;

  public:
    explicit ARMAsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
      : AsmPrinter(TM, Streamer), AFI(NULL), MCP(NULL) {
      Subtarget = &TM.getSubtarget<ARMSubtarget>();
    }

    virtual const char *getPassName() const {
      return "ARM Assembly Printer";
    }
    
    void printInstructionThroughMCStreamer(const MachineInstr *MI);
    

    void printOperand(const MachineInstr *MI, int OpNum, raw_ostream &O,
                      const char *Modifier = 0);
    void printSOImmOperand(const MachineInstr *MI, int OpNum, raw_ostream &O);
    void printSOImm2PartOperand(const MachineInstr *MI, int OpNum,
                                raw_ostream &O);
    void printSORegOperand(const MachineInstr *MI, int OpNum,
                           raw_ostream &O);
    void printAddrMode2Operand(const MachineInstr *MI, int OpNum,
                               raw_ostream &O);
    void printAddrMode2OffsetOperand(const MachineInstr *MI, int OpNum,
                                     raw_ostream &O);
    void printAddrMode3Operand(const MachineInstr *MI, int OpNum,
                               raw_ostream &O);
    void printAddrMode3OffsetOperand(const MachineInstr *MI, int OpNum,
                                     raw_ostream &O);
    void printAddrMode4Operand(const MachineInstr *MI, int OpNum,raw_ostream &O,
                               const char *Modifier = 0);
    void printAddrMode5Operand(const MachineInstr *MI, int OpNum,raw_ostream &O,
                               const char *Modifier = 0);
    void printAddrMode6Operand(const MachineInstr *MI, int OpNum,
                               raw_ostream &O);
    void printAddrMode6OffsetOperand(const MachineInstr *MI, int OpNum,
                                     raw_ostream &O);
    void printAddrModePCOperand(const MachineInstr *MI, int OpNum,
                                raw_ostream &O,
                                const char *Modifier = 0);
    void printBitfieldInvMaskImmOperand(const MachineInstr *MI, int OpNum,
                                        raw_ostream &O);
    void printMemBOption(const MachineInstr *MI, int OpNum,
                         raw_ostream &O);
    void printSatShiftOperand(const MachineInstr *MI, int OpNum,
                              raw_ostream &O);

    void printThumbS4ImmOperand(const MachineInstr *MI, int OpNum,
                                raw_ostream &O);
    void printThumbITMask(const MachineInstr *MI, int OpNum, raw_ostream &O);
    void printThumbAddrModeRROperand(const MachineInstr *MI, int OpNum,
                                     raw_ostream &O);
    void printThumbAddrModeRI5Operand(const MachineInstr *MI, int OpNum,
                                      raw_ostream &O,
                                      unsigned Scale);
    void printThumbAddrModeS1Operand(const MachineInstr *MI, int OpNum,
                                     raw_ostream &O);
    void printThumbAddrModeS2Operand(const MachineInstr *MI, int OpNum,
                                     raw_ostream &O);
    void printThumbAddrModeS4Operand(const MachineInstr *MI, int OpNum,
                                     raw_ostream &O);
    void printThumbAddrModeSPOperand(const MachineInstr *MI, int OpNum,
                                     raw_ostream &O);

    void printT2SOOperand(const MachineInstr *MI, int OpNum, raw_ostream &O);
    void printT2AddrModeImm12Operand(const MachineInstr *MI, int OpNum,
                                     raw_ostream &O);
    void printT2AddrModeImm8Operand(const MachineInstr *MI, int OpNum,
                                    raw_ostream &O);
    void printT2AddrModeImm8s4Operand(const MachineInstr *MI, int OpNum,
                                      raw_ostream &O);
    void printT2AddrModeImm8OffsetOperand(const MachineInstr *MI, int OpNum,
                                          raw_ostream &O);
    void printT2AddrModeImm8s4OffsetOperand(const MachineInstr *MI, int OpNum,
                                            raw_ostream &O) {}
    void printT2AddrModeSoRegOperand(const MachineInstr *MI, int OpNum,
                                     raw_ostream &O);

    void printCPSOptionOperand(const MachineInstr *MI, int OpNum,
                               raw_ostream &O) {}
    void printMSRMaskOperand(const MachineInstr *MI, int OpNum,
                             raw_ostream &O) {}
    void printNegZeroOperand(const MachineInstr *MI, int OpNum,
                             raw_ostream &O) {}
    void printPredicateOperand(const MachineInstr *MI, int OpNum,
                               raw_ostream &O);
    void printMandatoryPredicateOperand(const MachineInstr *MI, int OpNum,
                                        raw_ostream &O);
    void printSBitModifierOperand(const MachineInstr *MI, int OpNum,
                                  raw_ostream &O);
    void printPCLabel(const MachineInstr *MI, int OpNum,
                      raw_ostream &O);
    void printRegisterList(const MachineInstr *MI, int OpNum,
                           raw_ostream &O);
    void printCPInstOperand(const MachineInstr *MI, int OpNum,
                            raw_ostream &O,
                            const char *Modifier);
    void printJTBlockOperand(const MachineInstr *MI, int OpNum,
                             raw_ostream &O);
    void printJT2BlockOperand(const MachineInstr *MI, int OpNum,
                              raw_ostream &O);
    void printTBAddrMode(const MachineInstr *MI, int OpNum,
                         raw_ostream &O);
    void printNoHashImmediate(const MachineInstr *MI, int OpNum,
                              raw_ostream &O);
    void printVFPf32ImmOperand(const MachineInstr *MI, int OpNum,
                               raw_ostream &O);
    void printVFPf64ImmOperand(const MachineInstr *MI, int OpNum,
                               raw_ostream &O);
    void printNEONModImmOperand(const MachineInstr *MI, int OpNum,
                                raw_ostream &O);

    virtual bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNum,
                                 unsigned AsmVariant, const char *ExtraCode,
                                 raw_ostream &O);
    virtual bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNum,
                                       unsigned AsmVariant,
                                       const char *ExtraCode, raw_ostream &O);

    void printInstruction(const MachineInstr *MI, raw_ostream &O); // autogen
    static const char *getRegisterName(unsigned RegNo);

    virtual void EmitInstruction(const MachineInstr *MI);
    bool runOnMachineFunction(MachineFunction &F);
    
    virtual void EmitConstantPool() {} // we emit constant pools customly!
    virtual void EmitFunctionEntryLabel();
    void EmitStartOfAsmFile(Module &M);
    void EmitEndOfAsmFile(Module &M);

    MachineLocation getDebugValueLocation(const MachineInstr *MI) const {
      MachineLocation Location;
      assert (MI->getNumOperands() == 4 && "Invalid no. of machine operands!");
      // Frame address.  Currently handles register +- offset only.
      if (MI->getOperand(0).isReg() && MI->getOperand(1).isImm())
        Location.set(MI->getOperand(0).getReg(), MI->getOperand(1).getImm());
      else {
        DEBUG(dbgs() << "DBG_VALUE instruction ignored! " << *MI << "\n");
      }
      return Location;
    }

    virtual unsigned getISAEncoding() {
      // ARM/Darwin adds ISA to the DWARF info for each function.
      if (!Subtarget->isTargetDarwin())
        return 0;
      return Subtarget->isThumb() ?
        llvm::ARM::DW_ISA_ARM_thumb : llvm::ARM::DW_ISA_ARM_arm;
    }

    MCSymbol *GetARMSetPICJumpTableLabel2(unsigned uid, unsigned uid2,
                                          const MachineBasicBlock *MBB) const;
    MCSymbol *GetARMJTIPICJumpTableLabel2(unsigned uid, unsigned uid2) const;

    /// EmitMachineConstantPoolValue - Print a machine constantpool value to
    /// the .s file.
    virtual void EmitMachineConstantPoolValue(MachineConstantPoolValue *MCPV) {
      SmallString<128> Str;
      raw_svector_ostream OS(Str);
      EmitMachineConstantPoolValue(MCPV, OS);
      OutStreamer.EmitRawText(OS.str());
    }
    
    void EmitMachineConstantPoolValue(MachineConstantPoolValue *MCPV,
                                      raw_ostream &O) {
      switch (TM.getTargetData()->getTypeAllocSize(MCPV->getType())) {
      case 1: O << MAI->getData8bitsDirective(0); break;
      case 2: O << MAI->getData16bitsDirective(0); break;
      case 4: O << MAI->getData32bitsDirective(0); break;
      default: assert(0 && "Unknown CPV size");
      }

      ARMConstantPoolValue *ACPV = static_cast<ARMConstantPoolValue*>(MCPV);

      if (ACPV->isLSDA()) {
        O << MAI->getPrivateGlobalPrefix() << "_LSDA_" << getFunctionNumber();
      } else if (ACPV->isBlockAddress()) {
        O << *GetBlockAddressSymbol(ACPV->getBlockAddress());
      } else if (ACPV->isGlobalValue()) {
        const GlobalValue *GV = ACPV->getGV();
        bool isIndirect = Subtarget->isTargetDarwin() &&
          Subtarget->GVIsIndirectSymbol(GV, TM.getRelocationModel());
        if (!isIndirect)
          O << *Mang->getSymbol(GV);
        else {
          // FIXME: Remove this when Darwin transition to @GOT like syntax.
          MCSymbol *Sym = GetSymbolWithGlobalValueBase(GV, "$non_lazy_ptr");
          O << *Sym;
          
          MachineModuleInfoMachO &MMIMachO =
            MMI->getObjFileInfo<MachineModuleInfoMachO>();
          MachineModuleInfoImpl::StubValueTy &StubSym =
            GV->hasHiddenVisibility() ? MMIMachO.getHiddenGVStubEntry(Sym) :
                                        MMIMachO.getGVStubEntry(Sym);
          if (StubSym.getPointer() == 0)
            StubSym = MachineModuleInfoImpl::
              StubValueTy(Mang->getSymbol(GV), !GV->hasInternalLinkage());
        }
      } else {
        assert(ACPV->isExtSymbol() && "unrecognized constant pool value");
        O << *GetExternalSymbolSymbol(ACPV->getSymbol());
      }

      if (ACPV->hasModifier()) O << "(" << ACPV->getModifier() << ")";
      if (ACPV->getPCAdjustment() != 0) {
        O << "-(" << MAI->getPrivateGlobalPrefix() << "PC"
          << getFunctionNumber() << "_"  << ACPV->getLabelId()
          << "+" << (unsigned)ACPV->getPCAdjustment();
         if (ACPV->mustAddCurrentAddress())
           O << "-.";
         O << ')';
      }
    }
  };
} // end of anonymous namespace

#include "ARMGenAsmWriter.inc"

void ARMAsmPrinter::EmitFunctionEntryLabel() {
  if (AFI->isThumbFunction()) {
    OutStreamer.EmitRawText(StringRef("\t.code\t16"));
    if (!Subtarget->isTargetDarwin())
      OutStreamer.EmitRawText(StringRef("\t.thumb_func"));
    else {
      // This needs to emit to a temporary string to get properly quoted
      // MCSymbols when they have spaces in them.
      SmallString<128> Tmp;
      raw_svector_ostream OS(Tmp);
      OS << "\t.thumb_func\t" << *CurrentFnSym;
      OutStreamer.EmitRawText(OS.str());
    }
  }
  
  OutStreamer.EmitLabel(CurrentFnSym);
}

/// runOnMachineFunction - This uses the printInstruction()
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
  default:
    assert(0 && "<unknown operand type>");
  case MachineOperand::MO_Register: {
    unsigned Reg = MO.getReg();
    assert(TargetRegisterInfo::isPhysicalRegister(Reg));
    if (Modifier && strcmp(Modifier, "dregpair") == 0) {
      unsigned DRegLo = TM.getRegisterInfo()->getSubReg(Reg, ARM::dsub_0);
      unsigned DRegHi = TM.getRegisterInfo()->getSubReg(Reg, ARM::dsub_1);
      O << '{'
        << getRegisterName(DRegLo) << ", " << getRegisterName(DRegHi)
        << '}';
    } else if (Modifier && strcmp(Modifier, "lane") == 0) {
      unsigned RegNum = ARMRegisterInfo::getRegisterNumbering(Reg);
      unsigned DReg =
        TM.getRegisterInfo()->getMatchingSuperReg(Reg,
          RegNum & 1 ? ARM::ssub_1 : ARM::ssub_0, &ARM::DPR_VFP2RegClass);
      O << getRegisterName(DReg) << '[' << (RegNum & 1) << ']';
    } else {
      assert(!MO.getSubReg() && "Subregs should be eliminated!");
      O << getRegisterName(Reg);
    }
    break;
  }
  case MachineOperand::MO_Immediate: {
    int64_t Imm = MO.getImm();
    O << '#';
    if ((Modifier && strcmp(Modifier, "lo16") == 0) ||
        (TF & ARMII::MO_LO16))
      O << ":lower16:";
    else if ((Modifier && strcmp(Modifier, "hi16") == 0) ||
             (TF & ARMII::MO_HI16))
      O << ":upper16:";
    O << Imm;
    break;
  }
  case MachineOperand::MO_MachineBasicBlock:
    O << *MO.getMBB()->getSymbol();
    return;
  case MachineOperand::MO_GlobalAddress: {
    bool isCallOp = Modifier && !strcmp(Modifier, "call");
    const GlobalValue *GV = MO.getGlobal();

    if ((Modifier && strcmp(Modifier, "lo16") == 0) ||
        (TF & ARMII::MO_LO16))
      O << ":lower16:";
    else if ((Modifier && strcmp(Modifier, "hi16") == 0) ||
             (TF & ARMII::MO_HI16))
      O << ":upper16:";
    O << *Mang->getSymbol(GV);

    printOffset(MO.getOffset(), O);

    if (isCallOp && Subtarget->isTargetELF() &&
        TM.getRelocationModel() == Reloc::PIC_)
      O << "(PLT)";
    break;
  }
  case MachineOperand::MO_ExternalSymbol: {
    bool isCallOp = Modifier && !strcmp(Modifier, "call");
    O << *GetExternalSymbolSymbol(MO.getSymbolName());
    
    if (isCallOp && Subtarget->isTargetELF() &&
        TM.getRelocationModel() == Reloc::PIC_)
      O << "(PLT)";
    break;
  }
  case MachineOperand::MO_ConstantPoolIndex:
    O << *GetCPISymbol(MO.getIndex());
    break;
  case MachineOperand::MO_JumpTableIndex:
    O << *GetJTISymbol(MO.getIndex());
    break;
  }
}

static void printSOImm(raw_ostream &O, int64_t V, bool VerboseAsm,
                       const MCAsmInfo *MAI) {
  // Break it up into two parts that make up a shifter immediate.
  V = ARM_AM::getSOImmVal(V);
  assert(V != -1 && "Not a valid so_imm value!");

  unsigned Imm = ARM_AM::getSOImmValImm(V);
  unsigned Rot = ARM_AM::getSOImmValRot(V);

  // Print low-level immediate formation info, per
  // A5.1.3: "Data-processing operands - Immediate".
  if (Rot) {
    O << "#" << Imm << ", " << Rot;
    // Pretty printed version.
    if (VerboseAsm) {
      O << "\t" << MAI->getCommentString() << ' ';
      O << (int)ARM_AM::rotr32(Imm, Rot);
    }
  } else {
    O << "#" << Imm;
  }
}

/// printSOImmOperand - SOImm is 4-bit rotate amount in bits 8-11 with 8-bit
/// immediate in bits 0-7.
void ARMAsmPrinter::printSOImmOperand(const MachineInstr *MI, int OpNum,
                                      raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(OpNum);
  assert(MO.isImm() && "Not a valid so_imm value!");
  printSOImm(O, MO.getImm(), isVerbose(), MAI);
}

/// printSOImm2PartOperand - SOImm is broken into two pieces using a 'mov'
/// followed by an 'orr' to materialize.
void ARMAsmPrinter::printSOImm2PartOperand(const MachineInstr *MI, int OpNum,
                                           raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(OpNum);
  assert(MO.isImm() && "Not a valid so_imm value!");
  unsigned V1 = ARM_AM::getSOImmTwoPartFirst(MO.getImm());
  unsigned V2 = ARM_AM::getSOImmTwoPartSecond(MO.getImm());
  printSOImm(O, V1, isVerbose(), MAI);
  O << "\n\torr";
  printPredicateOperand(MI, 2, O);
  O << "\t";
  printOperand(MI, 0, O);
  O << ", ";
  printOperand(MI, 0, O);
  O << ", ";
  printSOImm(O, V2, isVerbose(), MAI);
}

// so_reg is a 4-operand unit corresponding to register forms of the A5.1
// "Addressing Mode 1 - Data-processing operands" forms.  This includes:
//    REG 0   0           - e.g. R5
//    REG REG 0,SH_OPC    - e.g. R5, ROR R3
//    REG 0   IMM,SH_OPC  - e.g. R5, LSL #3
void ARMAsmPrinter::printSORegOperand(const MachineInstr *MI, int Op,
                                      raw_ostream &O) {
  const MachineOperand &MO1 = MI->getOperand(Op);
  const MachineOperand &MO2 = MI->getOperand(Op+1);
  const MachineOperand &MO3 = MI->getOperand(Op+2);

  O << getRegisterName(MO1.getReg());

  // Print the shift opc.
  ARM_AM::ShiftOpc ShOpc = ARM_AM::getSORegShOp(MO3.getImm());
  O << ", " << ARM_AM::getShiftOpcStr(ShOpc);
  if (MO2.getReg()) {
    O << ' ' << getRegisterName(MO2.getReg());
    assert(ARM_AM::getSORegOffset(MO3.getImm()) == 0);
  } else if (ShOpc != ARM_AM::rrx) {
    O << " #" << ARM_AM::getSORegOffset(MO3.getImm());
  }
}

void ARMAsmPrinter::printAddrMode2Operand(const MachineInstr *MI, int Op,
                                          raw_ostream &O) {
  const MachineOperand &MO1 = MI->getOperand(Op);
  const MachineOperand &MO2 = MI->getOperand(Op+1);
  const MachineOperand &MO3 = MI->getOperand(Op+2);

  if (!MO1.isReg()) {   // FIXME: This is for CP entries, but isn't right.
    printOperand(MI, Op, O);
    return;
  }

  O << "[" << getRegisterName(MO1.getReg());

  if (!MO2.getReg()) {
    if (ARM_AM::getAM2Offset(MO3.getImm())) // Don't print +0.
      O << ", #"
        << ARM_AM::getAddrOpcStr(ARM_AM::getAM2Op(MO3.getImm()))
        << ARM_AM::getAM2Offset(MO3.getImm());
    O << "]";
    return;
  }

  O << ", "
    << ARM_AM::getAddrOpcStr(ARM_AM::getAM2Op(MO3.getImm()))
    << getRegisterName(MO2.getReg());

  if (unsigned ShImm = ARM_AM::getAM2Offset(MO3.getImm()))
    O << ", "
      << ARM_AM::getShiftOpcStr(ARM_AM::getAM2ShiftOpc(MO3.getImm()))
      << " #" << ShImm;
  O << "]";
}

void ARMAsmPrinter::printAddrMode2OffsetOperand(const MachineInstr *MI, int Op,
                                                raw_ostream &O) {
  const MachineOperand &MO1 = MI->getOperand(Op);
  const MachineOperand &MO2 = MI->getOperand(Op+1);

  if (!MO1.getReg()) {
    unsigned ImmOffs = ARM_AM::getAM2Offset(MO2.getImm());
    O << "#"
      << ARM_AM::getAddrOpcStr(ARM_AM::getAM2Op(MO2.getImm()))
      << ImmOffs;
    return;
  }

  O << ARM_AM::getAddrOpcStr(ARM_AM::getAM2Op(MO2.getImm()))
    << getRegisterName(MO1.getReg());

  if (unsigned ShImm = ARM_AM::getAM2Offset(MO2.getImm()))
    O << ", "
      << ARM_AM::getShiftOpcStr(ARM_AM::getAM2ShiftOpc(MO2.getImm()))
      << " #" << ShImm;
}

void ARMAsmPrinter::printAddrMode3Operand(const MachineInstr *MI, int Op,
                                          raw_ostream &O) {
  const MachineOperand &MO1 = MI->getOperand(Op);
  const MachineOperand &MO2 = MI->getOperand(Op+1);
  const MachineOperand &MO3 = MI->getOperand(Op+2);

  assert(TargetRegisterInfo::isPhysicalRegister(MO1.getReg()));
  O << "[" << getRegisterName(MO1.getReg());

  if (MO2.getReg()) {
    O << ", "
      << (char)ARM_AM::getAM3Op(MO3.getImm())
      << getRegisterName(MO2.getReg())
      << "]";
    return;
  }

  if (unsigned ImmOffs = ARM_AM::getAM3Offset(MO3.getImm()))
    O << ", #"
      << ARM_AM::getAddrOpcStr(ARM_AM::getAM3Op(MO3.getImm()))
      << ImmOffs;
  O << "]";
}

void ARMAsmPrinter::printAddrMode3OffsetOperand(const MachineInstr *MI, int Op,
                                                raw_ostream &O){
  const MachineOperand &MO1 = MI->getOperand(Op);
  const MachineOperand &MO2 = MI->getOperand(Op+1);

  if (MO1.getReg()) {
    O << (char)ARM_AM::getAM3Op(MO2.getImm())
      << getRegisterName(MO1.getReg());
    return;
  }

  unsigned ImmOffs = ARM_AM::getAM3Offset(MO2.getImm());
  O << "#"
    << ARM_AM::getAddrOpcStr(ARM_AM::getAM3Op(MO2.getImm()))
    << ImmOffs;
}

void ARMAsmPrinter::printAddrMode4Operand(const MachineInstr *MI, int Op,
                                          raw_ostream &O,
                                          const char *Modifier) {
  const MachineOperand &MO2 = MI->getOperand(Op+1);
  ARM_AM::AMSubMode Mode = ARM_AM::getAM4SubMode(MO2.getImm());
  if (Modifier && strcmp(Modifier, "submode") == 0) {
    O << ARM_AM::getAMSubModeStr(Mode);
  } else if (Modifier && strcmp(Modifier, "wide") == 0) {
    ARM_AM::AMSubMode Mode = ARM_AM::getAM4SubMode(MO2.getImm());
    if (Mode == ARM_AM::ia)
      O << ".w";
  } else {
    printOperand(MI, Op, O);
  }
}

void ARMAsmPrinter::printAddrMode5Operand(const MachineInstr *MI, int Op,
                                          raw_ostream &O,
                                          const char *Modifier) {
  const MachineOperand &MO1 = MI->getOperand(Op);
  const MachineOperand &MO2 = MI->getOperand(Op+1);

  if (!MO1.isReg()) {   // FIXME: This is for CP entries, but isn't right.
    printOperand(MI, Op, O);
    return;
  }

  assert(TargetRegisterInfo::isPhysicalRegister(MO1.getReg()));

  if (Modifier && strcmp(Modifier, "submode") == 0) {
    ARM_AM::AMSubMode Mode = ARM_AM::getAM5SubMode(MO2.getImm());
    O << ARM_AM::getAMSubModeStr(Mode);
    return;
  } else if (Modifier && strcmp(Modifier, "base") == 0) {
    // Used for FSTM{D|S} and LSTM{D|S} operations.
    O << getRegisterName(MO1.getReg());
    return;
  }

  O << "[" << getRegisterName(MO1.getReg());

  if (unsigned ImmOffs = ARM_AM::getAM5Offset(MO2.getImm())) {
    O << ", #"
      << ARM_AM::getAddrOpcStr(ARM_AM::getAM5Op(MO2.getImm()))
      << ImmOffs*4;
  }
  O << "]";
}

void ARMAsmPrinter::printAddrMode6Operand(const MachineInstr *MI, int Op,
                                          raw_ostream &O) {
  const MachineOperand &MO1 = MI->getOperand(Op);
  const MachineOperand &MO2 = MI->getOperand(Op+1);

  O << "[" << getRegisterName(MO1.getReg());
  if (MO2.getImm()) {
    // FIXME: Both darwin as and GNU as violate ARM docs here.
    O << ", :" << (MO2.getImm() << 3);
  }
  O << "]";
}

void ARMAsmPrinter::printAddrMode6OffsetOperand(const MachineInstr *MI, int Op,
                                                raw_ostream &O){
  const MachineOperand &MO = MI->getOperand(Op);
  if (MO.getReg() == 0)
    O << "!";
  else
    O << ", " << getRegisterName(MO.getReg());
}

void ARMAsmPrinter::printAddrModePCOperand(const MachineInstr *MI, int Op,
                                           raw_ostream &O,
                                           const char *Modifier) {
  if (Modifier && strcmp(Modifier, "label") == 0) {
    printPCLabel(MI, Op+1, O);
    return;
  }

  const MachineOperand &MO1 = MI->getOperand(Op);
  assert(TargetRegisterInfo::isPhysicalRegister(MO1.getReg()));
  O << "[pc, " << getRegisterName(MO1.getReg()) << "]";
}

void
ARMAsmPrinter::printBitfieldInvMaskImmOperand(const MachineInstr *MI, int Op,
                                              raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(Op);
  uint32_t v = ~MO.getImm();
  int32_t lsb = CountTrailingZeros_32(v);
  int32_t width = (32 - CountLeadingZeros_32 (v)) - lsb;
  assert(MO.isImm() && "Not a valid bf_inv_mask_imm value!");
  O << "#" << lsb << ", #" << width;
}

void
ARMAsmPrinter::printMemBOption(const MachineInstr *MI, int OpNum,
                               raw_ostream &O) {
  unsigned val = MI->getOperand(OpNum).getImm();
  O << ARM_MB::MemBOptToString(val);
}

void ARMAsmPrinter::printSatShiftOperand(const MachineInstr *MI, int OpNum,
                                         raw_ostream &O) {
  unsigned ShiftOp = MI->getOperand(OpNum).getImm();
  ARM_AM::ShiftOpc Opc = ARM_AM::getSORegShOp(ShiftOp);
  switch (Opc) {
  case ARM_AM::no_shift:
    return;
  case ARM_AM::lsl:
    O << ", lsl #";
    break;
  case ARM_AM::asr:
    O << ", asr #";
    break;
  default:
    assert(0 && "unexpected shift opcode for saturate shift operand");
  }
  O << ARM_AM::getSORegOffset(ShiftOp);
}

//===--------------------------------------------------------------------===//

void ARMAsmPrinter::printThumbS4ImmOperand(const MachineInstr *MI, int Op,
                                           raw_ostream &O) {
  O << "#" <<  MI->getOperand(Op).getImm() * 4;
}

void
ARMAsmPrinter::printThumbITMask(const MachineInstr *MI, int Op,
                                raw_ostream &O) {
  // (3 - the number of trailing zeros) is the number of then / else.
  unsigned Mask = MI->getOperand(Op).getImm();
  unsigned CondBit0 = Mask >> 4 & 1;
  unsigned NumTZ = CountTrailingZeros_32(Mask);
  assert(NumTZ <= 3 && "Invalid IT mask!");
  for (unsigned Pos = 3, e = NumTZ; Pos > e; --Pos) {
    bool T = ((Mask >> Pos) & 1) == CondBit0;
    if (T)
      O << 't';
    else
      O << 'e';
  }
}

void
ARMAsmPrinter::printThumbAddrModeRROperand(const MachineInstr *MI, int Op,
                                           raw_ostream &O) {
  const MachineOperand &MO1 = MI->getOperand(Op);
  const MachineOperand &MO2 = MI->getOperand(Op+1);
  O << "[" << getRegisterName(MO1.getReg());
  O << ", " << getRegisterName(MO2.getReg()) << "]";
}

void
ARMAsmPrinter::printThumbAddrModeRI5Operand(const MachineInstr *MI, int Op,
                                            raw_ostream &O,
                                            unsigned Scale) {
  const MachineOperand &MO1 = MI->getOperand(Op);
  const MachineOperand &MO2 = MI->getOperand(Op+1);
  const MachineOperand &MO3 = MI->getOperand(Op+2);

  if (!MO1.isReg()) {   // FIXME: This is for CP entries, but isn't right.
    printOperand(MI, Op, O);
    return;
  }

  O << "[" << getRegisterName(MO1.getReg());
  if (MO3.getReg())
    O << ", " << getRegisterName(MO3.getReg());
  else if (unsigned ImmOffs = MO2.getImm())
    O << ", #" << ImmOffs * Scale;
  O << "]";
}

void
ARMAsmPrinter::printThumbAddrModeS1Operand(const MachineInstr *MI, int Op,
                                           raw_ostream &O) {
  printThumbAddrModeRI5Operand(MI, Op, O, 1);
}
void
ARMAsmPrinter::printThumbAddrModeS2Operand(const MachineInstr *MI, int Op,
                                           raw_ostream &O) {
  printThumbAddrModeRI5Operand(MI, Op, O, 2);
}
void
ARMAsmPrinter::printThumbAddrModeS4Operand(const MachineInstr *MI, int Op,
                                           raw_ostream &O) {
  printThumbAddrModeRI5Operand(MI, Op, O, 4);
}

void ARMAsmPrinter::printThumbAddrModeSPOperand(const MachineInstr *MI,int Op,
                                                raw_ostream &O) {
  const MachineOperand &MO1 = MI->getOperand(Op);
  const MachineOperand &MO2 = MI->getOperand(Op+1);
  O << "[" << getRegisterName(MO1.getReg());
  if (unsigned ImmOffs = MO2.getImm())
    O << ", #" << ImmOffs*4;
  O << "]";
}

//===--------------------------------------------------------------------===//

// Constant shifts t2_so_reg is a 2-operand unit corresponding to the Thumb2
// register with shift forms.
// REG 0   0           - e.g. R5
// REG IMM, SH_OPC     - e.g. R5, LSL #3
void ARMAsmPrinter::printT2SOOperand(const MachineInstr *MI, int OpNum,
                                     raw_ostream &O) {
  const MachineOperand &MO1 = MI->getOperand(OpNum);
  const MachineOperand &MO2 = MI->getOperand(OpNum+1);

  unsigned Reg = MO1.getReg();
  assert(TargetRegisterInfo::isPhysicalRegister(Reg));
  O << getRegisterName(Reg);

  // Print the shift opc.
  assert(MO2.isImm() && "Not a valid t2_so_reg value!");
  ARM_AM::ShiftOpc ShOpc = ARM_AM::getSORegShOp(MO2.getImm());
  O << ", " << ARM_AM::getShiftOpcStr(ShOpc);
  if (ShOpc != ARM_AM::rrx)
    O << " #" << ARM_AM::getSORegOffset(MO2.getImm());
}

void ARMAsmPrinter::printT2AddrModeImm12Operand(const MachineInstr *MI,
                                                int OpNum,
                                                raw_ostream &O) {
  const MachineOperand &MO1 = MI->getOperand(OpNum);
  const MachineOperand &MO2 = MI->getOperand(OpNum+1);

  O << "[" << getRegisterName(MO1.getReg());

  unsigned OffImm = MO2.getImm();
  if (OffImm)  // Don't print +0.
    O << ", #" << OffImm;
  O << "]";
}

void ARMAsmPrinter::printT2AddrModeImm8Operand(const MachineInstr *MI,
                                               int OpNum,
                                               raw_ostream &O) {
  const MachineOperand &MO1 = MI->getOperand(OpNum);
  const MachineOperand &MO2 = MI->getOperand(OpNum+1);

  O << "[" << getRegisterName(MO1.getReg());

  int32_t OffImm = (int32_t)MO2.getImm();
  // Don't print +0.
  if (OffImm < 0)
    O << ", #-" << -OffImm;
  else if (OffImm > 0)
    O << ", #" << OffImm;
  O << "]";
}

void ARMAsmPrinter::printT2AddrModeImm8s4Operand(const MachineInstr *MI,
                                                 int OpNum,
                                                 raw_ostream &O) {
  const MachineOperand &MO1 = MI->getOperand(OpNum);
  const MachineOperand &MO2 = MI->getOperand(OpNum+1);

  O << "[" << getRegisterName(MO1.getReg());

  int32_t OffImm = (int32_t)MO2.getImm() / 4;
  // Don't print +0.
  if (OffImm < 0)
    O << ", #-" << -OffImm * 4;
  else if (OffImm > 0)
    O << ", #" << OffImm * 4;
  O << "]";
}

void ARMAsmPrinter::printT2AddrModeImm8OffsetOperand(const MachineInstr *MI,
                                                     int OpNum,
                                                     raw_ostream &O) {
  const MachineOperand &MO1 = MI->getOperand(OpNum);
  int32_t OffImm = (int32_t)MO1.getImm();
  // Don't print +0.
  if (OffImm < 0)
    O << "#-" << -OffImm;
  else if (OffImm > 0)
    O << "#" << OffImm;
}

void ARMAsmPrinter::printT2AddrModeSoRegOperand(const MachineInstr *MI,
                                                int OpNum,
                                                raw_ostream &O) {
  const MachineOperand &MO1 = MI->getOperand(OpNum);
  const MachineOperand &MO2 = MI->getOperand(OpNum+1);
  const MachineOperand &MO3 = MI->getOperand(OpNum+2);

  O << "[" << getRegisterName(MO1.getReg());

  assert(MO2.getReg() && "Invalid so_reg load / store address!");
  O << ", " << getRegisterName(MO2.getReg());

  unsigned ShAmt = MO3.getImm();
  if (ShAmt) {
    assert(ShAmt <= 3 && "Not a valid Thumb2 addressing mode!");
    O << ", lsl #" << ShAmt;
  }
  O << "]";
}


//===--------------------------------------------------------------------===//

void ARMAsmPrinter::printPredicateOperand(const MachineInstr *MI, int OpNum,
                                          raw_ostream &O) {
  ARMCC::CondCodes CC = (ARMCC::CondCodes)MI->getOperand(OpNum).getImm();
  if (CC != ARMCC::AL)
    O << ARMCondCodeToString(CC);
}

void ARMAsmPrinter::printMandatoryPredicateOperand(const MachineInstr *MI,
                                                   int OpNum,
                                                   raw_ostream &O) {
  ARMCC::CondCodes CC = (ARMCC::CondCodes)MI->getOperand(OpNum).getImm();
  O << ARMCondCodeToString(CC);
}

void ARMAsmPrinter::printSBitModifierOperand(const MachineInstr *MI, int OpNum,
                                             raw_ostream &O){
  unsigned Reg = MI->getOperand(OpNum).getReg();
  if (Reg) {
    assert(Reg == ARM::CPSR && "Expect ARM CPSR register!");
    O << 's';
  }
}

void ARMAsmPrinter::printPCLabel(const MachineInstr *MI, int OpNum,
                                 raw_ostream &O) {
  int Id = (int)MI->getOperand(OpNum).getImm();
  O << MAI->getPrivateGlobalPrefix()
    << "PC" << getFunctionNumber() << "_" << Id;
}

void ARMAsmPrinter::printRegisterList(const MachineInstr *MI, int OpNum,
                                      raw_ostream &O) {
  O << "{";
  for (unsigned i = OpNum, e = MI->getNumOperands(); i != e; ++i) {
    if (MI->getOperand(i).isImplicit())
      continue;
    if ((int)i != OpNum) O << ", ";
    printOperand(MI, i, O);
  }
  O << "}";
}

void ARMAsmPrinter::printCPInstOperand(const MachineInstr *MI, int OpNum,
                                       raw_ostream &O, const char *Modifier) {
  assert(Modifier && "This operand only works with a modifier!");
  // There are two aspects to a CONSTANTPOOL_ENTRY operand, the label and the
  // data itself.
  if (!strcmp(Modifier, "label")) {
    unsigned ID = MI->getOperand(OpNum).getImm();
    OutStreamer.EmitLabel(GetCPISymbol(ID));
  } else {
    assert(!strcmp(Modifier, "cpentry") && "Unknown modifier for CPE");
    unsigned CPI = MI->getOperand(OpNum).getIndex();

    const MachineConstantPoolEntry &MCPE = MCP->getConstants()[CPI];

    if (MCPE.isMachineConstantPoolEntry()) {
      EmitMachineConstantPoolValue(MCPE.Val.MachineCPVal);
    } else {
      EmitGlobalConstant(MCPE.Val.ConstVal);
    }
  }
}

MCSymbol *ARMAsmPrinter::
GetARMSetPICJumpTableLabel2(unsigned uid, unsigned uid2,
                            const MachineBasicBlock *MBB) const {
  SmallString<60> Name;
  raw_svector_ostream(Name) << MAI->getPrivateGlobalPrefix()
    << getFunctionNumber() << '_' << uid << '_' << uid2
    << "_set_" << MBB->getNumber();
  return OutContext.GetOrCreateSymbol(Name.str());
}

MCSymbol *ARMAsmPrinter::
GetARMJTIPICJumpTableLabel2(unsigned uid, unsigned uid2) const {
  SmallString<60> Name;
  raw_svector_ostream(Name) << MAI->getPrivateGlobalPrefix() << "JTI"
    << getFunctionNumber() << '_' << uid << '_' << uid2;
  return OutContext.GetOrCreateSymbol(Name.str());
}

void ARMAsmPrinter::printJTBlockOperand(const MachineInstr *MI, int OpNum,
                                        raw_ostream &O) {
  assert(!Subtarget->isThumb2() && "Thumb2 should use double-jump jumptables!");

  const MachineOperand &MO1 = MI->getOperand(OpNum);
  const MachineOperand &MO2 = MI->getOperand(OpNum+1); // Unique Id
  
  unsigned JTI = MO1.getIndex();
  MCSymbol *JTISymbol = GetARMJTIPICJumpTableLabel2(JTI, MO2.getImm());
  // Can't use EmitLabel until instprinter happens, label comes out in the wrong
  // order.
  O << "\n" << *JTISymbol << ":\n";

  const char *JTEntryDirective = MAI->getData32bitsDirective();

  const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  const std::vector<MachineBasicBlock*> &JTBBs = JT[JTI].MBBs;
  bool UseSet= MAI->hasSetDirective() && TM.getRelocationModel() == Reloc::PIC_;
  SmallPtrSet<MachineBasicBlock*, 8> JTSets;
  for (unsigned i = 0, e = JTBBs.size(); i != e; ++i) {
    MachineBasicBlock *MBB = JTBBs[i];
    bool isNew = JTSets.insert(MBB);

    if (UseSet && isNew) {
      O << "\t.set\t"
        << *GetARMSetPICJumpTableLabel2(JTI, MO2.getImm(), MBB) << ','
        << *MBB->getSymbol() << '-' << *JTISymbol << '\n';
    }

    O << JTEntryDirective << ' ';
    if (UseSet)
      O << *GetARMSetPICJumpTableLabel2(JTI, MO2.getImm(), MBB);
    else if (TM.getRelocationModel() == Reloc::PIC_)
      O << *MBB->getSymbol() << '-' << *JTISymbol;
    else
      O << *MBB->getSymbol();

    if (i != e-1)
      O << '\n';
  }
}

void ARMAsmPrinter::printJT2BlockOperand(const MachineInstr *MI, int OpNum,
                                         raw_ostream &O) {
  const MachineOperand &MO1 = MI->getOperand(OpNum);
  const MachineOperand &MO2 = MI->getOperand(OpNum+1); // Unique Id
  unsigned JTI = MO1.getIndex();
  
  MCSymbol *JTISymbol = GetARMJTIPICJumpTableLabel2(JTI, MO2.getImm());
  
  // Can't use EmitLabel until instprinter happens, label comes out in the wrong
  // order.
  O << "\n" << *JTISymbol << ":\n";

  const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  const std::vector<MachineBasicBlock*> &JTBBs = JT[JTI].MBBs;
  bool ByteOffset = false, HalfWordOffset = false;
  if (MI->getOpcode() == ARM::t2TBB)
    ByteOffset = true;
  else if (MI->getOpcode() == ARM::t2TBH)
    HalfWordOffset = true;

  for (unsigned i = 0, e = JTBBs.size(); i != e; ++i) {
    MachineBasicBlock *MBB = JTBBs[i];
    if (ByteOffset)
      O << MAI->getData8bitsDirective();
    else if (HalfWordOffset)
      O << MAI->getData16bitsDirective();
    
    if (ByteOffset || HalfWordOffset)
      O << '(' << *MBB->getSymbol() << "-" << *JTISymbol << ")/2";
    else
      O << "\tb.w " << *MBB->getSymbol();

    if (i != e-1)
      O << '\n';
  }
}

void ARMAsmPrinter::printTBAddrMode(const MachineInstr *MI, int OpNum,
                                    raw_ostream &O) {
  O << "[pc, " << getRegisterName(MI->getOperand(OpNum).getReg());
  if (MI->getOpcode() == ARM::t2TBH)
    O << ", lsl #1";
  O << ']';
}

void ARMAsmPrinter::printNoHashImmediate(const MachineInstr *MI, int OpNum,
                                         raw_ostream &O) {
  O << MI->getOperand(OpNum).getImm();
}

void ARMAsmPrinter::printVFPf32ImmOperand(const MachineInstr *MI, int OpNum,
                                          raw_ostream &O) {
  const ConstantFP *FP = MI->getOperand(OpNum).getFPImm();
  O << '#' << FP->getValueAPF().convertToFloat();
  if (isVerbose()) {
    O << "\t\t" << MAI->getCommentString() << ' ';
    WriteAsOperand(O, FP, /*PrintType=*/false);
  }
}

void ARMAsmPrinter::printVFPf64ImmOperand(const MachineInstr *MI, int OpNum,
                                          raw_ostream &O) {
  const ConstantFP *FP = MI->getOperand(OpNum).getFPImm();
  O << '#' << FP->getValueAPF().convertToDouble();
  if (isVerbose()) {
    O << "\t\t" << MAI->getCommentString() << ' ';
    WriteAsOperand(O, FP, /*PrintType=*/false);
  }
}

void ARMAsmPrinter::printNEONModImmOperand(const MachineInstr *MI, int OpNum,
                                           raw_ostream &O) {
  unsigned EncodedImm = MI->getOperand(OpNum).getImm();
  unsigned EltBits;
  uint64_t Val = ARM_AM::decodeNEONModImm(EncodedImm, EltBits);
  O << "#0x" << utohexstr(Val);
}

bool ARMAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNum,
                                    unsigned AsmVariant, const char *ExtraCode,
                                    raw_ostream &O) {
  // Does this asm operand have a single letter operand modifier?
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default: return true;  // Unknown modifier.
    case 'a': // Print as a memory address.
      if (MI->getOperand(OpNum).isReg()) {
        O << "[" << getRegisterName(MI->getOperand(OpNum).getReg()) << "]";
        return false;
      }
      // Fallthrough
    case 'c': // Don't print "#" before an immediate operand.
      if (!MI->getOperand(OpNum).isImm())
        return true;
      printNoHashImmediate(MI, OpNum, O);
      return false;
    case 'P': // Print a VFP double precision register.
    case 'q': // Print a NEON quad precision register.
      printOperand(MI, OpNum, O);
      return false;
    case 'Q':
    case 'R':
    case 'H':
      report_fatal_error("llvm does not support 'Q', 'R', and 'H' modifiers!");
      return true;
    }
  }

  printOperand(MI, OpNum, O);
  return false;
}

bool ARMAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                          unsigned OpNum, unsigned AsmVariant,
                                          const char *ExtraCode,
                                          raw_ostream &O) {
  if (ExtraCode && ExtraCode[0])
    return true; // Unknown modifier.

  const MachineOperand &MO = MI->getOperand(OpNum);
  assert(MO.isReg() && "unexpected inline asm memory operand");
  O << "[" << getRegisterName(MO.getReg()) << "]";
  return false;
}

void ARMAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  if (EnableMCInst) {
    printInstructionThroughMCStreamer(MI);
    return;
  }
  
  if (MI->getOpcode() == ARM::CONSTPOOL_ENTRY)
    EmitAlignment(2);
  
  SmallString<128> Str;
  raw_svector_ostream OS(Str);
  if (MI->getOpcode() == ARM::DBG_VALUE) {
    unsigned NOps = MI->getNumOperands();
    assert(NOps==4);
    OS << '\t' << MAI->getCommentString() << "DEBUG_VALUE: ";
    // cast away const; DIetc do not take const operands for some reason.
    DIVariable V(const_cast<MDNode *>(MI->getOperand(NOps-1).getMetadata()));
    OS << V.getName();
    OS << " <- ";
    // Frame address.  Currently handles register +- offset only.
    assert(MI->getOperand(0).isReg() && MI->getOperand(1).isImm());
    OS << '['; printOperand(MI, 0, OS); OS << '+'; printOperand(MI, 1, OS);
    OS << ']';
    OS << "+";
    printOperand(MI, NOps-2, OS);
    OutStreamer.EmitRawText(OS.str());
    return;
  }

  printInstruction(MI, OS);
  OutStreamer.EmitRawText(OS.str());
  
  // Make sure the instruction that follows TBB is 2-byte aligned.
  // FIXME: Constant island pass should insert an "ALIGN" instruction instead.
  if (MI->getOpcode() == ARM::t2TBB)
    EmitAlignment(1);
}

void ARMAsmPrinter::EmitStartOfAsmFile(Module &M) {
  if (Subtarget->isTargetDarwin()) {
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
      OutStreamer.SwitchSection(TLOFMacho.getTextSection());
      OutStreamer.SwitchSection(TLOFMacho.getTextCoalSection());
      OutStreamer.SwitchSection(TLOFMacho.getConstTextCoalSection());
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
  }

  // Use unified assembler syntax.
  OutStreamer.EmitRawText(StringRef("\t.syntax unified"));

  // Emit ARM Build Attributes
  if (Subtarget->isTargetELF()) {
    // CPU Type
    std::string CPUString = Subtarget->getCPUString();
    if (CPUString != "generic")
      OutStreamer.EmitRawText("\t.cpu " + Twine(CPUString));

    // FIXME: Emit FPU type
    if (Subtarget->hasVFP2())
      OutStreamer.EmitRawText("\t.eabi_attribute " +
                              Twine(ARMBuildAttrs::VFP_arch) + ", 2");

    // Signal various FP modes.
    if (!UnsafeFPMath) {
      OutStreamer.EmitRawText("\t.eabi_attribute " +
                              Twine(ARMBuildAttrs::ABI_FP_denormal) + ", 1");
      OutStreamer.EmitRawText("\t.eabi_attribute " +
                              Twine(ARMBuildAttrs::ABI_FP_exceptions) + ", 1");
    }
    
    if (NoInfsFPMath && NoNaNsFPMath)
      OutStreamer.EmitRawText("\t.eabi_attribute " +
                              Twine(ARMBuildAttrs::ABI_FP_number_model)+ ", 1");
    else
      OutStreamer.EmitRawText("\t.eabi_attribute " +
                              Twine(ARMBuildAttrs::ABI_FP_number_model)+ ", 3");

    // 8-bytes alignment stuff.
    OutStreamer.EmitRawText("\t.eabi_attribute " +
                            Twine(ARMBuildAttrs::ABI_align8_needed) + ", 1");
    OutStreamer.EmitRawText("\t.eabi_attribute " +
                            Twine(ARMBuildAttrs::ABI_align8_preserved) + ", 1");

    // Hard float.  Use both S and D registers and conform to AAPCS-VFP.
    if (Subtarget->isAAPCS_ABI() && FloatABIType == FloatABI::Hard) {
      OutStreamer.EmitRawText("\t.eabi_attribute " +
                              Twine(ARMBuildAttrs::ABI_HardFP_use) + ", 3");
      OutStreamer.EmitRawText("\t.eabi_attribute " +
                              Twine(ARMBuildAttrs::ABI_VFP_args) + ", 1");
    }
    // FIXME: Should we signal R9 usage?
  }
}


void ARMAsmPrinter::EmitEndOfAsmFile(Module &M) {
  if (Subtarget->isTargetDarwin()) {
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
          OutStreamer.EmitIntValue(0, 4/*size*/, 0/*addrspace*/);
        else
          // Internal to current translation unit.
          //
          // When we place the LSDA into the TEXT section, the type info pointers
          // need to be indirect and pc-rel. We accomplish this by using NLPs.
          // However, sometimes the types are local to the file. So we need to
          // fill in the value for the NLP in those cases.
          OutStreamer.EmitValue(MCSymbolRefExpr::Create(MCSym.getPointer(),
                                                        OutContext),
                                4/*size*/, 0/*addrspace*/);
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
                              4/*size*/, 0/*addrspace*/);
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

void ARMAsmPrinter::printInstructionThroughMCStreamer(const MachineInstr *MI) {
  ARMMCInstLower MCInstLowering(OutContext, *Mang, *this);
  switch (MI->getOpcode()) {
  case ARM::t2MOVi32imm:
    assert(0 && "Should be lowered by thumb2it pass");
  default: break;
  case ARM::PICADD: { // FIXME: Remove asm string from td file.
    // This is a pseudo op for a label + instruction sequence, which looks like:
    // LPC0:
    //     add r0, pc, r0
    // This adds the address of LPC0 to r0.
    
    // Emit the label.
    // FIXME: MOVE TO SHARED PLACE.
    unsigned Id = (unsigned)MI->getOperand(2).getImm();
    const char *Prefix = MAI->getPrivateGlobalPrefix();
    MCSymbol *Label =OutContext.GetOrCreateSymbol(Twine(Prefix)
                         + "PC" + Twine(getFunctionNumber()) + "_" + Twine(Id));
    OutStreamer.EmitLabel(Label);
    
    
    // Form and emit tha dd.
    MCInst AddInst;
    AddInst.setOpcode(ARM::ADDrr);
    AddInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    AddInst.addOperand(MCOperand::CreateReg(ARM::PC));
    AddInst.addOperand(MCOperand::CreateReg(MI->getOperand(1).getReg()));
    OutStreamer.EmitInstruction(AddInst);
    return;
  }
  case ARM::CONSTPOOL_ENTRY: { // FIXME: Remove asm string from td file.
    /// CONSTPOOL_ENTRY - This instruction represents a floating constant pool
    /// in the function.  The first operand is the ID# for this instruction, the
    /// second is the index into the MachineConstantPool that this is, the third
    /// is the size in bytes of this constant pool entry.
    unsigned LabelId = (unsigned)MI->getOperand(0).getImm();
    unsigned CPIdx   = (unsigned)MI->getOperand(1).getIndex();

    EmitAlignment(2);
    OutStreamer.EmitLabel(GetCPISymbol(LabelId));

    const MachineConstantPoolEntry &MCPE = MCP->getConstants()[CPIdx];
    if (MCPE.isMachineConstantPoolEntry())
      EmitMachineConstantPoolValue(MCPE.Val.MachineCPVal);
    else
      EmitGlobalConstant(MCPE.Val.ConstVal);
    
    return;
  }
  case ARM::MOVi2pieces: { // FIXME: Remove asmstring from td file.
    // This is a hack that lowers as a two instruction sequence.
    unsigned DstReg = MI->getOperand(0).getReg();
    unsigned ImmVal = (unsigned)MI->getOperand(1).getImm();

    unsigned SOImmValV1 = ARM_AM::getSOImmTwoPartFirst(ImmVal);
    unsigned SOImmValV2 = ARM_AM::getSOImmTwoPartSecond(ImmVal);
    
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::MOVi);
      TmpInst.addOperand(MCOperand::CreateReg(DstReg));
      TmpInst.addOperand(MCOperand::CreateImm(SOImmValV1));
      
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(MI->getOperand(2).getImm()));
      TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(3).getReg()));

      TmpInst.addOperand(MCOperand::CreateReg(0));          // cc_out
      OutStreamer.EmitInstruction(TmpInst);
    }

    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::ORRri);
      TmpInst.addOperand(MCOperand::CreateReg(DstReg));     // dstreg
      TmpInst.addOperand(MCOperand::CreateReg(DstReg));     // inreg
      TmpInst.addOperand(MCOperand::CreateImm(SOImmValV2)); // so_imm
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(MI->getOperand(2).getImm()));
      TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(3).getReg()));
      
      TmpInst.addOperand(MCOperand::CreateReg(0));          // cc_out
      OutStreamer.EmitInstruction(TmpInst);
    }
    return; 
  }
  case ARM::MOVi32imm: { // FIXME: Remove asmstring from td file.
    // This is a hack that lowers as a two instruction sequence.
    unsigned DstReg = MI->getOperand(0).getReg();
    const MachineOperand &MO = MI->getOperand(1);
    MCOperand V1, V2;
    if (MO.isImm()) {
      unsigned ImmVal = (unsigned)MI->getOperand(1).getImm();
      V1 = MCOperand::CreateImm(ImmVal & 65535);
      V2 = MCOperand::CreateImm(ImmVal >> 16);
    } else if (MO.isGlobal()) {
      MCSymbol *Symbol = MCInstLowering.GetGlobalAddressSymbol(MO);
      const MCSymbolRefExpr *SymRef1 =
        MCSymbolRefExpr::Create(Symbol,
                                MCSymbolRefExpr::VK_ARM_LO16, OutContext);
      const MCSymbolRefExpr *SymRef2 =
        MCSymbolRefExpr::Create(Symbol,
                                MCSymbolRefExpr::VK_ARM_HI16, OutContext);
      V1 = MCOperand::CreateExpr(SymRef1);
      V2 = MCOperand::CreateExpr(SymRef2);
    } else {
      MI->dump();
      llvm_unreachable("cannot handle this operand");
    }

    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::MOVi16);
      TmpInst.addOperand(MCOperand::CreateReg(DstReg));         // dstreg
      TmpInst.addOperand(V1); // lower16(imm)
      
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(MI->getOperand(2).getImm()));
      TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(3).getReg()));
      
      OutStreamer.EmitInstruction(TmpInst);
    }
    
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::MOVTi16);
      TmpInst.addOperand(MCOperand::CreateReg(DstReg));         // dstreg
      TmpInst.addOperand(MCOperand::CreateReg(DstReg));         // srcreg
      TmpInst.addOperand(V2);   // upper16(imm)
      
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(MI->getOperand(2).getImm()));
      TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(3).getReg()));
      
      OutStreamer.EmitInstruction(TmpInst);
    }
    
    return;
  }
  }
      
  MCInst TmpInst;
  MCInstLowering.Lower(MI, TmpInst);
  OutStreamer.EmitInstruction(TmpInst);
}

//===----------------------------------------------------------------------===//
// Target Registry Stuff
//===----------------------------------------------------------------------===//

static MCInstPrinter *createARMMCInstPrinter(const Target &T,
                                             unsigned SyntaxVariant,
                                             const MCAsmInfo &MAI) {
  if (SyntaxVariant == 0)
    return new ARMInstPrinter(MAI, false);
  return 0;
}

// Force static initialization.
extern "C" void LLVMInitializeARMAsmPrinter() {
  RegisterAsmPrinter<ARMAsmPrinter> X(TheARMTarget);
  RegisterAsmPrinter<ARMAsmPrinter> Y(TheThumbTarget);

  TargetRegistry::RegisterMCInstPrinter(TheARMTarget, createARMMCInstPrinter);
  TargetRegistry::RegisterMCInstPrinter(TheThumbTarget, createARMMCInstPrinter);
}

