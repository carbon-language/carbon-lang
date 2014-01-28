//===-- SparcAsmPrinter.cpp - Sparc LLVM assembly writer ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format SPARC assembly language.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "Sparc.h"
#include "InstPrinter/SparcInstPrinter.h"
#include "MCTargetDesc/SparcBaseInfo.h"
#include "MCTargetDesc/SparcMCExpr.h"
#include "SparcInstrInfo.h"
#include "SparcTargetMachine.h"
#include "SparcTargetStreamer.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
  class SparcAsmPrinter : public AsmPrinter {
    SparcTargetStreamer &getTargetStreamer() {
      return static_cast<SparcTargetStreamer &>(
          *OutStreamer.getTargetStreamer());
    }
  public:
    explicit SparcAsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
      : AsmPrinter(TM, Streamer) {}

    virtual const char *getPassName() const {
      return "Sparc Assembly Printer";
    }

    void printOperand(const MachineInstr *MI, int opNum, raw_ostream &OS);
    void printMemOperand(const MachineInstr *MI, int opNum, raw_ostream &OS,
                         const char *Modifier = 0);
    void printCCOperand(const MachineInstr *MI, int opNum, raw_ostream &OS);

    virtual void EmitFunctionBodyStart();
    virtual void EmitInstruction(const MachineInstr *MI);
    virtual void EmitEndOfAsmFile(Module &M);

    static const char *getRegisterName(unsigned RegNo) {
      return SparcInstPrinter::getRegisterName(RegNo);
    }

    bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                         unsigned AsmVariant, const char *ExtraCode,
                         raw_ostream &O);
    bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                               unsigned AsmVariant, const char *ExtraCode,
                               raw_ostream &O);

    void LowerGETPCXAndEmitMCInsts(const MachineInstr *MI,
                                   const MCSubtargetInfo &STI);

  };
} // end of anonymous namespace

static MCOperand createSparcMCOperand(SparcMCExpr::VariantKind Kind,
                                      MCSymbol *Sym, MCContext &OutContext) {
  const MCSymbolRefExpr *MCSym = MCSymbolRefExpr::Create(Sym,
                                                         OutContext);
  const SparcMCExpr *expr = SparcMCExpr::Create(Kind, MCSym, OutContext);
  return MCOperand::CreateExpr(expr);

}
static MCOperand createPCXCallOP(MCSymbol *Label,
                                 MCContext &OutContext) {
  return createSparcMCOperand(SparcMCExpr::VK_Sparc_None, Label, OutContext);
}

static MCOperand createPCXRelExprOp(SparcMCExpr::VariantKind Kind,
                                    MCSymbol *GOTLabel, MCSymbol *StartLabel,
                                    MCSymbol *CurLabel,
                                    MCContext &OutContext)
{
  const MCSymbolRefExpr *GOT = MCSymbolRefExpr::Create(GOTLabel, OutContext);
  const MCSymbolRefExpr *Start = MCSymbolRefExpr::Create(StartLabel,
                                                         OutContext);
  const MCSymbolRefExpr *Cur = MCSymbolRefExpr::Create(CurLabel,
                                                       OutContext);

  const MCBinaryExpr *Sub = MCBinaryExpr::CreateSub(Cur, Start, OutContext);
  const MCBinaryExpr *Add = MCBinaryExpr::CreateAdd(GOT, Sub, OutContext);
  const SparcMCExpr *expr = SparcMCExpr::Create(Kind,
                                                Add, OutContext);
  return MCOperand::CreateExpr(expr);
}

static void EmitCall(MCStreamer &OutStreamer,
                     MCOperand &Callee,
                     const MCSubtargetInfo &STI)
{
  MCInst CallInst;
  CallInst.setOpcode(SP::CALL);
  CallInst.addOperand(Callee);
  OutStreamer.EmitInstruction(CallInst, STI);
}

static void EmitSETHI(MCStreamer &OutStreamer,
                      MCOperand &Imm, MCOperand &RD,
                      const MCSubtargetInfo &STI)
{
  MCInst SETHIInst;
  SETHIInst.setOpcode(SP::SETHIi);
  SETHIInst.addOperand(RD);
  SETHIInst.addOperand(Imm);
  OutStreamer.EmitInstruction(SETHIInst, STI);
}

static void EmitBinary(MCStreamer &OutStreamer, unsigned Opcode,
                       MCOperand &RS1, MCOperand &Src2, MCOperand &RD,
                       const MCSubtargetInfo &STI)
{
  MCInst Inst;
  Inst.setOpcode(Opcode);
  Inst.addOperand(RD);
  Inst.addOperand(RS1);
  Inst.addOperand(Src2);
  OutStreamer.EmitInstruction(Inst, STI);
}

static void EmitOR(MCStreamer &OutStreamer,
                   MCOperand &RS1, MCOperand &Imm, MCOperand &RD,
                   const MCSubtargetInfo &STI) {
  EmitBinary(OutStreamer, SP::ORri, RS1, Imm, RD, STI);
}

static void EmitADD(MCStreamer &OutStreamer,
                    MCOperand &RS1, MCOperand &RS2, MCOperand &RD,
                    const MCSubtargetInfo &STI) {
  EmitBinary(OutStreamer, SP::ADDrr, RS1, RS2, RD, STI);
}

static void EmitSHL(MCStreamer &OutStreamer,
                    MCOperand &RS1, MCOperand &Imm, MCOperand &RD,
                    const MCSubtargetInfo &STI) {
  EmitBinary(OutStreamer, SP::SLLri, RS1, Imm, RD, STI);
}


static void EmitHiLo(MCStreamer &OutStreamer,  MCSymbol *GOTSym,
                     SparcMCExpr::VariantKind HiKind,
                     SparcMCExpr::VariantKind LoKind,
                     MCOperand &RD,
                     MCContext &OutContext,
                     const MCSubtargetInfo &STI) {

  MCOperand hi = createSparcMCOperand(HiKind, GOTSym, OutContext);
  MCOperand lo = createSparcMCOperand(LoKind, GOTSym, OutContext);
  EmitSETHI(OutStreamer, hi, RD, STI);
  EmitOR(OutStreamer, RD, lo, RD, STI);
}

void SparcAsmPrinter::LowerGETPCXAndEmitMCInsts(const MachineInstr *MI,
                                                const MCSubtargetInfo &STI)
{
  MCSymbol *GOTLabel   =
    OutContext.GetOrCreateSymbol(Twine("_GLOBAL_OFFSET_TABLE_"));

  const MachineOperand &MO = MI->getOperand(0);
  assert(MO.getReg() != SP::O7 &&
         "%o7 is assigned as destination for getpcx!");

  MCOperand MCRegOP = MCOperand::CreateReg(MO.getReg());


  if (TM.getRelocationModel() != Reloc::PIC_) {
    // Just load the address of GOT to MCRegOP.
    switch(TM.getCodeModel()) {
    default:
      llvm_unreachable("Unsupported absolute code model");
    case CodeModel::Small:
      EmitHiLo(OutStreamer, GOTLabel,
               SparcMCExpr::VK_Sparc_HI, SparcMCExpr::VK_Sparc_LO,
               MCRegOP, OutContext, STI);
      break;
    case CodeModel::Medium: {
      EmitHiLo(OutStreamer, GOTLabel,
               SparcMCExpr::VK_Sparc_H44, SparcMCExpr::VK_Sparc_M44,
               MCRegOP, OutContext, STI);
      MCOperand imm = MCOperand::CreateExpr(MCConstantExpr::Create(12,
                                                                   OutContext));
      EmitSHL(OutStreamer, MCRegOP, imm, MCRegOP, STI);
      MCOperand lo = createSparcMCOperand(SparcMCExpr::VK_Sparc_L44,
                                          GOTLabel, OutContext);
      EmitOR(OutStreamer, MCRegOP, lo, MCRegOP, STI);
      break;
    }
    case CodeModel::Large: {
      EmitHiLo(OutStreamer, GOTLabel,
               SparcMCExpr::VK_Sparc_HH, SparcMCExpr::VK_Sparc_HM,
               MCRegOP, OutContext, STI);
      MCOperand imm = MCOperand::CreateExpr(MCConstantExpr::Create(32,
                                                                   OutContext));
      EmitSHL(OutStreamer, MCRegOP, imm, MCRegOP, STI);
      // Use register %o7 to load the lower 32 bits.
      MCOperand RegO7 = MCOperand::CreateReg(SP::O7);
      EmitHiLo(OutStreamer, GOTLabel,
               SparcMCExpr::VK_Sparc_HI, SparcMCExpr::VK_Sparc_LO,
               RegO7, OutContext, STI);
      EmitADD(OutStreamer, MCRegOP, RegO7, MCRegOP, STI);
    }
    }
    return;
  }

  MCSymbol *StartLabel = OutContext.CreateTempSymbol();
  MCSymbol *EndLabel   = OutContext.CreateTempSymbol();
  MCSymbol *SethiLabel = OutContext.CreateTempSymbol();

  MCOperand RegO7   = MCOperand::CreateReg(SP::O7);

  // <StartLabel>:
  //   call <EndLabel>
  // <SethiLabel>:
  //     sethi %hi(_GLOBAL_OFFSET_TABLE_+(<SethiLabel>-<StartLabel>)), <MO>
  // <EndLabel>:
  //   or  <MO>, %lo(_GLOBAL_OFFSET_TABLE_+(<EndLabel>-<StartLabel>))), <MO>
  //   add <MO>, %o7, <MO>

  OutStreamer.EmitLabel(StartLabel);
  MCOperand Callee =  createPCXCallOP(EndLabel, OutContext);
  EmitCall(OutStreamer, Callee, STI);
  OutStreamer.EmitLabel(SethiLabel);
  MCOperand hiImm = createPCXRelExprOp(SparcMCExpr::VK_Sparc_HI,
                                       GOTLabel, StartLabel, SethiLabel,
                                       OutContext);
  EmitSETHI(OutStreamer, hiImm, MCRegOP, STI);
  OutStreamer.EmitLabel(EndLabel);
  MCOperand loImm = createPCXRelExprOp(SparcMCExpr::VK_Sparc_LO,
                                       GOTLabel, StartLabel, EndLabel,
                                       OutContext);
  EmitOR(OutStreamer, MCRegOP, loImm, MCRegOP, STI);
  EmitADD(OutStreamer, MCRegOP, RegO7, MCRegOP, STI);
}

void SparcAsmPrinter::EmitInstruction(const MachineInstr *MI)
{

  switch (MI->getOpcode()) {
  default: break;
  case TargetOpcode::DBG_VALUE:
    // FIXME: Debug Value.
    return;
  case SP::GETPCX:
    LowerGETPCXAndEmitMCInsts(MI, getSubtargetInfo());
    return;
  }
  MachineBasicBlock::const_instr_iterator I = MI;
  MachineBasicBlock::const_instr_iterator E = MI->getParent()->instr_end();
  do {
    MCInst TmpInst;
    LowerSparcMachineInstrToMCInst(I, TmpInst, *this);
    EmitToStreamer(OutStreamer, TmpInst);
  } while ((++I != E) && I->isInsideBundle()); // Delay slot check.
}

void SparcAsmPrinter::EmitFunctionBodyStart() {
  if (!TM.getSubtarget<SparcSubtarget>().is64Bit())
    return;

  const MachineRegisterInfo &MRI = MF->getRegInfo();
  const unsigned globalRegs[] = { SP::G2, SP::G3, SP::G6, SP::G7, 0 };
  for (unsigned i = 0; globalRegs[i] != 0; ++i) {
    unsigned reg = globalRegs[i];
    if (MRI.use_empty(reg))
      continue;

    if  (reg == SP::G6 || reg == SP::G7)
      getTargetStreamer().emitSparcRegisterIgnore(reg);
    else
      getTargetStreamer().emitSparcRegisterScratch(reg);
  }
}

void SparcAsmPrinter::printOperand(const MachineInstr *MI, int opNum,
                                   raw_ostream &O) {
  const DataLayout *DL = TM.getDataLayout();
  const MachineOperand &MO = MI->getOperand (opNum);
  unsigned TF = MO.getTargetFlags();
#ifndef NDEBUG
  // Verify the target flags.
  if (MO.isGlobal() || MO.isSymbol() || MO.isCPI()) {
    if (MI->getOpcode() == SP::CALL)
      assert(TF == SPII::MO_NO_FLAG &&
             "Cannot handle target flags on call address");
    else if (MI->getOpcode() == SP::SETHIi || MI->getOpcode() == SP::SETHIXi)
      assert((TF == SPII::MO_HI || TF == SPII::MO_H44 || TF == SPII::MO_HH
              || TF == SPII::MO_TLS_GD_HI22
              || TF == SPII::MO_TLS_LDM_HI22
              || TF == SPII::MO_TLS_LDO_HIX22
              || TF == SPII::MO_TLS_IE_HI22
              || TF == SPII::MO_TLS_LE_HIX22) &&
             "Invalid target flags for address operand on sethi");
    else if (MI->getOpcode() == SP::TLS_CALL)
      assert((TF == SPII::MO_NO_FLAG
              || TF == SPII::MO_TLS_GD_CALL
              || TF == SPII::MO_TLS_LDM_CALL) &&
             "Cannot handle target flags on tls call address");
    else if (MI->getOpcode() == SP::TLS_ADDrr)
      assert((TF == SPII::MO_TLS_GD_ADD || TF == SPII::MO_TLS_LDM_ADD
              || TF == SPII::MO_TLS_LDO_ADD || TF == SPII::MO_TLS_IE_ADD) &&
             "Cannot handle target flags on add for TLS");
    else if (MI->getOpcode() == SP::TLS_LDrr)
      assert(TF == SPII::MO_TLS_IE_LD &&
             "Cannot handle target flags on ld for TLS");
    else if (MI->getOpcode() == SP::TLS_LDXrr)
      assert(TF == SPII::MO_TLS_IE_LDX &&
             "Cannot handle target flags on ldx for TLS");
    else if (MI->getOpcode() == SP::XORri || MI->getOpcode() == SP::XORXri)
      assert((TF == SPII::MO_TLS_LDO_LOX10 || TF == SPII::MO_TLS_LE_LOX10) &&
             "Cannot handle target flags on xor for TLS");
    else
      assert((TF == SPII::MO_LO || TF == SPII::MO_M44 || TF == SPII::MO_L44
              || TF == SPII::MO_HM
              || TF == SPII::MO_TLS_GD_LO10
              || TF == SPII::MO_TLS_LDM_LO10
              || TF == SPII::MO_TLS_IE_LO10 ) &&
             "Invalid target flags for small address operand");
  }
#endif

  bool CloseParen = true;
  switch (TF) {
  default:
      llvm_unreachable("Unknown target flags on operand");
  case SPII::MO_NO_FLAG:
    CloseParen = false;
    break;
  case SPII::MO_LO:  O << "%lo(";  break;
  case SPII::MO_HI:  O << "%hi(";  break;
  case SPII::MO_H44: O << "%h44("; break;
  case SPII::MO_M44: O << "%m44("; break;
  case SPII::MO_L44: O << "%l44("; break;
  case SPII::MO_HH:  O << "%hh(";  break;
  case SPII::MO_HM:  O << "%hm(";  break;
  case SPII::MO_TLS_GD_HI22:   O << "%tgd_hi22(";   break;
  case SPII::MO_TLS_GD_LO10:   O << "%tgd_lo10(";   break;
  case SPII::MO_TLS_GD_ADD:    O << "%tgd_add(";    break;
  case SPII::MO_TLS_GD_CALL:   O << "%tgd_call(";   break;
  case SPII::MO_TLS_LDM_HI22:  O << "%tldm_hi22(";  break;
  case SPII::MO_TLS_LDM_LO10:  O << "%tldm_lo10(";  break;
  case SPII::MO_TLS_LDM_ADD:   O << "%tldm_add(";   break;
  case SPII::MO_TLS_LDM_CALL:  O << "%tldm_call(";  break;
  case SPII::MO_TLS_LDO_HIX22: O << "%tldo_hix22("; break;
  case SPII::MO_TLS_LDO_LOX10: O << "%tldo_lox10("; break;
  case SPII::MO_TLS_LDO_ADD:   O << "%tldo_add(";   break;
  case SPII::MO_TLS_IE_HI22:   O << "%tie_hi22(";   break;
  case SPII::MO_TLS_IE_LO10:   O << "%tie_lo10(";   break;
  case SPII::MO_TLS_IE_LD:     O << "%tie_ld(";     break;
  case SPII::MO_TLS_IE_LDX:    O << "%tie_ldx(";    break;
  case SPII::MO_TLS_IE_ADD:    O << "%tie_add(";    break;
  case SPII::MO_TLS_LE_HIX22:  O << "%tle_hix22(";  break;
  case SPII::MO_TLS_LE_LOX10:  O << "%tle_lox10(";   break;
  }

  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    O << "%" << StringRef(getRegisterName(MO.getReg())).lower();
    break;

  case MachineOperand::MO_Immediate:
    O << (int)MO.getImm();
    break;
  case MachineOperand::MO_MachineBasicBlock:
    O << *MO.getMBB()->getSymbol();
    return;
  case MachineOperand::MO_GlobalAddress:
    O << *getSymbol(MO.getGlobal());
    break;
  case MachineOperand::MO_BlockAddress:
    O <<  GetBlockAddressSymbol(MO.getBlockAddress())->getName();
    break;
  case MachineOperand::MO_ExternalSymbol:
    O << MO.getSymbolName();
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    O << DL->getPrivateGlobalPrefix() << "CPI" << getFunctionNumber() << "_"
      << MO.getIndex();
    break;
  default:
    llvm_unreachable("<unknown operand type>");
  }
  if (CloseParen) O << ")";
}

void SparcAsmPrinter::printMemOperand(const MachineInstr *MI, int opNum,
                                      raw_ostream &O, const char *Modifier) {
  printOperand(MI, opNum, O);

  // If this is an ADD operand, emit it like normal operands.
  if (Modifier && !strcmp(Modifier, "arith")) {
    O << ", ";
    printOperand(MI, opNum+1, O);
    return;
  }

  if (MI->getOperand(opNum+1).isReg() &&
      MI->getOperand(opNum+1).getReg() == SP::G0)
    return;   // don't print "+%g0"
  if (MI->getOperand(opNum+1).isImm() &&
      MI->getOperand(opNum+1).getImm() == 0)
    return;   // don't print "+0"

  O << "+";
  printOperand(MI, opNum+1, O);
}

/// PrintAsmOperand - Print out an operand for an inline asm expression.
///
bool SparcAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                      unsigned AsmVariant,
                                      const char *ExtraCode,
                                      raw_ostream &O) {
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default:
      // See if this is a generic print operand
      return AsmPrinter::PrintAsmOperand(MI, OpNo, AsmVariant, ExtraCode, O);
    case 'r':
     break;
    }
  }

  printOperand(MI, OpNo, O);

  return false;
}

bool SparcAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                            unsigned OpNo, unsigned AsmVariant,
                                            const char *ExtraCode,
                                            raw_ostream &O) {
  if (ExtraCode && ExtraCode[0])
    return true;  // Unknown modifier

  O << '[';
  printMemOperand(MI, OpNo, O);
  O << ']';

  return false;
}

void SparcAsmPrinter::EmitEndOfAsmFile(Module &M) {
  const TargetLoweringObjectFileELF &TLOFELF =
    static_cast<const TargetLoweringObjectFileELF &>(getObjFileLowering());
  MachineModuleInfoELF &MMIELF = MMI->getObjFileInfo<MachineModuleInfoELF>();

  // Generate stubs for global variables.
  MachineModuleInfoELF::SymbolListTy Stubs = MMIELF.GetGVStubList();
  if (!Stubs.empty()) {
    OutStreamer.SwitchSection(TLOFELF.getDataSection());
    unsigned PtrSize = TM.getDataLayout()->getPointerSize(0);
    for (unsigned i = 0, e = Stubs.size(); i != e; ++i) {
      OutStreamer.EmitLabel(Stubs[i].first);
      OutStreamer.EmitSymbolValue(Stubs[i].second.getPointer(), PtrSize);
    }
  }
}

// Force static initialization.
extern "C" void LLVMInitializeSparcAsmPrinter() {
  RegisterAsmPrinter<SparcAsmPrinter> X(TheSparcTarget);
  RegisterAsmPrinter<SparcAsmPrinter> Y(TheSparcV9Target);
}
