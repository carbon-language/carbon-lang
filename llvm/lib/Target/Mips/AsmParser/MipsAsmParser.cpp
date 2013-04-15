//===-- MipsAsmParser.cpp - Parse Mips assembly to MCInst instructions ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "MipsRegisterInfo.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

namespace {
class MipsAssemblerOptions {
public:
  MipsAssemblerOptions():
    aTReg(1), reorder(true), macro(true) {
  }

  unsigned getATRegNum() {return aTReg;}
  bool setATReg(unsigned Reg);

  bool isReorder() {return reorder;}
  void setReorder() {reorder = true;}
  void setNoreorder() {reorder = false;}

  bool isMacro() {return macro;}
  void setMacro() {macro = true;}
  void setNomacro() {macro = false;}

private:
  unsigned aTReg;
  bool reorder;
  bool macro;
};
}

namespace {
class MipsAsmParser : public MCTargetAsmParser {

  enum FpFormatTy {
    FP_FORMAT_NONE = -1,
    FP_FORMAT_S,
    FP_FORMAT_D,
    FP_FORMAT_L,
    FP_FORMAT_W
  } FpFormat;

  MCSubtargetInfo &STI;
  MCAsmParser &Parser;
  MipsAssemblerOptions Options;


#define GET_ASSEMBLER_HEADER
#include "MipsGenAsmMatcher.inc"

  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                               MCStreamer &Out, unsigned &ErrorInfo,
                               bool MatchingInlineAsm);

  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc);

  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc,
                        SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  bool parseMathOperation(StringRef Name, SMLoc NameLoc,
                        SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  bool ParseDirective(AsmToken DirectiveID);

  MipsAsmParser::OperandMatchResultTy
  parseMemOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseCPURegs(SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseCPU64Regs(SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseHWRegs(SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseHW64Regs(SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseCCRRegs(SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  bool searchSymbolAlias(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                         unsigned RegisterClass);

  bool ParseOperand(SmallVectorImpl<MCParsedAsmOperand*> &,
                    StringRef Mnemonic);

  int tryParseRegister(bool is64BitReg);

  bool tryParseRegisterOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                               bool is64BitReg);

  bool needsExpansion(MCInst &Inst);

  void expandInstruction(MCInst &Inst, SMLoc IDLoc,
                         SmallVectorImpl<MCInst> &Instructions);
  void expandLoadImm(MCInst &Inst, SMLoc IDLoc,
                     SmallVectorImpl<MCInst> &Instructions);
  void expandLoadAddressImm(MCInst &Inst, SMLoc IDLoc,
                            SmallVectorImpl<MCInst> &Instructions);
  void expandLoadAddressReg(MCInst &Inst, SMLoc IDLoc,
                            SmallVectorImpl<MCInst> &Instructions);
  void expandMemInst(MCInst &Inst, SMLoc IDLoc,
                     SmallVectorImpl<MCInst> &Instructions,
                     bool isLoad,bool isImmOpnd);
  bool reportParseError(StringRef ErrorMsg);

  bool parseMemOffset(const MCExpr *&Res);
  bool parseRelocOperand(const MCExpr *&Res);

  bool parseDirectiveSet();

  bool parseSetAtDirective();
  bool parseSetNoAtDirective();
  bool parseSetMacroDirective();
  bool parseSetNoMacroDirective();
  bool parseSetReorderDirective();
  bool parseSetNoReorderDirective();

  bool parseSetAssignment();

  bool parseDirectiveWord(unsigned Size, SMLoc L);

  MCSymbolRefExpr::VariantKind getVariantKind(StringRef Symbol);

  bool isMips64() const {
    return (STI.getFeatureBits() & Mips::FeatureMips64) != 0;
  }

  bool isFP64() const {
    return (STI.getFeatureBits() & Mips::FeatureFP64Bit) != 0;
  }

  int matchRegisterName(StringRef Symbol, bool is64BitReg);

  int matchCPURegisterName(StringRef Symbol);

  int matchRegisterByNumber(unsigned RegNum, unsigned RegClass);

  void setFpFormat(FpFormatTy Format) {
    FpFormat = Format;
  }

  void setDefaultFpFormat();

  void setFpFormat(StringRef Format);

  FpFormatTy getFpFormat() {return FpFormat;}

  bool requestsDoubleOperand(StringRef Mnemonic);

  unsigned getReg(int RC,int RegNo);

  int getATReg();

  bool processInstruction(MCInst &Inst, SMLoc IDLoc,
                        SmallVectorImpl<MCInst> &Instructions);
public:
  MipsAsmParser(MCSubtargetInfo &sti, MCAsmParser &parser)
    : MCTargetAsmParser(), STI(sti), Parser(parser) {
    // Initialize the set of available features.
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }

  MCAsmParser &getParser() const { return Parser; }
  MCAsmLexer &getLexer() const { return Parser.getLexer(); }

};
}

namespace {

/// MipsOperand - Instances of this class represent a parsed Mips machine
/// instruction.
class MipsOperand : public MCParsedAsmOperand {

public:
  enum RegisterKind {
    Kind_None,
    Kind_CPURegs,
    Kind_CPU64Regs,
    Kind_HWRegs,
    Kind_HW64Regs,
    Kind_FGR32Regs,
    Kind_FGR64Regs,
    Kind_AFGR64Regs,
    Kind_CCRRegs
  };

private:
  enum KindTy {
    k_CondCode,
    k_CoprocNum,
    k_Immediate,
    k_Memory,
    k_PostIndexRegister,
    k_Register,
    k_Token
  } Kind;

  MipsOperand(KindTy K) : MCParsedAsmOperand(), Kind(K) {}

  struct Token {
    const char *Data;
    unsigned Length;
  };

  struct RegOp {
    unsigned RegNum;
    RegisterKind Kind;
  };

  struct ImmOp {
    const MCExpr *Val;
  };

  struct MemOp {
    unsigned Base;
    const MCExpr *Off;
  };

  union {
    struct Token Tok;
    struct RegOp Reg;
    struct ImmOp Imm;
    struct MemOp Mem;
  };

  SMLoc StartLoc, EndLoc;

public:
  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getReg()));
  }

  void addExpr(MCInst &Inst, const MCExpr *Expr) const{
    // Add as immediate when possible.  Null MCExpr = 0.
    if (Expr == 0)
      Inst.addOperand(MCOperand::CreateImm(0));
    else if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::CreateImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::CreateExpr(Expr));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCExpr *Expr = getImm();
    addExpr(Inst,Expr);
  }

  void addMemOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");

    Inst.addOperand(MCOperand::CreateReg(getMemBase()));

    const MCExpr *Expr = getMemOff();
    addExpr(Inst,Expr);
  }

  bool isReg() const { return Kind == k_Register; }
  bool isImm() const { return Kind == k_Immediate; }
  bool isToken() const { return Kind == k_Token; }
  bool isMem() const { return Kind == k_Memory; }

  StringRef getToken() const {
    assert(Kind == k_Token && "Invalid access!");
    return StringRef(Tok.Data, Tok.Length);
  }

  unsigned getReg() const {
    assert((Kind == k_Register) && "Invalid access!");
    return Reg.RegNum;
  }

  void setRegKind(RegisterKind RegKind) {
    assert((Kind == k_Register) && "Invalid access!");
    Reg.Kind = RegKind;
  }

  const MCExpr *getImm() const {
    assert((Kind == k_Immediate) && "Invalid access!");
    return Imm.Val;
  }

  unsigned getMemBase() const {
    assert((Kind == k_Memory) && "Invalid access!");
    return Mem.Base;
  }

  const MCExpr *getMemOff() const {
    assert((Kind == k_Memory) && "Invalid access!");
    return Mem.Off;
  }

  static MipsOperand *CreateToken(StringRef Str, SMLoc S) {
    MipsOperand *Op = new MipsOperand(k_Token);
    Op->Tok.Data = Str.data();
    Op->Tok.Length = Str.size();
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static MipsOperand *CreateReg(unsigned RegNum, SMLoc S, SMLoc E) {
    MipsOperand *Op = new MipsOperand(k_Register);
    Op->Reg.RegNum = RegNum;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static MipsOperand *CreateImm(const MCExpr *Val, SMLoc S, SMLoc E) {
    MipsOperand *Op = new MipsOperand(k_Immediate);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static MipsOperand *CreateMem(unsigned Base, const MCExpr *Off,
                                 SMLoc S, SMLoc E) {
    MipsOperand *Op = new MipsOperand(k_Memory);
    Op->Mem.Base = Base;
    Op->Mem.Off = Off;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  bool isCPURegsAsm() const {
    return Kind == k_Register && Reg.Kind == Kind_CPURegs;
  }
  void addCPURegsAsmOperands(MCInst &Inst, unsigned N) const {
    Inst.addOperand(MCOperand::CreateReg(Reg.RegNum));
  }

  bool isCPU64RegsAsm() const {
    return Kind == k_Register && Reg.Kind == Kind_CPU64Regs;
  }
  void addCPU64RegsAsmOperands(MCInst &Inst, unsigned N) const {
    Inst.addOperand(MCOperand::CreateReg(Reg.RegNum));
  }

  bool isHWRegsAsm() const {
    assert((Kind == k_Register) && "Invalid access!");
    return Reg.Kind == Kind_HWRegs;
  }
  void addHWRegsAsmOperands(MCInst &Inst, unsigned N) const {
    Inst.addOperand(MCOperand::CreateReg(Reg.RegNum));
  }

  bool isHW64RegsAsm() const {
    assert((Kind == k_Register) && "Invalid access!");
    return Reg.Kind == Kind_HW64Regs;
  }
  void addHW64RegsAsmOperands(MCInst &Inst, unsigned N) const {
    Inst.addOperand(MCOperand::CreateReg(Reg.RegNum));
  }

  void addCCRAsmOperands(MCInst &Inst, unsigned N) const {
    Inst.addOperand(MCOperand::CreateReg(Reg.RegNum));
  }

  bool isCCRAsm() const {
    assert((Kind == k_Register) && "Invalid access!");
    return Reg.Kind == Kind_CCRRegs;
  }

  /// getStartLoc - Get the location of the first token of this operand.
  SMLoc getStartLoc() const { return StartLoc; }
  /// getEndLoc - Get the location of the last token of this operand.
  SMLoc getEndLoc() const { return EndLoc; }

  virtual void print(raw_ostream &OS) const {
    llvm_unreachable("unimplemented!");
  }
};
}

namespace llvm {
extern const MCInstrDesc MipsInsts[];
}
static const MCInstrDesc &getInstDesc(unsigned Opcode) {
  return MipsInsts[Opcode];
}

bool MipsAsmParser::processInstruction(MCInst &Inst, SMLoc IDLoc,
                        SmallVectorImpl<MCInst> &Instructions) {
  const MCInstrDesc &MCID = getInstDesc(Inst.getOpcode());
  Inst.setLoc(IDLoc);
  if (MCID.mayLoad() || MCID.mayStore()) {
    // Check the offset of memory operand, if it is a symbol
    // reference or immediate we may have to expand instructions
    for (unsigned i=0;i<MCID.getNumOperands();i++) {
      const MCOperandInfo &OpInfo = MCID.OpInfo[i];
      if ((OpInfo.OperandType == MCOI::OPERAND_MEMORY) ||
          (OpInfo.OperandType == MCOI::OPERAND_UNKNOWN)) {
        MCOperand &Op = Inst.getOperand(i);
        if (Op.isImm()) {
          int MemOffset = Op.getImm();
          if (MemOffset < -32768 || MemOffset > 32767) {
            // Offset can't exceed 16bit value
            expandMemInst(Inst,IDLoc,Instructions,MCID.mayLoad(),true);
            return false;
          }
        } else if (Op.isExpr()) {
          const MCExpr *Expr = Op.getExpr();
          if (Expr->getKind() == MCExpr::SymbolRef){
            const MCSymbolRefExpr *SR =
                    static_cast<const MCSymbolRefExpr*>(Expr);
            if (SR->getKind() == MCSymbolRefExpr::VK_None) {
              // Expand symbol
              expandMemInst(Inst,IDLoc,Instructions,MCID.mayLoad(),false);
              return false;
            }
          }
        }
      }
    }
  }

  if (needsExpansion(Inst))
    expandInstruction(Inst, IDLoc, Instructions);
  else
    Instructions.push_back(Inst);

  return false;
}

bool MipsAsmParser::needsExpansion(MCInst &Inst) {

  switch(Inst.getOpcode()) {
    case Mips::LoadImm32Reg:
    case Mips::LoadAddr32Imm:
    case Mips::LoadAddr32Reg:
      return true;
    default:
      return false;
  }
}

void MipsAsmParser::expandInstruction(MCInst &Inst, SMLoc IDLoc,
                        SmallVectorImpl<MCInst> &Instructions){
  switch(Inst.getOpcode()) {
    case Mips::LoadImm32Reg:
      return expandLoadImm(Inst, IDLoc, Instructions);
    case Mips::LoadAddr32Imm:
      return expandLoadAddressImm(Inst,IDLoc,Instructions);
    case Mips::LoadAddr32Reg:
      return expandLoadAddressReg(Inst,IDLoc,Instructions);
    }
}

void MipsAsmParser::expandLoadImm(MCInst &Inst, SMLoc IDLoc,
                                  SmallVectorImpl<MCInst> &Instructions){
  MCInst tmpInst;
  const MCOperand &ImmOp = Inst.getOperand(1);
  assert(ImmOp.isImm() && "expected immediate operand kind");
  const MCOperand &RegOp = Inst.getOperand(0);
  assert(RegOp.isReg() && "expected register operand kind");

  int ImmValue = ImmOp.getImm();
  tmpInst.setLoc(IDLoc);
  if ( 0 <= ImmValue && ImmValue <= 65535) {
    // for 0 <= j <= 65535.
    // li d,j => ori d,$zero,j
    tmpInst.setOpcode(Mips::ORi);
    tmpInst.addOperand(MCOperand::CreateReg(RegOp.getReg()));
    tmpInst.addOperand(
              MCOperand::CreateReg(Mips::ZERO));
    tmpInst.addOperand(MCOperand::CreateImm(ImmValue));
    Instructions.push_back(tmpInst);
  } else if ( ImmValue < 0 && ImmValue >= -32768) {
    // for -32768 <= j < 0.
    // li d,j => addiu d,$zero,j
    tmpInst.setOpcode(Mips::ADDiu);
    tmpInst.addOperand(MCOperand::CreateReg(RegOp.getReg()));
    tmpInst.addOperand(
              MCOperand::CreateReg(Mips::ZERO));
    tmpInst.addOperand(MCOperand::CreateImm(ImmValue));
    Instructions.push_back(tmpInst);
  } else {
    // for any other value of j that is representable as a 32-bit integer.
    // li d,j => lui d,hi16(j)
    //           ori d,d,lo16(j)
    tmpInst.setOpcode(Mips::LUi);
    tmpInst.addOperand(MCOperand::CreateReg(RegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateImm((ImmValue & 0xffff0000) >> 16));
    Instructions.push_back(tmpInst);
    tmpInst.clear();
    tmpInst.setOpcode(Mips::ORi);
    tmpInst.addOperand(MCOperand::CreateReg(RegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateReg(RegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateImm(ImmValue & 0xffff));
    tmpInst.setLoc(IDLoc);
    Instructions.push_back(tmpInst);
  }
}

void MipsAsmParser::expandLoadAddressReg(MCInst &Inst, SMLoc IDLoc,
                                         SmallVectorImpl<MCInst> &Instructions){
  MCInst tmpInst;
  const MCOperand &ImmOp = Inst.getOperand(2);
  assert(ImmOp.isImm() && "expected immediate operand kind");
  const MCOperand &SrcRegOp = Inst.getOperand(1);
  assert(SrcRegOp.isReg() && "expected register operand kind");
  const MCOperand &DstRegOp = Inst.getOperand(0);
  assert(DstRegOp.isReg() && "expected register operand kind");
  int ImmValue = ImmOp.getImm();
  if ( -32768 <= ImmValue && ImmValue <= 65535) {
    //for -32768 <= j <= 65535.
    //la d,j(s) => addiu d,s,j
    tmpInst.setOpcode(Mips::ADDiu);
    tmpInst.addOperand(MCOperand::CreateReg(DstRegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateReg(SrcRegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateImm(ImmValue));
    Instructions.push_back(tmpInst);
  } else {
    //for any other value of j that is representable as a 32-bit integer.
    //la d,j(s) => lui d,hi16(j)
    //             ori d,d,lo16(j)
    //             addu d,d,s
    tmpInst.setOpcode(Mips::LUi);
    tmpInst.addOperand(MCOperand::CreateReg(DstRegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateImm((ImmValue & 0xffff0000) >> 16));
    Instructions.push_back(tmpInst);
    tmpInst.clear();
    tmpInst.setOpcode(Mips::ORi);
    tmpInst.addOperand(MCOperand::CreateReg(DstRegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateReg(DstRegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateImm(ImmValue & 0xffff));
    Instructions.push_back(tmpInst);
    tmpInst.clear();
    tmpInst.setOpcode(Mips::ADDu);
    tmpInst.addOperand(MCOperand::CreateReg(DstRegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateReg(DstRegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateReg(SrcRegOp.getReg()));
    Instructions.push_back(tmpInst);
  }
}

void MipsAsmParser::expandLoadAddressImm(MCInst &Inst, SMLoc IDLoc,
                                         SmallVectorImpl<MCInst> &Instructions){
  MCInst tmpInst;
  const MCOperand &ImmOp = Inst.getOperand(1);
  assert(ImmOp.isImm() && "expected immediate operand kind");
  const MCOperand &RegOp = Inst.getOperand(0);
  assert(RegOp.isReg() && "expected register operand kind");
  int ImmValue = ImmOp.getImm();
  if ( -32768 <= ImmValue && ImmValue <= 65535) {
    //for -32768 <= j <= 65535.
    //la d,j => addiu d,$zero,j
    tmpInst.setOpcode(Mips::ADDiu);
    tmpInst.addOperand(MCOperand::CreateReg(RegOp.getReg()));
    tmpInst.addOperand(
              MCOperand::CreateReg(Mips::ZERO));
    tmpInst.addOperand(MCOperand::CreateImm(ImmValue));
    Instructions.push_back(tmpInst);
  } else {
    //for any other value of j that is representable as a 32-bit integer.
    //la d,j => lui d,hi16(j)
    //          ori d,d,lo16(j)
    tmpInst.setOpcode(Mips::LUi);
    tmpInst.addOperand(MCOperand::CreateReg(RegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateImm((ImmValue & 0xffff0000) >> 16));
    Instructions.push_back(tmpInst);
    tmpInst.clear();
    tmpInst.setOpcode(Mips::ORi);
    tmpInst.addOperand(MCOperand::CreateReg(RegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateReg(RegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateImm(ImmValue & 0xffff));
    Instructions.push_back(tmpInst);
  }
}

void MipsAsmParser::expandMemInst(MCInst &Inst, SMLoc IDLoc,
                     SmallVectorImpl<MCInst> &Instructions,
                     bool isLoad,bool isImmOpnd) {
  const MCSymbolRefExpr *SR;
  MCInst TempInst;
  unsigned ImmOffset,HiOffset,LoOffset;
  const MCExpr *ExprOffset;
  unsigned TmpRegNum;
  unsigned AtRegNum = getReg((isMips64()) ? Mips::CPU64RegsRegClassID:
                                            Mips::CPURegsRegClassID,
                                            getATReg());
  // 1st operand is either source or dst register
  assert(Inst.getOperand(0).isReg() && "expected register operand kind");
  unsigned RegOpNum = Inst.getOperand(0).getReg();
  // 2nd operand is base register
  assert(Inst.getOperand(1).isReg() && "expected register operand kind");
  unsigned BaseRegNum = Inst.getOperand(1).getReg();
  // 3rd operand is either immediate or expression
  if (isImmOpnd) {
    assert(Inst.getOperand(2).isImm() && "expected immediate operand kind");
    ImmOffset = Inst.getOperand(2).getImm();
    LoOffset = ImmOffset & 0x0000ffff;
    HiOffset = (ImmOffset & 0xffff0000) >> 16;
    // If msb of LoOffset is 1(negative number) we must increment HiOffset
    if (LoOffset & 0x8000)
      HiOffset++;
  }
  else
    ExprOffset = Inst.getOperand(2).getExpr();
  // All instructions will have the same location
  TempInst.setLoc(IDLoc);
  // 1st instruction in expansion is LUi. For load instruction we can use
  // the dst register as a temporary if base and dst are different,
  // but for stores we must use $at
  TmpRegNum = (isLoad && (BaseRegNum != RegOpNum))?RegOpNum:AtRegNum;
  TempInst.setOpcode(Mips::LUi);
  TempInst.addOperand(MCOperand::CreateReg(TmpRegNum));
  if (isImmOpnd)
    TempInst.addOperand(MCOperand::CreateImm(HiOffset));
  else {
    if (ExprOffset->getKind() == MCExpr::SymbolRef) {
      SR = static_cast<const MCSymbolRefExpr*>(ExprOffset);
      const MCSymbolRefExpr *HiExpr = MCSymbolRefExpr::
                                        Create(SR->getSymbol().getName(),
                                        MCSymbolRefExpr::VK_Mips_ABS_HI,
                                        getContext());
      TempInst.addOperand(MCOperand::CreateExpr(HiExpr));
    }
  }
  // Add the instruction to the list
  Instructions.push_back(TempInst);
  // and prepare TempInst for next instruction
  TempInst.clear();
  // which is add temp register to base
  TempInst.setOpcode(Mips::ADDu);
  TempInst.addOperand(MCOperand::CreateReg(TmpRegNum));
  TempInst.addOperand(MCOperand::CreateReg(TmpRegNum));
  TempInst.addOperand(MCOperand::CreateReg(BaseRegNum));
  Instructions.push_back(TempInst);
  TempInst.clear();
  // and finaly, create original instruction with low part
  // of offset and new base
  TempInst.setOpcode(Inst.getOpcode());
  TempInst.addOperand(MCOperand::CreateReg(RegOpNum));
  TempInst.addOperand(MCOperand::CreateReg(TmpRegNum));
  if (isImmOpnd)
    TempInst.addOperand(MCOperand::CreateImm(LoOffset));
  else {
    if (ExprOffset->getKind() == MCExpr::SymbolRef) {
      const MCSymbolRefExpr *LoExpr = MCSymbolRefExpr::
                                      Create(SR->getSymbol().getName(),
                                      MCSymbolRefExpr::VK_Mips_ABS_LO,
                                      getContext());
      TempInst.addOperand(MCOperand::CreateExpr(LoExpr));
    }
  }
  Instructions.push_back(TempInst);
  TempInst.clear();
}

bool MipsAsmParser::
MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                        SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                        MCStreamer &Out, unsigned &ErrorInfo,
                        bool MatchingInlineAsm) {
  MCInst Inst;
  SmallVector<MCInst, 8> Instructions;
  unsigned MatchResult = MatchInstructionImpl(Operands, Inst, ErrorInfo,
                                              MatchingInlineAsm);

  switch (MatchResult) {
  default: break;
  case Match_Success: {
    if (processInstruction(Inst,IDLoc,Instructions))
      return true;
    for(unsigned i =0; i < Instructions.size(); i++)
      Out.EmitInstruction(Instructions[i]);
    return false;
  }
  case Match_MissingFeature:
    Error(IDLoc, "instruction requires a CPU feature not currently enabled");
    return true;
  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0U) {
      if (ErrorInfo >= Operands.size())
        return Error(IDLoc, "too few operands for instruction");

      ErrorLoc = ((MipsOperand*)Operands[ErrorInfo])->getStartLoc();
      if (ErrorLoc == SMLoc()) ErrorLoc = IDLoc;
    }

    return Error(ErrorLoc, "invalid operand for instruction");
  }
  case Match_MnemonicFail:
    return Error(IDLoc, "invalid instruction");
  }
  return true;
}

int MipsAsmParser::matchCPURegisterName(StringRef Name) {
   int CC;

  if (Name == "at")
    return getATReg();

    CC = StringSwitch<unsigned>(Name)
    .Case("zero", 0)
    .Case("a0",   4)
    .Case("a1",   5)
    .Case("a2",   6)
    .Case("a3",   7)
    .Case("v0",   2)
    .Case("v1",   3)
    .Case("s0",  16)
    .Case("s1",  17)
    .Case("s2",  18)
    .Case("s3",  19)
    .Case("s4",  20)
    .Case("s5",  21)
    .Case("s6",  22)
    .Case("s7",  23)
    .Case("k0",  26)
    .Case("k1",  27)
    .Case("sp",  29)
    .Case("fp",  30)
    .Case("gp",  28)
    .Case("ra",  31)
    .Case("t0",   8)
    .Case("t1",   9)
    .Case("t2",  10)
    .Case("t3",  11)
    .Case("t4",  12)
    .Case("t5",  13)
    .Case("t6",  14)
    .Case("t7",  15)
    .Case("t8",  24)
    .Case("t9",  25)
    .Default(-1);

  // Although SGI documentation just cut out t0-t3 for n32/n64,
  // GNU pushes the values of t0-t3 to override the o32/o64 values for t4-t7
  // We are supporting both cases, so for t0-t3 we'll just push them to t4-t7.
  if (isMips64() && 8 <= CC  && CC <= 11)
    CC += 4;

  if (CC == -1 && isMips64())
    CC = StringSwitch<unsigned>(Name)
      .Case("a4",   8)
      .Case("a5",   9)
      .Case("a6",  10)
      .Case("a7",  11)
      .Case("kt0", 26)
      .Case("kt1", 27)
      .Case("s8",  30)
      .Default(-1);

  return CC;
}
int MipsAsmParser::matchRegisterName(StringRef Name, bool is64BitReg) {

  if (Name.equals("fcc0"))
    return Mips::FCC0;

  int CC;
  CC = matchCPURegisterName(Name);
  if (CC != -1)
    return matchRegisterByNumber(CC,is64BitReg?Mips::CPU64RegsRegClassID:
                               Mips::CPURegsRegClassID);

  if (Name[0] == 'f') {
    StringRef NumString = Name.substr(1);
    unsigned IntVal;
    if( NumString.getAsInteger(10, IntVal))
      return -1; // not integer
    if (IntVal > 31)
      return -1;

    FpFormatTy Format = getFpFormat();

    if (Format == FP_FORMAT_S || Format == FP_FORMAT_W)
      return getReg(Mips::FGR32RegClassID, IntVal);
    if (Format == FP_FORMAT_D) {
      if(isFP64()) {
        return getReg(Mips::FGR64RegClassID, IntVal);
      }
      // only even numbers available as register pairs
      if (( IntVal > 31) || (IntVal%2 !=  0))
        return -1;
      return getReg(Mips::AFGR64RegClassID, IntVal/2);
    }
  }

  return -1;
}
void MipsAsmParser::setDefaultFpFormat() {

  if (isMips64() || isFP64())
    FpFormat = FP_FORMAT_D;
  else
    FpFormat = FP_FORMAT_S;
}

bool MipsAsmParser::requestsDoubleOperand(StringRef Mnemonic){

  bool IsDouble = StringSwitch<bool>(Mnemonic.lower())
    .Case("ldxc1", true)
    .Case("ldc1",  true)
    .Case("sdxc1", true)
    .Case("sdc1",  true)
    .Default(false);

  return IsDouble;
}
void MipsAsmParser::setFpFormat(StringRef Format) {

  FpFormat = StringSwitch<FpFormatTy>(Format.lower())
    .Case(".s",  FP_FORMAT_S)
    .Case(".d",  FP_FORMAT_D)
    .Case(".l",  FP_FORMAT_L)
    .Case(".w",  FP_FORMAT_W)
    .Default(FP_FORMAT_NONE);
}

bool MipsAssemblerOptions::setATReg(unsigned Reg) {
  if (Reg > 31)
    return false;

  aTReg = Reg;
  return true;
}

int MipsAsmParser::getATReg() {
  return Options.getATRegNum();
}

unsigned MipsAsmParser::getReg(int RC,int RegNo) {
  return *(getContext().getRegisterInfo().getRegClass(RC).begin() + RegNo);
}

int MipsAsmParser::matchRegisterByNumber(unsigned RegNum, unsigned RegClass) {

  if (RegNum > 31)
    return -1;

  return getReg(RegClass, RegNum);
}

int MipsAsmParser::tryParseRegister(bool is64BitReg) {
  const AsmToken &Tok = Parser.getTok();
  int RegNum = -1;

  if (Tok.is(AsmToken::Identifier)) {
    std::string lowerCase = Tok.getString().lower();
    RegNum = matchRegisterName(lowerCase, is64BitReg);
  } else if (Tok.is(AsmToken::Integer))
    RegNum = matchRegisterByNumber(static_cast<unsigned>(Tok.getIntVal()),
                                   is64BitReg ? Mips::CPU64RegsRegClassID
                                              : Mips::CPURegsRegClassID);
  return RegNum;
}

bool MipsAsmParser::
  tryParseRegisterOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                          bool is64BitReg){

  SMLoc S = Parser.getTok().getLoc();
  int RegNo = -1;

  RegNo = tryParseRegister(is64BitReg);
  if (RegNo == -1)
    return true;

  Operands.push_back(MipsOperand::CreateReg(RegNo, S,
      Parser.getTok().getLoc()));
  Parser.Lex(); // Eat register token.
  return false;
}

bool MipsAsmParser::ParseOperand(SmallVectorImpl<MCParsedAsmOperand*>&Operands,
                                 StringRef Mnemonic) {
  // Check if the current operand has a custom associated parser, if so, try to
  // custom parse the operand, or fallback to the general approach.
  OperandMatchResultTy ResTy = MatchOperandParserImpl(Operands, Mnemonic);
  if (ResTy == MatchOperand_Success)
    return false;
  // If there wasn't a custom match, try the generic matcher below. Otherwise,
  // there was a match, but an error occurred, in which case, just return that
  // the operand parsing failed.
  if (ResTy == MatchOperand_ParseFail)
    return true;

  switch (getLexer().getKind()) {
  default:
    Error(Parser.getTok().getLoc(), "unexpected token in operand");
    return true;
  case AsmToken::Dollar: {
    // parse register
    SMLoc S = Parser.getTok().getLoc();
    Parser.Lex(); // Eat dollar token.
    // parse register operand
    if (!tryParseRegisterOperand(Operands, isMips64())) {
      if (getLexer().is(AsmToken::LParen)) {
        // check if it is indexed addressing operand
        Operands.push_back(MipsOperand::CreateToken("(", S));
        Parser.Lex(); // eat parenthesis
        if (getLexer().isNot(AsmToken::Dollar))
          return true;

        Parser.Lex(); // eat dollar
        if (tryParseRegisterOperand(Operands, isMips64()))
          return true;

        if (!getLexer().is(AsmToken::RParen))
          return true;

        S = Parser.getTok().getLoc();
        Operands.push_back(MipsOperand::CreateToken(")", S));
        Parser.Lex();
      }
      return false;
    }
    // maybe it is a symbol reference
    StringRef Identifier;
    if (Parser.parseIdentifier(Identifier))
      return true;

    SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);

    MCSymbol *Sym = getContext().GetOrCreateSymbol("$" + Identifier);

    // Otherwise create a symbol ref.
    const MCExpr *Res = MCSymbolRefExpr::Create(Sym, MCSymbolRefExpr::VK_None,
                                                getContext());

    Operands.push_back(MipsOperand::CreateImm(Res, S, E));
    return false;
  }
  case AsmToken::Identifier:
    // Look for the existing symbol, we should check if
    // we need to assigne the propper RegisterKind
   if (searchSymbolAlias(Operands,MipsOperand::Kind_None))
     return false;
    //else drop to expression parsing
  case AsmToken::LParen:
  case AsmToken::Minus:
  case AsmToken::Plus:
  case AsmToken::Integer:
  case AsmToken::String: {
     // quoted label names
    const MCExpr *IdVal;
    SMLoc S = Parser.getTok().getLoc();
    if (getParser().parseExpression(IdVal))
      return true;
    SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
    Operands.push_back(MipsOperand::CreateImm(IdVal, S, E));
    return false;
  }
  case AsmToken::Percent: {
    // it is a symbol reference or constant expression
    const MCExpr *IdVal;
    SMLoc S = Parser.getTok().getLoc(); // start location of the operand
    if (parseRelocOperand(IdVal))
      return true;

    SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);

    Operands.push_back(MipsOperand::CreateImm(IdVal, S, E));
    return false;
  } // case AsmToken::Percent
  } // switch(getLexer().getKind())
  return true;
}

bool MipsAsmParser::parseRelocOperand(const MCExpr *&Res) {

  Parser.Lex(); // eat % token
  const AsmToken &Tok = Parser.getTok(); // get next token, operation
  if (Tok.isNot(AsmToken::Identifier))
    return true;

  std::string Str = Tok.getIdentifier().str();

  Parser.Lex(); // eat identifier
  // now make expression from the rest of the operand
  const MCExpr *IdVal;
  SMLoc EndLoc;

  if (getLexer().getKind() == AsmToken::LParen) {
    while (1) {
      Parser.Lex(); // eat '(' token
      if (getLexer().getKind() == AsmToken::Percent) {
        Parser.Lex(); // eat % token
        const AsmToken &nextTok = Parser.getTok();
        if (nextTok.isNot(AsmToken::Identifier))
          return true;
        Str += "(%";
        Str += nextTok.getIdentifier();
        Parser.Lex(); // eat identifier
        if (getLexer().getKind() != AsmToken::LParen)
          return true;
      } else
        break;
    }
    if (getParser().parseParenExpression(IdVal,EndLoc))
      return true;

    while (getLexer().getKind() == AsmToken::RParen)
      Parser.Lex(); // eat ')' token

  } else
    return true; // parenthesis must follow reloc operand

  // Check the type of the expression
  if (const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(IdVal)) {
    // It's a constant, evaluate lo or hi value
    if (Str == "lo") {
      short Val = MCE->getValue();
      Res = MCConstantExpr::Create(Val, getContext());
    } else if (Str == "hi") {
      int Val = MCE->getValue();
      int LoSign = Val & 0x8000;
      Val = (Val & 0xffff0000) >> 16;
      // Lower part is treated as a signed int, so if it is negative
      // we must add 1 to the hi part to compensate
      if (LoSign)
        Val++;
      Res = MCConstantExpr::Create(Val, getContext());
    }
    return false;
  }

  if (const MCSymbolRefExpr *MSRE = dyn_cast<MCSymbolRefExpr>(IdVal)) {
    // It's a symbol, create symbolic expression from symbol
    StringRef Symbol = MSRE->getSymbol().getName();
    MCSymbolRefExpr::VariantKind VK = getVariantKind(Str);
    Res = MCSymbolRefExpr::Create(Symbol,VK,getContext());
    return false;
  }
  return true;
}

bool MipsAsmParser::ParseRegister(unsigned &RegNo, SMLoc &StartLoc,
                                  SMLoc &EndLoc) {

  StartLoc = Parser.getTok().getLoc();
  RegNo = tryParseRegister(isMips64());
  EndLoc = Parser.getTok().getLoc();
  return (RegNo == (unsigned)-1);
}

bool MipsAsmParser::parseMemOffset(const MCExpr *&Res) {

  SMLoc S;

  switch(getLexer().getKind()) {
  default:
    return true;
  case AsmToken::Identifier:
  case AsmToken::Integer:
  case AsmToken::Minus:
  case AsmToken::Plus:
    return (getParser().parseExpression(Res));
  case AsmToken::Percent:
    return parseRelocOperand(Res);
  case AsmToken::LParen:
    return false;  // it's probably assuming 0
  }
  return true;
}

MipsAsmParser::OperandMatchResultTy MipsAsmParser::parseMemOperand(
               SmallVectorImpl<MCParsedAsmOperand*>&Operands) {

  const MCExpr *IdVal = 0;
  SMLoc S;
  // first operand is the offset
  S = Parser.getTok().getLoc();

  if (parseMemOffset(IdVal))
    return MatchOperand_ParseFail;

  const AsmToken &Tok = Parser.getTok(); // get next token
  if (Tok.isNot(AsmToken::LParen)) {
    MipsOperand *Mnemonic = static_cast<MipsOperand*>(Operands[0]);
    if (Mnemonic->getToken() == "la") {
      SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() -1);
      Operands.push_back(MipsOperand::CreateImm(IdVal, S, E));
      return MatchOperand_Success;
    }
    Error(Parser.getTok().getLoc(), "'(' expected");
    return MatchOperand_ParseFail;
  }

  Parser.Lex(); // Eat '(' token.

  const AsmToken &Tok1 = Parser.getTok(); // get next token
  if (Tok1.is(AsmToken::Dollar)) {
    Parser.Lex(); // Eat '$' token.
    if (tryParseRegisterOperand(Operands, isMips64())) {
      Error(Parser.getTok().getLoc(), "unexpected token in operand");
      return MatchOperand_ParseFail;
    }

  } else {
    Error(Parser.getTok().getLoc(), "unexpected token in operand");
    return MatchOperand_ParseFail;
  }

  const AsmToken &Tok2 = Parser.getTok(); // get next token
  if (Tok2.isNot(AsmToken::RParen)) {
    Error(Parser.getTok().getLoc(), "')' expected");
    return MatchOperand_ParseFail;
  }

  SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);

  Parser.Lex(); // Eat ')' token.

  if (IdVal == 0)
    IdVal = MCConstantExpr::Create(0, getContext());

  // now replace register operand with the mem operand
  MipsOperand* op = static_cast<MipsOperand*>(Operands.back());
  int RegNo = op->getReg();
  // remove register from operands
  Operands.pop_back();
  // and add memory operand
  Operands.push_back(MipsOperand::CreateMem(RegNo, IdVal, S, E));
  delete op;
  return MatchOperand_Success;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseCPU64Regs(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {

  if (!isMips64())
    return MatchOperand_NoMatch;
  if (getLexer().getKind() == AsmToken::Identifier) {
    if (searchSymbolAlias(Operands,MipsOperand::Kind_CPU64Regs))
      return MatchOperand_Success;
    return MatchOperand_NoMatch;
  }
  // if the first token is not '$' we have an error
  if (Parser.getTok().isNot(AsmToken::Dollar))
    return MatchOperand_NoMatch;

  Parser.Lex(); // Eat $
  if(!tryParseRegisterOperand(Operands, true)) {
    // set the proper register kind
    MipsOperand* op = static_cast<MipsOperand*>(Operands.back());
    op->setRegKind(MipsOperand::Kind_CPU64Regs);
    return MatchOperand_Success;
  }
  return MatchOperand_NoMatch;
}

bool MipsAsmParser::
searchSymbolAlias(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                  unsigned RegisterKind) {

  MCSymbol *Sym = getContext().LookupSymbol(Parser.getTok().getIdentifier());
  if (Sym) {
    SMLoc S = Parser.getTok().getLoc();
    const MCExpr *Expr;
    if (Sym->isVariable())
      Expr = Sym->getVariableValue();
    else
      return false;
    if (Expr->getKind() == MCExpr::SymbolRef) {
      const MCSymbolRefExpr *Ref = static_cast<const MCSymbolRefExpr*>(Expr);
      const StringRef DefSymbol = Ref->getSymbol().getName();
      if (DefSymbol.startswith("$")) {
        // Lookup for the register with corresponding name
        int RegNum = matchRegisterName(DefSymbol.substr(1),isMips64());
        if (RegNum > -1) {
          Parser.Lex();
          MipsOperand *op = MipsOperand::CreateReg(RegNum,S,
                                         Parser.getTok().getLoc());
          op->setRegKind((MipsOperand::RegisterKind)RegisterKind);
          Operands.push_back(op);
          return true;
        }
      }
    } else if (Expr->getKind() == MCExpr::Constant) {
      Parser.Lex();
      const MCConstantExpr *Const = static_cast<const MCConstantExpr*>(Expr);
      MipsOperand *op = MipsOperand::CreateImm(Const,S,
                                     Parser.getTok().getLoc());
      Operands.push_back(op);
      return true;
    }
  }
  return false;
}
MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseCPURegs(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {

  if (getLexer().getKind() == AsmToken::Identifier) {
    if (searchSymbolAlias(Operands,MipsOperand::Kind_CPURegs))
      return MatchOperand_Success;
    return MatchOperand_NoMatch;
  }
  // if the first token is not '$' we have an error
  if (Parser.getTok().isNot(AsmToken::Dollar))
    return MatchOperand_NoMatch;

  Parser.Lex(); // Eat $
  if(!tryParseRegisterOperand(Operands, false)) {
    // set the propper register kind
    MipsOperand* op = static_cast<MipsOperand*>(Operands.back());
    op->setRegKind(MipsOperand::Kind_CPURegs);
    return MatchOperand_Success;
  }
  return MatchOperand_NoMatch;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseHWRegs(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {

  if (isMips64())
    return MatchOperand_NoMatch;

  // if the first token is not '$' we have error
  if (Parser.getTok().isNot(AsmToken::Dollar))
    return MatchOperand_NoMatch;
  SMLoc S = Parser.getTok().getLoc();
  Parser.Lex(); // Eat $

  const AsmToken &Tok = Parser.getTok(); // get next token
  if (Tok.isNot(AsmToken::Integer))
    return MatchOperand_NoMatch;

  unsigned RegNum = Tok.getIntVal();
  // at the moment only hwreg29 is supported
  if (RegNum != 29)
    return MatchOperand_ParseFail;

  MipsOperand *op = MipsOperand::CreateReg(Mips::HWR29, S,
        Parser.getTok().getLoc());
  op->setRegKind(MipsOperand::Kind_HWRegs);
  Operands.push_back(op);

  Parser.Lex(); // Eat reg number
  return MatchOperand_Success;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseHW64Regs(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {

  if (!isMips64())
    return MatchOperand_NoMatch;
    //if the first token is not '$' we have error
  if (Parser.getTok().isNot(AsmToken::Dollar))
    return MatchOperand_NoMatch;
  SMLoc S = Parser.getTok().getLoc();
  Parser.Lex(); // Eat $

  const AsmToken &Tok = Parser.getTok(); // get next token
  if (Tok.isNot(AsmToken::Integer))
    return MatchOperand_NoMatch;

  unsigned RegNum = Tok.getIntVal();
  // at the moment only hwreg29 is supported
  if (RegNum != 29)
    return MatchOperand_ParseFail;

  MipsOperand *op = MipsOperand::CreateReg(Mips::HWR29_64, S,
        Parser.getTok().getLoc());
  op->setRegKind(MipsOperand::Kind_HW64Regs);
  Operands.push_back(op);

  Parser.Lex(); // Eat reg number
  return MatchOperand_Success;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseCCRRegs(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  unsigned RegNum;
  //if the first token is not '$' we have error
  if (Parser.getTok().isNot(AsmToken::Dollar))
    return MatchOperand_NoMatch;
  SMLoc S = Parser.getTok().getLoc();
  Parser.Lex(); // Eat $

  const AsmToken &Tok = Parser.getTok(); // get next token
  if (Tok.is(AsmToken::Integer)) {
    RegNum = Tok.getIntVal();
    // at the moment only fcc0 is supported
    if (RegNum != 0)
      return MatchOperand_ParseFail;
  } else if (Tok.is(AsmToken::Identifier)) {
    // at the moment only fcc0 is supported
    if (Tok.getIdentifier() != "fcc0")
      return MatchOperand_ParseFail;
  } else
    return MatchOperand_NoMatch;

  MipsOperand *op = MipsOperand::CreateReg(Mips::FCC0, S,
        Parser.getTok().getLoc());
  op->setRegKind(MipsOperand::Kind_CCRRegs);
  Operands.push_back(op);

  Parser.Lex(); // Eat reg number
  return MatchOperand_Success;
}

MCSymbolRefExpr::VariantKind MipsAsmParser::getVariantKind(StringRef Symbol) {

  MCSymbolRefExpr::VariantKind VK
                   = StringSwitch<MCSymbolRefExpr::VariantKind>(Symbol)
    .Case("hi",          MCSymbolRefExpr::VK_Mips_ABS_HI)
    .Case("lo",          MCSymbolRefExpr::VK_Mips_ABS_LO)
    .Case("gp_rel",      MCSymbolRefExpr::VK_Mips_GPREL)
    .Case("call16",      MCSymbolRefExpr::VK_Mips_GOT_CALL)
    .Case("got",         MCSymbolRefExpr::VK_Mips_GOT)
    .Case("tlsgd",       MCSymbolRefExpr::VK_Mips_TLSGD)
    .Case("tlsldm",      MCSymbolRefExpr::VK_Mips_TLSLDM)
    .Case("dtprel_hi",   MCSymbolRefExpr::VK_Mips_DTPREL_HI)
    .Case("dtprel_lo",   MCSymbolRefExpr::VK_Mips_DTPREL_LO)
    .Case("gottprel",    MCSymbolRefExpr::VK_Mips_GOTTPREL)
    .Case("tprel_hi",    MCSymbolRefExpr::VK_Mips_TPREL_HI)
    .Case("tprel_lo",    MCSymbolRefExpr::VK_Mips_TPREL_LO)
    .Case("got_disp",    MCSymbolRefExpr::VK_Mips_GOT_DISP)
    .Case("got_page",    MCSymbolRefExpr::VK_Mips_GOT_PAGE)
    .Case("got_ofst",    MCSymbolRefExpr::VK_Mips_GOT_OFST)
    .Case("hi(%neg(%gp_rel",    MCSymbolRefExpr::VK_Mips_GPOFF_HI)
    .Case("lo(%neg(%gp_rel",    MCSymbolRefExpr::VK_Mips_GPOFF_LO)
    .Default(MCSymbolRefExpr::VK_None);

  return VK;
}

static int ConvertCcString(StringRef CondString) {
  int CC = StringSwitch<unsigned>(CondString)
      .Case(".f",    0)
      .Case(".un",   1)
      .Case(".eq",   2)
      .Case(".ueq",  3)
      .Case(".olt",  4)
      .Case(".ult",  5)
      .Case(".ole",  6)
      .Case(".ule",  7)
      .Case(".sf",   8)
      .Case(".ngle", 9)
      .Case(".seq",  10)
      .Case(".ngl",  11)
      .Case(".lt",   12)
      .Case(".nge",  13)
      .Case(".le",   14)
      .Case(".ngt",  15)
      .Default(-1);

  return CC;
}

bool MipsAsmParser::
parseMathOperation(StringRef Name, SMLoc NameLoc,
                   SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  // split the format
  size_t Start = Name.find('.'), Next = Name.rfind('.');
  StringRef Format1 = Name.slice(Start, Next);
  // and add the first format to the operands
  Operands.push_back(MipsOperand::CreateToken(Format1, NameLoc));
  // now for the second format
  StringRef Format2 = Name.slice(Next, StringRef::npos);
  Operands.push_back(MipsOperand::CreateToken(Format2, NameLoc));

  // set the format for the first register
  setFpFormat(Format1);

  // Read the remaining operands.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    // Read the first operand.
    if (ParseOperand(Operands, Name)) {
      SMLoc Loc = getLexer().getLoc();
      Parser.eatToEndOfStatement();
      return Error(Loc, "unexpected token in argument list");
    }

    if (getLexer().isNot(AsmToken::Comma)) {
      SMLoc Loc = getLexer().getLoc();
      Parser.eatToEndOfStatement();
      return Error(Loc, "unexpected token in argument list");

    }
    Parser.Lex();  // Eat the comma.

    //set the format for the first register
    setFpFormat(Format2);

    // Parse and remember the operand.
    if (ParseOperand(Operands, Name)) {
      SMLoc Loc = getLexer().getLoc();
      Parser.eatToEndOfStatement();
      return Error(Loc, "unexpected token in argument list");
    }
  }

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    SMLoc Loc = getLexer().getLoc();
    Parser.eatToEndOfStatement();
    return Error(Loc, "unexpected token in argument list");
  }

  Parser.Lex(); // Consume the EndOfStatement
  return false;
}

bool MipsAsmParser::
ParseInstruction(ParseInstructionInfo &Info, StringRef Name, SMLoc NameLoc,
                 SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  StringRef Mnemonic;
  // floating point instructions: should register be treated as double?
  if (requestsDoubleOperand(Name)) {
    setFpFormat(FP_FORMAT_D);
  Operands.push_back(MipsOperand::CreateToken(Name, NameLoc));
  Mnemonic = Name;
  }
  else {
    setDefaultFpFormat();
    // Create the leading tokens for the mnemonic, split by '.' characters.
    size_t Start = 0, Next = Name.find('.');
    Mnemonic = Name.slice(Start, Next);

    Operands.push_back(MipsOperand::CreateToken(Mnemonic, NameLoc));

    if (Next != StringRef::npos) {
      // there is a format token in mnemonic
      // StringRef Rest = Name.slice(Next, StringRef::npos);
      size_t Dot = Name.find('.', Next+1);
      StringRef Format = Name.slice(Next, Dot);
      if (Dot == StringRef::npos) //only one '.' in a string, it's a format
        Operands.push_back(MipsOperand::CreateToken(Format, NameLoc));
      else {
        if (Name.startswith("c.")){
          // floating point compare, add '.' and immediate represent for cc
          Operands.push_back(MipsOperand::CreateToken(".", NameLoc));
          int Cc = ConvertCcString(Format);
          if (Cc == -1) {
            return Error(NameLoc, "Invalid conditional code");
          }
          SMLoc E = SMLoc::getFromPointer(
              Parser.getTok().getLoc().getPointer() -1 );
          Operands.push_back(MipsOperand::CreateImm(
              MCConstantExpr::Create(Cc, getContext()), NameLoc, E));
        } else {
          // trunc, ceil, floor ...
          return parseMathOperation(Name, NameLoc, Operands);
        }

        // the rest is a format
        Format = Name.slice(Dot, StringRef::npos);
        Operands.push_back(MipsOperand::CreateToken(Format, NameLoc));
      }

      setFpFormat(Format);
    }
  }

  // Read the remaining operands.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    // Read the first operand.
    if (ParseOperand(Operands, Mnemonic)) {
      SMLoc Loc = getLexer().getLoc();
      Parser.eatToEndOfStatement();
      return Error(Loc, "unexpected token in argument list");
    }

    while (getLexer().is(AsmToken::Comma) ) {
      Parser.Lex();  // Eat the comma.

      // Parse and remember the operand.
      if (ParseOperand(Operands, Name)) {
        SMLoc Loc = getLexer().getLoc();
        Parser.eatToEndOfStatement();
        return Error(Loc, "unexpected token in argument list");
      }
    }
  }

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    SMLoc Loc = getLexer().getLoc();
    Parser.eatToEndOfStatement();
    return Error(Loc, "unexpected token in argument list");
  }

  Parser.Lex(); // Consume the EndOfStatement
  return false;
}

bool MipsAsmParser::reportParseError(StringRef ErrorMsg) {
   SMLoc Loc = getLexer().getLoc();
   Parser.eatToEndOfStatement();
   return Error(Loc, ErrorMsg);
}

bool MipsAsmParser::parseSetNoAtDirective() {
  // Line should look like:
  //  .set noat
  // set at reg to 0
  Options.setATReg(0);
  // eat noat
  Parser.Lex();
  // If this is not the end of the statement, report error
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token in statement");
    return false;
  }
  Parser.Lex(); // Consume the EndOfStatement
  return false;
}
bool MipsAsmParser::parseSetAtDirective() {
  // line can be
  //  .set at - defaults to $1
  // or .set at=$reg
  int AtRegNo;
  getParser().Lex();
  if (getLexer().is(AsmToken::EndOfStatement)) {
    Options.setATReg(1);
    Parser.Lex(); // Consume the EndOfStatement
    return false;
  } else if (getLexer().is(AsmToken::Equal)) {
    getParser().Lex(); // eat '='
    if (getLexer().isNot(AsmToken::Dollar)) {
      reportParseError("unexpected token in statement");
      return false;
    }
    Parser.Lex(); // Eat '$'
    const AsmToken &Reg = Parser.getTok();
    if (Reg.is(AsmToken::Identifier)) {
      AtRegNo = matchCPURegisterName(Reg.getIdentifier());
    } else if (Reg.is(AsmToken::Integer)) {
      AtRegNo = Reg.getIntVal();
    } else {
      reportParseError("unexpected token in statement");
      return false;
    }

    if ( AtRegNo < 1 || AtRegNo > 31) {
      reportParseError("unexpected token in statement");
      return false;
    }

    if (!Options.setATReg(AtRegNo)) {
      reportParseError("unexpected token in statement");
      return false;
    }
    getParser().Lex(); // Eat reg

    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      reportParseError("unexpected token in statement");
      return false;
     }
    Parser.Lex(); // Consume the EndOfStatement
    return false;
  } else {
    reportParseError("unexpected token in statement");
    return false;
  }
}

bool MipsAsmParser::parseSetReorderDirective() {
  Parser.Lex();
  // If this is not the end of the statement, report error
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token in statement");
    return false;
  }
  Options.setReorder();
  Parser.Lex(); // Consume the EndOfStatement
  return false;
}

bool MipsAsmParser::parseSetNoReorderDirective() {
    Parser.Lex();
    // if this is not the end of the statement, report error
    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      reportParseError("unexpected token in statement");
      return false;
    }
    Options.setNoreorder();
    Parser.Lex(); // Consume the EndOfStatement
    return false;
}

bool MipsAsmParser::parseSetMacroDirective() {
  Parser.Lex();
  // if this is not the end of the statement, report error
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token in statement");
    return false;
  }
  Options.setMacro();
  Parser.Lex(); // Consume the EndOfStatement
  return false;
}

bool MipsAsmParser::parseSetNoMacroDirective() {
  Parser.Lex();
  // if this is not the end of the statement, report error
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("`noreorder' must be set before `nomacro'");
    return false;
  }
  if (Options.isReorder()) {
    reportParseError("`noreorder' must be set before `nomacro'");
    return false;
  }
  Options.setNomacro();
  Parser.Lex(); // Consume the EndOfStatement
  return false;
}

bool MipsAsmParser::parseSetAssignment() {
  StringRef Name;
  const MCExpr *Value;

  if (Parser.parseIdentifier(Name))
    reportParseError("expected identifier after .set");

  if (getLexer().isNot(AsmToken::Comma))
    return reportParseError("unexpected token in .set directive");
  Lex(); //eat comma

  if (Parser.parseExpression(Value))
    reportParseError("expected valid expression after comma");

  // check if the Name already exists as a symbol
  MCSymbol *Sym = getContext().LookupSymbol(Name);
  if (Sym) {
    return reportParseError("symbol already defined");
  }
  Sym = getContext().GetOrCreateSymbol(Name);
  Sym->setVariableValue(Value);

  return false;
}
bool MipsAsmParser::parseDirectiveSet() {

  // get next token
  const AsmToken &Tok = Parser.getTok();

  if (Tok.getString() == "noat") {
    return parseSetNoAtDirective();
  } else if (Tok.getString() == "at") {
    return parseSetAtDirective();
  } else if (Tok.getString() == "reorder") {
    return parseSetReorderDirective();
  } else if (Tok.getString() == "noreorder") {
    return parseSetNoReorderDirective();
  } else if (Tok.getString() == "macro") {
    return parseSetMacroDirective();
  } else if (Tok.getString() == "nomacro") {
    return parseSetNoMacroDirective();
  } else if (Tok.getString() == "nomips16") {
    // ignore this directive for now
    Parser.eatToEndOfStatement();
    return false;
  } else if (Tok.getString() == "nomicromips") {
    // ignore this directive for now
    Parser.eatToEndOfStatement();
    return false;
  } else {
    // it is just an identifier, look for assignment
    parseSetAssignment();
    return false;
  }

  return true;
}

/// parseDirectiveWord
///  ::= .word [ expression (, expression)* ]
bool MipsAsmParser::parseDirectiveWord(unsigned Size, SMLoc L) {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      const MCExpr *Value;
      if (getParser().parseExpression(Value))
        return true;

      getParser().getStreamer().EmitValue(Value, Size);

      if (getLexer().is(AsmToken::EndOfStatement))
        break;

      // FIXME: Improve diagnostic.
      if (getLexer().isNot(AsmToken::Comma))
        return Error(L, "unexpected token in directive");
      Parser.Lex();
    }
  }

  Parser.Lex();
  return false;
}

bool MipsAsmParser::ParseDirective(AsmToken DirectiveID) {

  StringRef IDVal = DirectiveID.getString();

  if ( IDVal == ".ent") {
    // ignore this directive for now
    Parser.Lex();
    return false;
  }

  if (IDVal == ".end") {
    // ignore this directive for now
    Parser.Lex();
    return false;
  }

  if (IDVal == ".frame") {
    // ignore this directive for now
    Parser.eatToEndOfStatement();
    return false;
  }

  if (IDVal == ".set") {
    return parseDirectiveSet();
  }

  if (IDVal == ".fmask") {
    // ignore this directive for now
    Parser.eatToEndOfStatement();
    return false;
  }

  if (IDVal == ".mask") {
    // ignore this directive for now
    Parser.eatToEndOfStatement();
    return false;
  }

  if (IDVal == ".gpword") {
    // ignore this directive for now
    Parser.eatToEndOfStatement();
    return false;
  }

  if (IDVal == ".word") {
    parseDirectiveWord(4, DirectiveID.getLoc());
    return false;
  }

  return true;
}

extern "C" void LLVMInitializeMipsAsmParser() {
  RegisterMCAsmParser<MipsAsmParser> X(TheMipsTarget);
  RegisterMCAsmParser<MipsAsmParser> Y(TheMipselTarget);
  RegisterMCAsmParser<MipsAsmParser> A(TheMips64Target);
  RegisterMCAsmParser<MipsAsmParser> B(TheMips64elTarget);
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "MipsGenAsmMatcher.inc"
