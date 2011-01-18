//===-- ARMAsmParser.cpp - Parse ARM assembly to MCInst instructions ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMMCExpr.h"
#include "ARMBaseRegisterInfo.h"
#include "ARMSubtarget.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetAsmParser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
using namespace llvm;

/// Shift types used for register controlled shifts in ARM memory addressing.
enum ShiftType {
  Lsl,
  Lsr,
  Asr,
  Ror,
  Rrx
};

namespace {

class ARMOperand;

class ARMAsmParser : public TargetAsmParser {
  MCAsmParser &Parser;
  TargetMachine &TM;

  MCAsmParser &getParser() const { return Parser; }
  MCAsmLexer &getLexer() const { return Parser.getLexer(); }

  void Warning(SMLoc L, const Twine &Msg) { Parser.Warning(L, Msg); }
  bool Error(SMLoc L, const Twine &Msg) { return Parser.Error(L, Msg); }

  int TryParseRegister();
  bool TryParseMCRName(SmallVectorImpl<MCParsedAsmOperand*>&);
  bool TryParseRegisterWithWriteBack(SmallVectorImpl<MCParsedAsmOperand*> &);
  bool ParseRegisterList(SmallVectorImpl<MCParsedAsmOperand*> &);
  bool ParseMemory(SmallVectorImpl<MCParsedAsmOperand*> &);
  bool ParseOperand(SmallVectorImpl<MCParsedAsmOperand*> &, bool isMCR);
  bool ParsePrefix(ARMMCExpr::VariantKind &RefKind);
  const MCExpr *ApplyPrefixToExpr(const MCExpr *E,
                                  MCSymbolRefExpr::VariantKind Variant);


  bool ParseMemoryOffsetReg(bool &Negative,
                            bool &OffsetRegShifted,
                            enum ShiftType &ShiftType,
                            const MCExpr *&ShiftAmount,
                            const MCExpr *&Offset,
                            bool &OffsetIsReg,
                            int &OffsetRegNum,
                            SMLoc &E);
  bool ParseShift(enum ShiftType &St, const MCExpr *&ShiftAmount, SMLoc &E);
  bool ParseDirectiveWord(unsigned Size, SMLoc L);
  bool ParseDirectiveThumb(SMLoc L);
  bool ParseDirectiveThumbFunc(SMLoc L);
  bool ParseDirectiveCode(SMLoc L);
  bool ParseDirectiveSyntax(SMLoc L);

  bool MatchAndEmitInstruction(SMLoc IDLoc,
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                               MCStreamer &Out);

  /// @name Auto-generated Match Functions
  /// {

#define GET_ASSEMBLER_HEADER
#include "ARMGenAsmMatcher.inc"

  /// }

public:
  ARMAsmParser(const Target &T, MCAsmParser &_Parser, TargetMachine &_TM)
    : TargetAsmParser(T), Parser(_Parser), TM(_TM) {
      // Initialize the set of available features.
      setAvailableFeatures(ComputeAvailableFeatures(
          &TM.getSubtarget<ARMSubtarget>()));
    }

  virtual bool ParseInstruction(StringRef Name, SMLoc NameLoc,
                                SmallVectorImpl<MCParsedAsmOperand*> &Operands);
  virtual bool ParseDirective(AsmToken DirectiveID);
};
} // end anonymous namespace

namespace {

/// ARMOperand - Instances of this class represent a parsed ARM machine
/// instruction.
class ARMOperand : public MCParsedAsmOperand {
  enum KindTy {
    CondCode,
    CCOut,
    Immediate,
    Memory,
    Register,
    RegisterList,
    DPRRegisterList,
    SPRRegisterList,
    Token
  } Kind;

  SMLoc StartLoc, EndLoc;
  SmallVector<unsigned, 8> Registers;

  union {
    struct {
      ARMCC::CondCodes Val;
    } CC;

    struct {
      const char *Data;
      unsigned Length;
    } Tok;

    struct {
      unsigned RegNum;
    } Reg;

    struct {
      const MCExpr *Val;
    } Imm;

    /// Combined record for all forms of ARM address expressions.
    struct {
      unsigned BaseRegNum;
      union {
        unsigned RegNum;     ///< Offset register num, when OffsetIsReg.
        const MCExpr *Value; ///< Offset value, when !OffsetIsReg.
      } Offset;
      const MCExpr *ShiftAmount;     // used when OffsetRegShifted is true
      enum ShiftType ShiftType;      // used when OffsetRegShifted is true
      unsigned OffsetRegShifted : 1; // only used when OffsetIsReg is true
      unsigned Preindexed       : 1;
      unsigned Postindexed      : 1;
      unsigned OffsetIsReg      : 1;
      unsigned Negative         : 1; // only used when OffsetIsReg is true
      unsigned Writeback        : 1;
    } Mem;
  };

  ARMOperand(KindTy K) : MCParsedAsmOperand(), Kind(K) {}
public:
  ARMOperand(const ARMOperand &o) : MCParsedAsmOperand() {
    Kind = o.Kind;
    StartLoc = o.StartLoc;
    EndLoc = o.EndLoc;
    switch (Kind) {
    case CondCode:
      CC = o.CC;
      break;
    case Token:
      Tok = o.Tok;
      break;
    case CCOut:
    case Register:
      Reg = o.Reg;
      break;
    case RegisterList:
    case DPRRegisterList:
    case SPRRegisterList:
      Registers = o.Registers;
      break;
    case Immediate:
      Imm = o.Imm;
      break;
    case Memory:
      Mem = o.Mem;
      break;
    }
  }

  /// getStartLoc - Get the location of the first token of this operand.
  SMLoc getStartLoc() const { return StartLoc; }
  /// getEndLoc - Get the location of the last token of this operand.
  SMLoc getEndLoc() const { return EndLoc; }

  ARMCC::CondCodes getCondCode() const {
    assert(Kind == CondCode && "Invalid access!");
    return CC.Val;
  }

  StringRef getToken() const {
    assert(Kind == Token && "Invalid access!");
    return StringRef(Tok.Data, Tok.Length);
  }

  unsigned getReg() const {
    assert((Kind == Register || Kind == CCOut) && "Invalid access!");
    return Reg.RegNum;
  }

  const SmallVectorImpl<unsigned> &getRegList() const {
    assert((Kind == RegisterList || Kind == DPRRegisterList ||
            Kind == SPRRegisterList) && "Invalid access!");
    return Registers;
  }

  const MCExpr *getImm() const {
    assert(Kind == Immediate && "Invalid access!");
    return Imm.Val;
  }

  /// @name Memory Operand Accessors
  /// @{

  unsigned getMemBaseRegNum() const {
    return Mem.BaseRegNum;
  }
  unsigned getMemOffsetRegNum() const {
    assert(Mem.OffsetIsReg && "Invalid access!");
    return Mem.Offset.RegNum;
  }
  const MCExpr *getMemOffset() const {
    assert(!Mem.OffsetIsReg && "Invalid access!");
    return Mem.Offset.Value;
  }
  unsigned getMemOffsetRegShifted() const {
    assert(Mem.OffsetIsReg && "Invalid access!");
    return Mem.OffsetRegShifted;
  }
  const MCExpr *getMemShiftAmount() const {
    assert(Mem.OffsetIsReg && Mem.OffsetRegShifted && "Invalid access!");
    return Mem.ShiftAmount;
  }
  enum ShiftType getMemShiftType() const {
    assert(Mem.OffsetIsReg && Mem.OffsetRegShifted && "Invalid access!");
    return Mem.ShiftType;
  }
  bool getMemPreindexed() const { return Mem.Preindexed; }
  bool getMemPostindexed() const { return Mem.Postindexed; }
  bool getMemOffsetIsReg() const { return Mem.OffsetIsReg; }
  bool getMemNegative() const { return Mem.Negative; }
  bool getMemWriteback() const { return Mem.Writeback; }

  /// @}

  bool isCondCode() const { return Kind == CondCode; }
  bool isCCOut() const { return Kind == CCOut; }
  bool isImm() const { return Kind == Immediate; }
  bool isReg() const { return Kind == Register; }
  bool isRegList() const { return Kind == RegisterList; }
  bool isDPRRegList() const { return Kind == DPRRegisterList; }
  bool isSPRRegList() const { return Kind == SPRRegisterList; }
  bool isToken() const { return Kind == Token; }
  bool isMemory() const { return Kind == Memory; }
  bool isMemMode5() const {
    if (!isMemory() || Mem.OffsetIsReg || Mem.OffsetRegShifted ||
        Mem.Writeback || Mem.Negative)
      return false;

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Mem.Offset.Value);
    if (!CE) return false;

    // The offset must be a multiple of 4 in the range 0-1020.
    int64_t Value = CE->getValue();
    return ((Value & 0x3) == 0 && Value <= 1020 && Value >= -1020);
  }
  bool isMemModeRegThumb() const {
    if (!isMemory() || !Mem.OffsetIsReg || Mem.Writeback)
      return false;
    return true;
  }
  bool isMemModeImmThumb() const {
    if (!isMemory() || Mem.OffsetIsReg || Mem.Writeback)
      return false;

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Mem.Offset.Value);
    if (!CE) return false;

    // The offset must be a multiple of 4 in the range 0-124.
    uint64_t Value = CE->getValue();
    return ((Value & 0x3) == 0 && Value <= 124);
  }

  void addExpr(MCInst &Inst, const MCExpr *Expr) const {
    // Add as immediates when possible.  Null MCExpr = 0.
    if (Expr == 0)
      Inst.addOperand(MCOperand::CreateImm(0));
    else if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::CreateImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::CreateExpr(Expr));
  }

  void addCondCodeOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(unsigned(getCondCode())));
    unsigned RegNum = getCondCode() == ARMCC::AL ? 0: ARM::CPSR;
    Inst.addOperand(MCOperand::CreateReg(RegNum));
  }

  void addCCOutOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getReg()));
  }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getReg()));
  }

  void addRegListOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const SmallVectorImpl<unsigned> &RegList = getRegList();
    for (SmallVectorImpl<unsigned>::const_iterator
           I = RegList.begin(), E = RegList.end(); I != E; ++I)
      Inst.addOperand(MCOperand::CreateReg(*I));
  }

  void addDPRRegListOperands(MCInst &Inst, unsigned N) const {
    addRegListOperands(Inst, N);
  }

  void addSPRRegListOperands(MCInst &Inst, unsigned N) const {
    addRegListOperands(Inst, N);
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }

  void addMemMode5Operands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && isMemMode5() && "Invalid number of operands!");

    Inst.addOperand(MCOperand::CreateReg(Mem.BaseRegNum));
    assert(!Mem.OffsetIsReg && "Invalid mode 5 operand");

    // FIXME: #-0 is encoded differently than #0. Does the parser preserve
    // the difference?
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Mem.Offset.Value);
    assert(CE && "Non-constant mode 5 offset operand!");

    // The MCInst offset operand doesn't include the low two bits (like
    // the instruction encoding).
    int64_t Offset = CE->getValue() / 4;
    if (Offset >= 0)
      Inst.addOperand(MCOperand::CreateImm(ARM_AM::getAM5Opc(ARM_AM::add,
                                                             Offset)));
    else
      Inst.addOperand(MCOperand::CreateImm(ARM_AM::getAM5Opc(ARM_AM::sub,
                                                             -Offset)));
  }

  void addMemModeRegThumbOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && isMemModeRegThumb() && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(Mem.BaseRegNum));
    Inst.addOperand(MCOperand::CreateReg(Mem.Offset.RegNum));
  }

  void addMemModeImmThumbOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && isMemModeImmThumb() && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(Mem.BaseRegNum));
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Mem.Offset.Value);
    assert(CE && "Non-constant mode offset operand!");
    Inst.addOperand(MCOperand::CreateImm(CE->getValue()));
  }

  virtual void dump(raw_ostream &OS) const;

  static ARMOperand *CreateCondCode(ARMCC::CondCodes CC, SMLoc S) {
    ARMOperand *Op = new ARMOperand(CondCode);
    Op->CC.Val = CC;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static ARMOperand *CreateCCOut(unsigned RegNum, SMLoc S) {
    ARMOperand *Op = new ARMOperand(CCOut);
    Op->Reg.RegNum = RegNum;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static ARMOperand *CreateToken(StringRef Str, SMLoc S) {
    ARMOperand *Op = new ARMOperand(Token);
    Op->Tok.Data = Str.data();
    Op->Tok.Length = Str.size();
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static ARMOperand *CreateReg(unsigned RegNum, SMLoc S, SMLoc E) {
    ARMOperand *Op = new ARMOperand(Register);
    Op->Reg.RegNum = RegNum;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static ARMOperand *
  CreateRegList(const SmallVectorImpl<std::pair<unsigned, SMLoc> > &Regs,
                SMLoc StartLoc, SMLoc EndLoc) {
    KindTy Kind = RegisterList;

    if (ARM::DPRRegClass.contains(Regs.front().first))
      Kind = DPRRegisterList;
    else if (ARM::SPRRegClass.contains(Regs.front().first))
      Kind = SPRRegisterList;

    ARMOperand *Op = new ARMOperand(Kind);
    for (SmallVectorImpl<std::pair<unsigned, SMLoc> >::const_iterator
           I = Regs.begin(), E = Regs.end(); I != E; ++I)
      Op->Registers.push_back(I->first);
    array_pod_sort(Op->Registers.begin(), Op->Registers.end());
    Op->StartLoc = StartLoc;
    Op->EndLoc = EndLoc;
    return Op;
  }

  static ARMOperand *CreateImm(const MCExpr *Val, SMLoc S, SMLoc E) {
    ARMOperand *Op = new ARMOperand(Immediate);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static ARMOperand *CreateMem(unsigned BaseRegNum, bool OffsetIsReg,
                               const MCExpr *Offset, int OffsetRegNum,
                               bool OffsetRegShifted, enum ShiftType ShiftType,
                               const MCExpr *ShiftAmount, bool Preindexed,
                               bool Postindexed, bool Negative, bool Writeback,
                               SMLoc S, SMLoc E) {
    assert((OffsetRegNum == -1 || OffsetIsReg) &&
           "OffsetRegNum must imply OffsetIsReg!");
    assert((!OffsetRegShifted || OffsetIsReg) &&
           "OffsetRegShifted must imply OffsetIsReg!");
    assert((Offset || OffsetIsReg) &&
           "Offset must exists unless register offset is used!");
    assert((!ShiftAmount || (OffsetIsReg && OffsetRegShifted)) &&
           "Cannot have shift amount without shifted register offset!");
    assert((!Offset || !OffsetIsReg) &&
           "Cannot have expression offset and register offset!");

    ARMOperand *Op = new ARMOperand(Memory);
    Op->Mem.BaseRegNum = BaseRegNum;
    Op->Mem.OffsetIsReg = OffsetIsReg;
    if (OffsetIsReg)
      Op->Mem.Offset.RegNum = OffsetRegNum;
    else
      Op->Mem.Offset.Value = Offset;
    Op->Mem.OffsetRegShifted = OffsetRegShifted;
    Op->Mem.ShiftType = ShiftType;
    Op->Mem.ShiftAmount = ShiftAmount;
    Op->Mem.Preindexed = Preindexed;
    Op->Mem.Postindexed = Postindexed;
    Op->Mem.Negative = Negative;
    Op->Mem.Writeback = Writeback;

    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }
};

} // end anonymous namespace.

void ARMOperand::dump(raw_ostream &OS) const {
  switch (Kind) {
  case CondCode:
    OS << "<ARMCC::" << ARMCondCodeToString(getCondCode()) << ">";
    break;
  case CCOut:
    OS << "<ccout " << getReg() << ">";
    break;
  case Immediate:
    getImm()->print(OS);
    break;
  case Memory:
    OS << "<memory "
       << "base:" << getMemBaseRegNum();
    if (getMemOffsetIsReg()) {
      OS << " offset:<register " << getMemOffsetRegNum();
      if (getMemOffsetRegShifted()) {
        OS << " offset-shift-type:" << getMemShiftType();
        OS << " offset-shift-amount:" << *getMemShiftAmount();
      }
    } else {
      OS << " offset:" << *getMemOffset();
    }
    if (getMemOffsetIsReg())
      OS << " (offset-is-reg)";
    if (getMemPreindexed())
      OS << " (pre-indexed)";
    if (getMemPostindexed())
      OS << " (post-indexed)";
    if (getMemNegative())
      OS << " (negative)";
    if (getMemWriteback())
      OS << " (writeback)";
    OS << ">";
    break;
  case Register:
    OS << "<register " << getReg() << ">";
    break;
  case RegisterList:
  case DPRRegisterList:
  case SPRRegisterList: {
    OS << "<register_list ";

    const SmallVectorImpl<unsigned> &RegList = getRegList();
    for (SmallVectorImpl<unsigned>::const_iterator
           I = RegList.begin(), E = RegList.end(); I != E; ) {
      OS << *I;
      if (++I < E) OS << ", ";
    }

    OS << ">";
    break;
  }
  case Token:
    OS << "'" << getToken() << "'";
    break;
  }
}

/// @name Auto-generated Match Functions
/// {

static unsigned MatchRegisterName(StringRef Name);

/// }

/// Try to parse a register name.  The token must be an Identifier when called,
/// and if it is a register name the token is eaten and the register number is
/// returned.  Otherwise return -1.
///
int ARMAsmParser::TryParseRegister() {
  const AsmToken &Tok = Parser.getTok();
  assert(Tok.is(AsmToken::Identifier) && "Token is not an Identifier");

  // FIXME: Validate register for the current architecture; we have to do
  // validation later, so maybe there is no need for this here.
  std::string upperCase = Tok.getString().str();
  std::string lowerCase = LowercaseString(upperCase);
  unsigned RegNum = MatchRegisterName(lowerCase);
  if (!RegNum) {
    RegNum = StringSwitch<unsigned>(lowerCase)
      .Case("r13", ARM::SP)
      .Case("r14", ARM::LR)
      .Case("r15", ARM::PC)
      .Case("ip", ARM::R12)
      .Default(0);
  }
  if (!RegNum) return -1;
  
  Parser.Lex(); // Eat identifier token.
  return RegNum;
}


/// Try to parse a register name.  The token must be an Identifier when called.
/// If it's a register, an AsmOperand is created. Another AsmOperand is created
/// if there is a "writeback". 'true' if it's not a register.
///
/// TODO this is likely to change to allow different register types and or to
/// parse for a specific register type.
bool ARMAsmParser::
TryParseRegisterWithWriteBack(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  SMLoc S = Parser.getTok().getLoc();
  int RegNo = TryParseRegister();
  if (RegNo == -1)
    return true;

  Operands.push_back(ARMOperand::CreateReg(RegNo, S, Parser.getTok().getLoc()));

  const AsmToken &ExclaimTok = Parser.getTok();
  if (ExclaimTok.is(AsmToken::Exclaim)) {
    Operands.push_back(ARMOperand::CreateToken(ExclaimTok.getString(),
                                               ExclaimTok.getLoc()));
    Parser.Lex(); // Eat exclaim token
  }

  return false;
}

static int MatchMCRName(StringRef Name) {
  // Use the same layout as the tablegen'erated register name matcher. Ugly,
  // but efficient.
  switch (Name.size()) {
  default: break;
  case 2:
    if (Name[0] != 'p' && Name[0] != 'c')
      return -1;
    switch (Name[1]) {
    default:  return -1;
    case '0': return 0;
    case '1': return 1;
    case '2': return 2;
    case '3': return 3;
    case '4': return 4;
    case '5': return 5;
    case '6': return 6;
    case '7': return 7;
    case '8': return 8;
    case '9': return 9;
    }
    break;
  case 3:
    if ((Name[0] != 'p' && Name[0] != 'c') || Name[1] != '1')
      return -1;
    switch (Name[2]) {
    default:  return -1;
    case '0': return 10;
    case '1': return 11;
    case '2': return 12;
    case '3': return 13;
    case '4': return 14;
    case '5': return 15;
    }
    break;
  }

  llvm_unreachable("Unhandled coprocessor operand string!");
  return -1;
}

/// TryParseMCRName - Try to parse an MCR/MRC symbolic operand
/// name.  The token must be an Identifier when called, and if it is a MCR 
/// operand name, the token is eaten and the operand is added to the
/// operand list.
bool ARMAsmParser::
TryParseMCRName(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  SMLoc S = Parser.getTok().getLoc();
  const AsmToken &Tok = Parser.getTok();
  assert(Tok.is(AsmToken::Identifier) && "Token is not an Identifier");

  int Num = MatchMCRName(Tok.getString());
  if (Num == -1)
    return true;

  Parser.Lex(); // Eat identifier token.
  Operands.push_back(ARMOperand::CreateImm(
       MCConstantExpr::Create(Num, getContext()), S, Parser.getTok().getLoc()));
  return false;
}

/// Parse a register list, return it if successful else return null.  The first
/// token must be a '{' when called.
bool ARMAsmParser::
ParseRegisterList(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  assert(Parser.getTok().is(AsmToken::LCurly) &&
         "Token is not a Left Curly Brace");
  SMLoc S = Parser.getTok().getLoc();

  // Read the rest of the registers in the list.
  unsigned PrevRegNum = 0;
  SmallVector<std::pair<unsigned, SMLoc>, 32> Registers;

  do {
    bool IsRange = Parser.getTok().is(AsmToken::Minus);
    Parser.Lex(); // Eat non-identifier token.

    const AsmToken &RegTok = Parser.getTok();
    SMLoc RegLoc = RegTok.getLoc();
    if (RegTok.isNot(AsmToken::Identifier)) {
      Error(RegLoc, "register expected");
      return true;
    }

    int RegNum = TryParseRegister();
    if (RegNum == -1) {
      Error(RegLoc, "register expected");
      return true;
    }

    if (IsRange) {
      int Reg = PrevRegNum;
      do {
        ++Reg;
        Registers.push_back(std::make_pair(Reg, RegLoc));
      } while (Reg != RegNum);
    } else {
      Registers.push_back(std::make_pair(RegNum, RegLoc));
    }

    PrevRegNum = RegNum;
  } while (Parser.getTok().is(AsmToken::Comma) ||
           Parser.getTok().is(AsmToken::Minus));

  // Process the right curly brace of the list.
  const AsmToken &RCurlyTok = Parser.getTok();
  if (RCurlyTok.isNot(AsmToken::RCurly)) {
    Error(RCurlyTok.getLoc(), "'}' expected");
    return true;
  }

  SMLoc E = RCurlyTok.getLoc();
  Parser.Lex(); // Eat right curly brace token.

  // Verify the register list.
  SmallVectorImpl<std::pair<unsigned, SMLoc> >::const_iterator
    RI = Registers.begin(), RE = Registers.end();

  unsigned HighRegNum = getARMRegisterNumbering(RI->first);
  bool EmittedWarning = false;

  DenseMap<unsigned, bool> RegMap;
  RegMap[HighRegNum] = true;

  for (++RI; RI != RE; ++RI) {
    const std::pair<unsigned, SMLoc> &RegInfo = *RI;
    unsigned Reg = getARMRegisterNumbering(RegInfo.first);

    if (RegMap[Reg]) {
      Error(RegInfo.second, "register duplicated in register list");
      return true;
    }

    if (!EmittedWarning && Reg < HighRegNum)
      Warning(RegInfo.second,
              "register not in ascending order in register list");

    RegMap[Reg] = true;
    HighRegNum = std::max(Reg, HighRegNum);
  }

  Operands.push_back(ARMOperand::CreateRegList(Registers, S, E));
  return false;
}

/// Parse an ARM memory expression, return false if successful else return true
/// or an error.  The first token must be a '[' when called.
///
/// TODO Only preindexing and postindexing addressing are started, unindexed
/// with option, etc are still to do.
bool ARMAsmParser::
ParseMemory(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  SMLoc S, E;
  assert(Parser.getTok().is(AsmToken::LBrac) &&
         "Token is not a Left Bracket");
  S = Parser.getTok().getLoc();
  Parser.Lex(); // Eat left bracket token.

  const AsmToken &BaseRegTok = Parser.getTok();
  if (BaseRegTok.isNot(AsmToken::Identifier)) {
    Error(BaseRegTok.getLoc(), "register expected");
    return true;
  }
  int BaseRegNum = TryParseRegister();
  if (BaseRegNum == -1) {
    Error(BaseRegTok.getLoc(), "register expected");
    return true;
  }

  // The next token must either be a comma or a closing bracket.
  const AsmToken &Tok = Parser.getTok();
  if (!Tok.is(AsmToken::Comma) && !Tok.is(AsmToken::RBrac))
    return true;

  bool Preindexed = false;
  bool Postindexed = false;
  bool OffsetIsReg = false;
  bool Negative = false;
  bool Writeback = false;
  ARMOperand *WBOp = 0;
  int OffsetRegNum = -1;
  bool OffsetRegShifted = false;
  enum ShiftType ShiftType = Lsl;
  const MCExpr *ShiftAmount = 0;
  const MCExpr *Offset = 0;

  // First look for preindexed address forms, that is after the "[Rn" we now
  // have to see if the next token is a comma.
  if (Tok.is(AsmToken::Comma)) {
    Preindexed = true;
    Parser.Lex(); // Eat comma token.

    if (ParseMemoryOffsetReg(Negative, OffsetRegShifted, ShiftType, ShiftAmount,
                             Offset, OffsetIsReg, OffsetRegNum, E))
      return true;
    const AsmToken &RBracTok = Parser.getTok();
    if (RBracTok.isNot(AsmToken::RBrac)) {
      Error(RBracTok.getLoc(), "']' expected");
      return true;
    }
    E = RBracTok.getLoc();
    Parser.Lex(); // Eat right bracket token.

    const AsmToken &ExclaimTok = Parser.getTok();
    if (ExclaimTok.is(AsmToken::Exclaim)) {
      WBOp = ARMOperand::CreateToken(ExclaimTok.getString(),
                                     ExclaimTok.getLoc());
      Writeback = true;
      Parser.Lex(); // Eat exclaim token
    }
  } else {
    // The "[Rn" we have so far was not followed by a comma.

    // If there's anything other than the right brace, this is a post indexing
    // addressing form.
    E = Tok.getLoc();
    Parser.Lex(); // Eat right bracket token.

    const AsmToken &NextTok = Parser.getTok();

    if (NextTok.isNot(AsmToken::EndOfStatement)) {
      Postindexed = true;
      Writeback = true;

      if (NextTok.isNot(AsmToken::Comma)) {
        Error(NextTok.getLoc(), "',' expected");
        return true;
      }

      Parser.Lex(); // Eat comma token.

      if (ParseMemoryOffsetReg(Negative, OffsetRegShifted, ShiftType,
                               ShiftAmount, Offset, OffsetIsReg, OffsetRegNum,
                               E))
        return true;
    }
  }

  // Force Offset to exist if used.
  if (!OffsetIsReg) {
    if (!Offset)
      Offset = MCConstantExpr::Create(0, getContext());
  }

  Operands.push_back(ARMOperand::CreateMem(BaseRegNum, OffsetIsReg, Offset,
                                           OffsetRegNum, OffsetRegShifted,
                                           ShiftType, ShiftAmount, Preindexed,
                                           Postindexed, Negative, Writeback,
                                           S, E));
  if (WBOp)
    Operands.push_back(WBOp);

  return false;
}

/// Parse the offset of a memory operand after we have seen "[Rn," or "[Rn],"
/// we will parse the following (were +/- means that a plus or minus is
/// optional):
///   +/-Rm
///   +/-Rm, shift
///   #offset
/// we return false on success or an error otherwise.
bool ARMAsmParser::ParseMemoryOffsetReg(bool &Negative,
                                        bool &OffsetRegShifted,
                                        enum ShiftType &ShiftType,
                                        const MCExpr *&ShiftAmount,
                                        const MCExpr *&Offset,
                                        bool &OffsetIsReg,
                                        int &OffsetRegNum,
                                        SMLoc &E) {
  Negative = false;
  OffsetRegShifted = false;
  OffsetIsReg = false;
  OffsetRegNum = -1;
  const AsmToken &NextTok = Parser.getTok();
  E = NextTok.getLoc();
  if (NextTok.is(AsmToken::Plus))
    Parser.Lex(); // Eat plus token.
  else if (NextTok.is(AsmToken::Minus)) {
    Negative = true;
    Parser.Lex(); // Eat minus token
  }
  // See if there is a register following the "[Rn," or "[Rn]," we have so far.
  const AsmToken &OffsetRegTok = Parser.getTok();
  if (OffsetRegTok.is(AsmToken::Identifier)) {
    SMLoc CurLoc = OffsetRegTok.getLoc();
    OffsetRegNum = TryParseRegister();
    if (OffsetRegNum != -1) {
      OffsetIsReg = true;
      E = CurLoc;
    }
  }

  // If we parsed a register as the offset then there can be a shift after that.
  if (OffsetRegNum != -1) {
    // Look for a comma then a shift
    const AsmToken &Tok = Parser.getTok();
    if (Tok.is(AsmToken::Comma)) {
      Parser.Lex(); // Eat comma token.

      const AsmToken &Tok = Parser.getTok();
      if (ParseShift(ShiftType, ShiftAmount, E))
        return Error(Tok.getLoc(), "shift expected");
      OffsetRegShifted = true;
    }
  }
  else { // the "[Rn," or "[Rn,]" we have so far was not followed by "Rm"
    // Look for #offset following the "[Rn," or "[Rn],"
    const AsmToken &HashTok = Parser.getTok();
    if (HashTok.isNot(AsmToken::Hash))
      return Error(HashTok.getLoc(), "'#' expected");

    Parser.Lex(); // Eat hash token.

    if (getParser().ParseExpression(Offset))
     return true;
    E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
  }
  return false;
}

/// ParseShift as one of these two:
///   ( lsl | lsr | asr | ror ) , # shift_amount
///   rrx
/// and returns true if it parses a shift otherwise it returns false.
bool ARMAsmParser::ParseShift(ShiftType &St, const MCExpr *&ShiftAmount,
                              SMLoc &E) {
  const AsmToken &Tok = Parser.getTok();
  if (Tok.isNot(AsmToken::Identifier))
    return true;
  StringRef ShiftName = Tok.getString();
  if (ShiftName == "lsl" || ShiftName == "LSL")
    St = Lsl;
  else if (ShiftName == "lsr" || ShiftName == "LSR")
    St = Lsr;
  else if (ShiftName == "asr" || ShiftName == "ASR")
    St = Asr;
  else if (ShiftName == "ror" || ShiftName == "ROR")
    St = Ror;
  else if (ShiftName == "rrx" || ShiftName == "RRX")
    St = Rrx;
  else
    return true;
  Parser.Lex(); // Eat shift type token.

  // Rrx stands alone.
  if (St == Rrx)
    return false;

  // Otherwise, there must be a '#' and a shift amount.
  const AsmToken &HashTok = Parser.getTok();
  if (HashTok.isNot(AsmToken::Hash))
    return Error(HashTok.getLoc(), "'#' expected");
  Parser.Lex(); // Eat hash token.

  if (getParser().ParseExpression(ShiftAmount))
    return true;

  return false;
}

/// Parse a arm instruction operand.  For now this parses the operand regardless
/// of the mnemonic.
bool ARMAsmParser::ParseOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                                bool isMCR){
  SMLoc S, E;
  switch (getLexer().getKind()) {
  default:
    Error(Parser.getTok().getLoc(), "unexpected token in operand");
    return true;
  case AsmToken::Identifier:
    if (!TryParseRegisterWithWriteBack(Operands))
      return false;
    if (isMCR && !TryParseMCRName(Operands))
      return false;

    // Fall though for the Identifier case that is not a register or a
    // special name.
  case AsmToken::Integer: // things like 1f and 2b as a branch targets
  case AsmToken::Dot: {   // . as a branch target
    // This was not a register so parse other operands that start with an
    // identifier (like labels) as expressions and create them as immediates.
    const MCExpr *IdVal;
    S = Parser.getTok().getLoc();
    if (getParser().ParseExpression(IdVal))
      return true;
    E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
    Operands.push_back(ARMOperand::CreateImm(IdVal, S, E));
    return false;
  }
  case AsmToken::LBrac:
    return ParseMemory(Operands);
  case AsmToken::LCurly:
    return ParseRegisterList(Operands);
  case AsmToken::Hash:
    // #42 -> immediate.
    // TODO: ":lower16:" and ":upper16:" modifiers after # before immediate
    S = Parser.getTok().getLoc();
    Parser.Lex();
    const MCExpr *ImmVal;
    if (getParser().ParseExpression(ImmVal))
      return true;
    E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
    Operands.push_back(ARMOperand::CreateImm(ImmVal, S, E));
    return false;
  case AsmToken::Colon: {
    // ":lower16:" and ":upper16:" expression prefixes
    // FIXME: Check it's an expression prefix,
    // e.g. (FOO - :lower16:BAR) isn't legal.
    ARMMCExpr::VariantKind RefKind;
    if (ParsePrefix(RefKind))
      return true;

    const MCExpr *SubExprVal;
    if (getParser().ParseExpression(SubExprVal))
      return true;

    const MCExpr *ExprVal = ARMMCExpr::Create(RefKind, SubExprVal,
                                                   getContext());
    E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
    Operands.push_back(ARMOperand::CreateImm(ExprVal, S, E));
    return false;
  }
  }
}

// ParsePrefix - Parse ARM 16-bit relocations expression prefix, i.e.
//  :lower16: and :upper16:.
bool ARMAsmParser::ParsePrefix(ARMMCExpr::VariantKind &RefKind) {
  RefKind = ARMMCExpr::VK_ARM_None;

  // :lower16: and :upper16: modifiers
  assert(getLexer().is(AsmToken::Colon) && "expected a :");
  Parser.Lex(); // Eat ':'

  if (getLexer().isNot(AsmToken::Identifier)) {
    Error(Parser.getTok().getLoc(), "expected prefix identifier in operand");
    return true;
  }

  StringRef IDVal = Parser.getTok().getIdentifier();
  if (IDVal == "lower16") {
    RefKind = ARMMCExpr::VK_ARM_LO16;
  } else if (IDVal == "upper16") {
    RefKind = ARMMCExpr::VK_ARM_HI16;
  } else {
    Error(Parser.getTok().getLoc(), "unexpected prefix in operand");
    return true;
  }
  Parser.Lex();

  if (getLexer().isNot(AsmToken::Colon)) {
    Error(Parser.getTok().getLoc(), "unexpected token after prefix");
    return true;
  }
  Parser.Lex(); // Eat the last ':'
  return false;
}

const MCExpr *
ARMAsmParser::ApplyPrefixToExpr(const MCExpr *E,
                                MCSymbolRefExpr::VariantKind Variant) {
  // Recurse over the given expression, rebuilding it to apply the given variant
  // to the leftmost symbol.
  if (Variant == MCSymbolRefExpr::VK_None)
    return E;

  switch (E->getKind()) {
  case MCExpr::Target:
    llvm_unreachable("Can't handle target expr yet");
  case MCExpr::Constant:
    llvm_unreachable("Can't handle lower16/upper16 of constant yet");

  case MCExpr::SymbolRef: {
    const MCSymbolRefExpr *SRE = cast<MCSymbolRefExpr>(E);

    if (SRE->getKind() != MCSymbolRefExpr::VK_None)
      return 0;

    return MCSymbolRefExpr::Create(&SRE->getSymbol(), Variant, getContext());
  }

  case MCExpr::Unary:
    llvm_unreachable("Can't handle unary expressions yet");

  case MCExpr::Binary: {
    const MCBinaryExpr *BE = cast<MCBinaryExpr>(E);
    const MCExpr *LHS = ApplyPrefixToExpr(BE->getLHS(), Variant);
    const MCExpr *RHS = BE->getRHS();
    if (!LHS)
      return 0;

    return MCBinaryExpr::Create(BE->getOpcode(), LHS, RHS, getContext());
  }
  }

  assert(0 && "Invalid expression kind!");
  return 0;
}

/// \brief Given a mnemonic, split out possible predication code and carry
/// setting letters to form a canonical mnemonic and flags.
//
// FIXME: Would be nice to autogen this.
static StringRef SplitMnemonicAndCC(StringRef Mnemonic,
                                    unsigned &PredicationCode,
                                    bool &CarrySetting) {
  PredicationCode = ARMCC::AL;
  CarrySetting = false;

  // Ignore some mnemonics we know aren't predicated forms.
  //
  // FIXME: Would be nice to autogen this.
  if (Mnemonic == "teq" || Mnemonic == "vceq" ||
      Mnemonic == "movs" ||
      Mnemonic == "svc" ||
      (Mnemonic == "mls" || Mnemonic == "smmls" || Mnemonic == "vcls" ||
       Mnemonic == "vmls" || Mnemonic == "vnmls") ||
      Mnemonic == "vacge" || Mnemonic == "vcge" ||
      Mnemonic == "vclt" ||
      Mnemonic == "vacgt" || Mnemonic == "vcgt" ||
      Mnemonic == "vcle" ||
      (Mnemonic == "smlal" || Mnemonic == "umaal" || Mnemonic == "umlal" ||
       Mnemonic == "vabal" || Mnemonic == "vmlal" || Mnemonic == "vpadal" ||
       Mnemonic == "vqdmlal"))
    return Mnemonic;

  // First, split out any predication code.
  unsigned CC = StringSwitch<unsigned>(Mnemonic.substr(Mnemonic.size()-2))
    .Case("eq", ARMCC::EQ)
    .Case("ne", ARMCC::NE)
    .Case("hs", ARMCC::HS)
    .Case("lo", ARMCC::LO)
    .Case("mi", ARMCC::MI)
    .Case("pl", ARMCC::PL)
    .Case("vs", ARMCC::VS)
    .Case("vc", ARMCC::VC)
    .Case("hi", ARMCC::HI)
    .Case("ls", ARMCC::LS)
    .Case("ge", ARMCC::GE)
    .Case("lt", ARMCC::LT)
    .Case("gt", ARMCC::GT)
    .Case("le", ARMCC::LE)
    .Case("al", ARMCC::AL)
    .Default(~0U);
  if (CC != ~0U) {
    Mnemonic = Mnemonic.slice(0, Mnemonic.size() - 2);
    PredicationCode = CC;
  }

  // Next, determine if we have a carry setting bit. We explicitly ignore all
  // the instructions we know end in 's'.
  if (Mnemonic.endswith("s") &&
      !(Mnemonic == "asrs" || Mnemonic == "cps" || Mnemonic == "mls" ||
        Mnemonic == "movs" || Mnemonic == "mrs" || Mnemonic == "smmls" ||
        Mnemonic == "vabs" || Mnemonic == "vcls" || Mnemonic == "vmls" ||
        Mnemonic == "vmrs" || Mnemonic == "vnmls" || Mnemonic == "vqabs" ||
        Mnemonic == "vrecps" || Mnemonic == "vrsqrts")) {
    Mnemonic = Mnemonic.slice(0, Mnemonic.size() - 1);
    CarrySetting = true;
  }

  return Mnemonic;
}

/// \brief Given a canonical mnemonic, determine if the instruction ever allows
/// inclusion of carry set or predication code operands.
//
// FIXME: It would be nice to autogen this.
static void GetMnemonicAcceptInfo(StringRef Mnemonic, bool &CanAcceptCarrySet,
                                  bool &CanAcceptPredicationCode) {
  if (Mnemonic == "and" || Mnemonic == "lsl" || Mnemonic == "lsr" ||
      Mnemonic == "rrx" || Mnemonic == "ror" || Mnemonic == "sub" ||
      Mnemonic == "smull" || Mnemonic == "add" || Mnemonic == "adc" ||
      Mnemonic == "mul" || Mnemonic == "bic" || Mnemonic == "asr" ||
      Mnemonic == "umlal" || Mnemonic == "orr" || Mnemonic == "mov" ||
      Mnemonic == "rsb" || Mnemonic == "rsc" || Mnemonic == "orn" ||
      Mnemonic == "sbc" || Mnemonic == "mla" || Mnemonic == "umull" ||
      Mnemonic == "eor" || Mnemonic == "smlal" || Mnemonic == "mvn") {
    CanAcceptCarrySet = true;
  } else {
    CanAcceptCarrySet = false;
  }

  if (Mnemonic == "cbnz" || Mnemonic == "setend" || Mnemonic == "dmb" ||
      Mnemonic == "cps" || Mnemonic == "mcr2" || Mnemonic == "it" ||
      Mnemonic == "mcrr2" || Mnemonic == "cbz" || Mnemonic == "cdp2" ||
      Mnemonic == "trap" || Mnemonic == "mrc2" || Mnemonic == "mrrc2" ||
      Mnemonic == "dsb" || Mnemonic == "movs") {
    CanAcceptPredicationCode = false;
  } else {
    CanAcceptPredicationCode = true;
  }
}

/// Parse an arm instruction mnemonic followed by its operands.
bool ARMAsmParser::ParseInstruction(StringRef Name, SMLoc NameLoc,
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  // Create the leading tokens for the mnemonic, split by '.' characters.
  size_t Start = 0, Next = Name.find('.');
  StringRef Head = Name.slice(Start, Next);

  // Split out the predication code and carry setting flag from the mnemonic.
  unsigned PredicationCode;
  bool CarrySetting;
  Head = SplitMnemonicAndCC(Head, PredicationCode, CarrySetting);

  Operands.push_back(ARMOperand::CreateToken(Head, NameLoc));

  // Next, add the CCOut and ConditionCode operands, if needed.
  //
  // For mnemonics which can ever incorporate a carry setting bit or predication
  // code, our matching model involves us always generating CCOut and
  // ConditionCode operands to match the mnemonic "as written" and then we let
  // the matcher deal with finding the right instruction or generating an
  // appropriate error.
  bool CanAcceptCarrySet, CanAcceptPredicationCode;
  GetMnemonicAcceptInfo(Head, CanAcceptCarrySet, CanAcceptPredicationCode);

  // Add the carry setting operand, if necessary.
  //
  // FIXME: It would be awesome if we could somehow invent a location such that
  // match errors on this operand would print a nice diagnostic about how the
  // 's' character in the mnemonic resulted in a CCOut operand.
  if (CanAcceptCarrySet) {
    Operands.push_back(ARMOperand::CreateCCOut(CarrySetting ? ARM::CPSR : 0,
                                               NameLoc));
  } else {
    // This mnemonic can't ever accept a carry set, but the user wrote one (or
    // misspelled another mnemonic).

    // FIXME: Issue a nice error.
  }

  // Add the predication code operand, if necessary.
  if (CanAcceptPredicationCode) {
    Operands.push_back(ARMOperand::CreateCondCode(
                         ARMCC::CondCodes(PredicationCode), NameLoc));
  } else {
    // This mnemonic can't ever accept a predication code, but the user wrote
    // one (or misspelled another mnemonic).

    // FIXME: Issue a nice error.
  }

  // Add the remaining tokens in the mnemonic.
  while (Next != StringRef::npos) {
    Start = Next;
    Next = Name.find('.', Start + 1);
    Head = Name.slice(Start, Next);

    Operands.push_back(ARMOperand::CreateToken(Head, NameLoc));
  }

  bool isMCR = (Head == "mcr"  || Head == "mcr2" ||
                Head == "mcrr" || Head == "mcrr2" ||
                Head == "mrc"  || Head == "mrc2" ||
                Head == "mrrc" || Head == "mrrc2");

  // Read the remaining operands.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    // Read the first operand.
    if (ParseOperand(Operands, isMCR)) {
      Parser.EatToEndOfStatement();
      return true;
    }

    while (getLexer().is(AsmToken::Comma)) {
      Parser.Lex();  // Eat the comma.

      // Parse and remember the operand.
      if (ParseOperand(Operands, isMCR)) {
        Parser.EatToEndOfStatement();
        return true;
      }
    }
  }

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    Parser.EatToEndOfStatement();
    return TokError("unexpected token in argument list");
  }

  Parser.Lex(); // Consume the EndOfStatement
  return false;
}

bool ARMAsmParser::
MatchAndEmitInstruction(SMLoc IDLoc,
                        SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                        MCStreamer &Out) {
  MCInst Inst;
  unsigned ErrorInfo;
  MatchResultTy MatchResult, MatchResult2;
  MatchResult = MatchInstructionImpl(Operands, Inst, ErrorInfo);
  if (MatchResult != Match_Success) {
    // If we get a Match_InvalidOperand it might be some arithmetic instruction
    // that does not update the condition codes.  So try adding a CCOut operand
    // with a value of reg0.
    if (MatchResult == Match_InvalidOperand) {
      Operands.insert(Operands.begin() + 1,
                      ARMOperand::CreateCCOut(0,
                                  ((ARMOperand*)Operands[0])->getStartLoc()));
      MatchResult2 = MatchInstructionImpl(Operands, Inst, ErrorInfo);
      if (MatchResult2 == Match_Success)
        MatchResult = Match_Success;
      else {
        ARMOperand *CCOut = ((ARMOperand*)Operands[1]);
        Operands.erase(Operands.begin() + 1);
        delete CCOut;
      }
    }
    // If we get a Match_MnemonicFail it might be some arithmetic instruction
    // that updates the condition codes if it ends in 's'.  So see if the
    // mnemonic ends in 's' and if so try removing the 's' and adding a CCOut
    // operand with a value of CPSR.
    else if(MatchResult == Match_MnemonicFail) {
      // Get the instruction mnemonic, which is the first token.
      StringRef Mnemonic = ((ARMOperand*)Operands[0])->getToken();
      if (Mnemonic.substr(Mnemonic.size()-1) == "s") {
        // removed the 's' from the mnemonic for matching.
        StringRef MnemonicNoS = Mnemonic.slice(0, Mnemonic.size() - 1);
        SMLoc NameLoc = ((ARMOperand*)Operands[0])->getStartLoc();
        ARMOperand *OldMnemonic = ((ARMOperand*)Operands[0]);
        Operands.erase(Operands.begin());
        delete OldMnemonic;
        Operands.insert(Operands.begin(),
                        ARMOperand::CreateToken(MnemonicNoS, NameLoc));
        Operands.insert(Operands.begin() + 1,
                        ARMOperand::CreateCCOut(ARM::CPSR, NameLoc));
        MatchResult2 = MatchInstructionImpl(Operands, Inst, ErrorInfo);
        if (MatchResult2 == Match_Success)
          MatchResult = Match_Success;
        else {
          ARMOperand *OldMnemonic = ((ARMOperand*)Operands[0]);
          Operands.erase(Operands.begin());
          delete OldMnemonic;
          Operands.insert(Operands.begin(),
                          ARMOperand::CreateToken(Mnemonic, NameLoc));
          ARMOperand *CCOut = ((ARMOperand*)Operands[1]);
          Operands.erase(Operands.begin() + 1);
          delete CCOut;
        }
      }
    }
  }
  switch (MatchResult) {
  case Match_Success:
    Out.EmitInstruction(Inst);
    return false;
  case Match_MissingFeature:
    Error(IDLoc, "instruction requires a CPU feature not currently enabled");
    return true;
  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0U) {
      if (ErrorInfo >= Operands.size())
        return Error(IDLoc, "too few operands for instruction");

      ErrorLoc = ((ARMOperand*)Operands[ErrorInfo])->getStartLoc();
      if (ErrorLoc == SMLoc()) ErrorLoc = IDLoc;
    }

    return Error(ErrorLoc, "invalid operand for instruction");
  }
  case Match_MnemonicFail:
    return Error(IDLoc, "unrecognized instruction mnemonic");
  }

  llvm_unreachable("Implement any new match types added!");
  return true;
}

/// ParseDirective parses the arm specific directives
bool ARMAsmParser::ParseDirective(AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getIdentifier();
  if (IDVal == ".word")
    return ParseDirectiveWord(4, DirectiveID.getLoc());
  else if (IDVal == ".thumb")
    return ParseDirectiveThumb(DirectiveID.getLoc());
  else if (IDVal == ".thumb_func")
    return ParseDirectiveThumbFunc(DirectiveID.getLoc());
  else if (IDVal == ".code")
    return ParseDirectiveCode(DirectiveID.getLoc());
  else if (IDVal == ".syntax")
    return ParseDirectiveSyntax(DirectiveID.getLoc());
  return true;
}

/// ParseDirectiveWord
///  ::= .word [ expression (, expression)* ]
bool ARMAsmParser::ParseDirectiveWord(unsigned Size, SMLoc L) {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      const MCExpr *Value;
      if (getParser().ParseExpression(Value))
        return true;

      getParser().getStreamer().EmitValue(Value, Size, 0/*addrspace*/);

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

/// ParseDirectiveThumb
///  ::= .thumb
bool ARMAsmParser::ParseDirectiveThumb(SMLoc L) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return Error(L, "unexpected token in directive");
  Parser.Lex();

  // TODO: set thumb mode
  // TODO: tell the MC streamer the mode
  // getParser().getStreamer().Emit???();
  return false;
}

/// ParseDirectiveThumbFunc
///  ::= .thumbfunc symbol_name
bool ARMAsmParser::ParseDirectiveThumbFunc(SMLoc L) {
  const AsmToken &Tok = Parser.getTok();
  if (Tok.isNot(AsmToken::Identifier) && Tok.isNot(AsmToken::String))
    return Error(L, "unexpected token in .thumb_func directive");
  StringRef Name = Tok.getString();
  Parser.Lex(); // Consume the identifier token.
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return Error(L, "unexpected token in directive");
  Parser.Lex();

  // Mark symbol as a thumb symbol.
  MCSymbol *Func = getParser().getContext().GetOrCreateSymbol(Name);
  getParser().getStreamer().EmitThumbFunc(Func);
  return false;
}

/// ParseDirectiveSyntax
///  ::= .syntax unified | divided
bool ARMAsmParser::ParseDirectiveSyntax(SMLoc L) {
  const AsmToken &Tok = Parser.getTok();
  if (Tok.isNot(AsmToken::Identifier))
    return Error(L, "unexpected token in .syntax directive");
  StringRef Mode = Tok.getString();
  if (Mode == "unified" || Mode == "UNIFIED")
    Parser.Lex();
  else if (Mode == "divided" || Mode == "DIVIDED")
    Parser.Lex();
  else
    return Error(L, "unrecognized syntax mode in .syntax directive");

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return Error(Parser.getTok().getLoc(), "unexpected token in directive");
  Parser.Lex();

  // TODO tell the MC streamer the mode
  // getParser().getStreamer().Emit???();
  return false;
}

/// ParseDirectiveCode
///  ::= .code 16 | 32
bool ARMAsmParser::ParseDirectiveCode(SMLoc L) {
  const AsmToken &Tok = Parser.getTok();
  if (Tok.isNot(AsmToken::Integer))
    return Error(L, "unexpected token in .code directive");
  int64_t Val = Parser.getTok().getIntVal();
  if (Val == 16)
    Parser.Lex();
  else if (Val == 32)
    Parser.Lex();
  else
    return Error(L, "invalid operand to .code directive");

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return Error(Parser.getTok().getLoc(), "unexpected token in directive");
  Parser.Lex();

  // FIXME: We need to be able switch subtargets at this point so that
  // MatchInstructionImpl() will work when it gets the AvailableFeatures which
  // includes Feature_IsThumb or not to match the right instructions.  This is
  // blocked on the FIXME in llvm-mc.cpp when creating the TargetMachine.
  if (Val == 16){
    assert(TM.getSubtarget<ARMSubtarget>().isThumb() &&
	   "switching between arm/thumb not yet suppported via .code 16)");
    getParser().getStreamer().EmitAssemblerFlag(MCAF_Code16);
  }
  else{
    assert(!TM.getSubtarget<ARMSubtarget>().isThumb() &&
           "switching between thumb/arm not yet suppported via .code 32)");
    getParser().getStreamer().EmitAssemblerFlag(MCAF_Code32);
   }

  return false;
}

extern "C" void LLVMInitializeARMAsmLexer();

/// Force static initialization.
extern "C" void LLVMInitializeARMAsmParser() {
  RegisterAsmParser<ARMAsmParser> X(TheARMTarget);
  RegisterAsmParser<ARMAsmParser> Y(TheThumbTarget);
  LLVMInitializeARMAsmLexer();
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "ARMGenAsmMatcher.inc"
