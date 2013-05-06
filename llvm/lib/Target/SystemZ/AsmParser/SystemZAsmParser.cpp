//===-- SystemZAsmParser.cpp - Parse SystemZ assembly instructions --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SystemZMCTargetDesc.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

// Return true if Expr is in the range [MinValue, MaxValue].
static bool inRange(const MCExpr *Expr, int64_t MinValue, int64_t MaxValue) {
  if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr)) {
    int64_t Value = CE->getValue();
    return Value >= MinValue && Value <= MaxValue;
  }
  return false;
}

namespace {
class SystemZOperand : public MCParsedAsmOperand {
public:
  enum RegisterKind {
    GR32Reg,
    GR64Reg,
    GR128Reg,
    ADDR32Reg,
    ADDR64Reg,
    FP32Reg,
    FP64Reg,
    FP128Reg
  };

private:
  enum OperandKind {
    KindToken,
    KindReg,
    KindAccessReg,
    KindImm,
    KindMem
  };

  OperandKind Kind;
  SMLoc StartLoc, EndLoc;

  // A string of length Length, starting at Data.
  struct TokenOp {
    const char *Data;
    unsigned Length;
  };

  // LLVM register Num, which has kind Kind.
  struct RegOp {
    RegisterKind Kind;
    unsigned Num;
  };

  // Base + Disp + Index, where Base and Index are LLVM registers or 0.
  // RegKind says what type the registers have (ADDR32Reg or ADDR64Reg).
  struct MemOp {
    unsigned Base : 8;
    unsigned Index : 8;
    unsigned RegKind : 8;
    unsigned Unused : 8;
    const MCExpr *Disp;
  };

  union {
    TokenOp Token;
    RegOp Reg;
    unsigned AccessReg;
    const MCExpr *Imm;
    MemOp Mem;
  };

  SystemZOperand(OperandKind kind, SMLoc startLoc, SMLoc endLoc)
    : Kind(kind), StartLoc(startLoc), EndLoc(endLoc)
  {}

  void addExpr(MCInst &Inst, const MCExpr *Expr) const {
    // Add as immediates when possible.  Null MCExpr = 0.
    if (Expr == 0)
      Inst.addOperand(MCOperand::CreateImm(0));
    else if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::CreateImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::CreateExpr(Expr));
  }

public:
  // Create particular kinds of operand.
  static SystemZOperand *createToken(StringRef Str, SMLoc Loc) {
    SystemZOperand *Op = new SystemZOperand(KindToken, Loc, Loc);
    Op->Token.Data = Str.data();
    Op->Token.Length = Str.size();
    return Op;
  }
  static SystemZOperand *createReg(RegisterKind Kind, unsigned Num,
                                   SMLoc StartLoc, SMLoc EndLoc) {
    SystemZOperand *Op = new SystemZOperand(KindReg, StartLoc, EndLoc);
    Op->Reg.Kind = Kind;
    Op->Reg.Num = Num;
    return Op;
  }
  static SystemZOperand *createAccessReg(unsigned Num, SMLoc StartLoc,
                                         SMLoc EndLoc) {
    SystemZOperand *Op = new SystemZOperand(KindAccessReg, StartLoc, EndLoc);
    Op->AccessReg = Num;
    return Op;
  }
  static SystemZOperand *createImm(const MCExpr *Expr, SMLoc StartLoc,
                                   SMLoc EndLoc) {
    SystemZOperand *Op = new SystemZOperand(KindImm, StartLoc, EndLoc);
    Op->Imm = Expr;
    return Op;
  }
  static SystemZOperand *createMem(RegisterKind RegKind, unsigned Base,
                                   const MCExpr *Disp, unsigned Index,
                                   SMLoc StartLoc, SMLoc EndLoc) {
    SystemZOperand *Op = new SystemZOperand(KindMem, StartLoc, EndLoc);
    Op->Mem.RegKind = RegKind;
    Op->Mem.Base = Base;
    Op->Mem.Index = Index;
    Op->Mem.Disp = Disp;
    return Op;
  }

  // Token operands
  virtual bool isToken() const LLVM_OVERRIDE {
    return Kind == KindToken;
  }
  StringRef getToken() const {
    assert(Kind == KindToken && "Not a token");
    return StringRef(Token.Data, Token.Length);
  }

  // Register operands.
  virtual bool isReg() const LLVM_OVERRIDE {
    return Kind == KindReg;
  }
  bool isReg(RegisterKind RegKind) const {
    return Kind == KindReg && Reg.Kind == RegKind;
  }
  virtual unsigned getReg() const LLVM_OVERRIDE {
    assert(Kind == KindReg && "Not a register");
    return Reg.Num;
  }

  // Access register operands.  Access registers aren't exposed to LLVM
  // as registers.
  bool isAccessReg() const {
    return Kind == KindAccessReg;
  }

  // Immediate operands.
  virtual bool isImm() const LLVM_OVERRIDE {
    return Kind == KindImm;
  }
  bool isImm(int64_t MinValue, int64_t MaxValue) const {
    return Kind == KindImm && inRange(Imm, MinValue, MaxValue);
  }
  const MCExpr *getImm() const {
    assert(Kind == KindImm && "Not an immediate");
    return Imm;
  }

  // Memory operands.
  virtual bool isMem() const LLVM_OVERRIDE {
    return Kind == KindMem;
  }
  bool isMem(RegisterKind RegKind, bool HasIndex) const {
    return (Kind == KindMem &&
            Mem.RegKind == RegKind &&
            (HasIndex || !Mem.Index));
  }
  bool isMemDisp12(RegisterKind RegKind, bool HasIndex) const {
    return isMem(RegKind, HasIndex) && inRange(Mem.Disp, 0, 0xfff);
  }
  bool isMemDisp20(RegisterKind RegKind, bool HasIndex) const {
    return isMem(RegKind, HasIndex) && inRange(Mem.Disp, -524288, 524287);
  }

  // Override MCParsedAsmOperand.
  virtual SMLoc getStartLoc() const LLVM_OVERRIDE { return StartLoc; }
  virtual SMLoc getEndLoc() const LLVM_OVERRIDE { return EndLoc; }
  virtual void print(raw_ostream &OS) const LLVM_OVERRIDE;

  // Used by the TableGen code to add particular types of operand
  // to an instruction.
  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands");
    Inst.addOperand(MCOperand::CreateReg(getReg()));
  }
  void addAccessRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands");
    assert(Kind == KindAccessReg && "Invalid operand type");
    Inst.addOperand(MCOperand::CreateImm(AccessReg));
  }
  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands");
    addExpr(Inst, getImm());
  }
  void addBDAddrOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands");
    assert(Kind == KindMem && Mem.Index == 0 && "Invalid operand type");
    Inst.addOperand(MCOperand::CreateReg(Mem.Base));
    addExpr(Inst, Mem.Disp);
  }
  void addBDXAddrOperands(MCInst &Inst, unsigned N) const {
    assert(N == 3 && "Invalid number of operands");
    assert(Kind == KindMem && "Invalid operand type");
    Inst.addOperand(MCOperand::CreateReg(Mem.Base));
    addExpr(Inst, Mem.Disp);
    Inst.addOperand(MCOperand::CreateReg(Mem.Index));
  }

  // Used by the TableGen code to check for particular operand types.
  bool isGR32() const { return isReg(GR32Reg); }
  bool isGR64() const { return isReg(GR64Reg); }
  bool isGR128() const { return isReg(GR128Reg); }
  bool isADDR32() const { return isReg(ADDR32Reg); }
  bool isADDR64() const { return isReg(ADDR64Reg); }
  bool isADDR128() const { return false; }
  bool isFP32() const { return isReg(FP32Reg); }
  bool isFP64() const { return isReg(FP64Reg); }
  bool isFP128() const { return isReg(FP128Reg); }
  bool isBDAddr32Disp12() const { return isMemDisp12(ADDR32Reg, false); }
  bool isBDAddr32Disp20() const { return isMemDisp20(ADDR32Reg, false); }
  bool isBDAddr64Disp12() const { return isMemDisp12(ADDR64Reg, false); }
  bool isBDAddr64Disp20() const { return isMemDisp20(ADDR64Reg, false); }
  bool isBDXAddr64Disp12() const { return isMemDisp12(ADDR64Reg, true); }
  bool isBDXAddr64Disp20() const { return isMemDisp20(ADDR64Reg, true); }
  bool isU4Imm() const { return isImm(0, 15); }
  bool isU6Imm() const { return isImm(0, 63); }
  bool isU8Imm() const { return isImm(0, 255); }
  bool isS8Imm() const { return isImm(-128, 127); }
  bool isU16Imm() const { return isImm(0, 65535); }
  bool isS16Imm() const { return isImm(-32768, 32767); }
  bool isU32Imm() const { return isImm(0, (1LL << 32) - 1); }
  bool isS32Imm() const { return isImm(-(1LL << 31), (1LL << 31) - 1); }
};

// Maps of asm register numbers to LLVM register numbers, with 0 indicating
// an invalid register.  We don't use register class directly because that
// specifies the allocation order.
static const unsigned GR32Regs[] = {
  SystemZ::R0W, SystemZ::R1W, SystemZ::R2W, SystemZ::R3W,
  SystemZ::R4W, SystemZ::R5W, SystemZ::R6W, SystemZ::R7W,
  SystemZ::R8W, SystemZ::R9W, SystemZ::R10W, SystemZ::R11W,
  SystemZ::R12W, SystemZ::R13W, SystemZ::R14W, SystemZ::R15W
};
static const unsigned GR64Regs[] = {
  SystemZ::R0D, SystemZ::R1D, SystemZ::R2D, SystemZ::R3D,
  SystemZ::R4D, SystemZ::R5D, SystemZ::R6D, SystemZ::R7D,
  SystemZ::R8D, SystemZ::R9D, SystemZ::R10D, SystemZ::R11D,
  SystemZ::R12D, SystemZ::R13D, SystemZ::R14D, SystemZ::R15D
};
static const unsigned GR128Regs[] = {
  SystemZ::R0Q, 0, SystemZ::R2Q, 0,
  SystemZ::R4Q, 0, SystemZ::R6Q, 0,
  SystemZ::R8Q, 0, SystemZ::R10Q, 0,
  SystemZ::R12Q, 0, SystemZ::R14Q, 0
};
static const unsigned FP32Regs[] = {
  SystemZ::F0S, SystemZ::F1S, SystemZ::F2S, SystemZ::F3S,
  SystemZ::F4S, SystemZ::F5S, SystemZ::F6S, SystemZ::F7S,
  SystemZ::F8S, SystemZ::F9S, SystemZ::F10S, SystemZ::F11S,
  SystemZ::F12S, SystemZ::F13S, SystemZ::F14S, SystemZ::F15S
};
static const unsigned FP64Regs[] = {
  SystemZ::F0D, SystemZ::F1D, SystemZ::F2D, SystemZ::F3D,
  SystemZ::F4D, SystemZ::F5D, SystemZ::F6D, SystemZ::F7D,
  SystemZ::F8D, SystemZ::F9D, SystemZ::F10D, SystemZ::F11D,
  SystemZ::F12D, SystemZ::F13D, SystemZ::F14D, SystemZ::F15D
};
static const unsigned FP128Regs[] = {
  SystemZ::F0Q, SystemZ::F1Q, 0, 0,
  SystemZ::F4Q, SystemZ::F5Q, 0, 0,
  SystemZ::F8Q, SystemZ::F9Q, 0, 0,
  SystemZ::F12Q, SystemZ::F13Q, 0, 0
};

class SystemZAsmParser : public MCTargetAsmParser {
#define GET_ASSEMBLER_HEADER
#include "SystemZGenAsmMatcher.inc"

private:
  MCSubtargetInfo &STI;
  MCAsmParser &Parser;
  struct Register {
    char Prefix;
    unsigned Number;
    SMLoc StartLoc, EndLoc;
  };

  bool parseRegister(Register &Reg);

  OperandMatchResultTy
  parseRegister(Register &Reg, char Prefix, const unsigned *Regs,
                bool IsAddress = false);

  OperandMatchResultTy
  parseRegister(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                char Prefix, const unsigned *Regs,
                SystemZOperand::RegisterKind Kind,
                bool IsAddress = false);

  OperandMatchResultTy
  parseAddress(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
               const unsigned *Regs, SystemZOperand::RegisterKind RegKind,
               bool HasIndex);

  bool parseOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                    StringRef Mnemonic);

public:
  SystemZAsmParser(MCSubtargetInfo &sti, MCAsmParser &parser)
    : MCTargetAsmParser(), STI(sti), Parser(parser) {
    MCAsmParserExtension::Initialize(Parser);

    // Initialize the set of available features.
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }

  // Override MCTargetAsmParser.
  virtual bool ParseDirective(AsmToken DirectiveID) LLVM_OVERRIDE;
  virtual bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc,
                             SMLoc &EndLoc) LLVM_OVERRIDE;
  virtual bool ParseInstruction(ParseInstructionInfo &Info,
                                StringRef Name, SMLoc NameLoc,
                                SmallVectorImpl<MCParsedAsmOperand*> &Operands)
    LLVM_OVERRIDE;
  virtual bool
    MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                            SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                            MCStreamer &Out, unsigned &ErrorInfo,
                            bool MatchingInlineAsm) LLVM_OVERRIDE;

  // Used by the TableGen code to parse particular operand types.
  OperandMatchResultTy
  parseGR32(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
    return parseRegister(Operands, 'r', GR32Regs, SystemZOperand::GR32Reg);
  }
  OperandMatchResultTy
  parseGR64(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
    return parseRegister(Operands, 'r', GR64Regs, SystemZOperand::GR64Reg);
  }
  OperandMatchResultTy
  parseGR128(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
    return parseRegister(Operands, 'r', GR128Regs, SystemZOperand::GR128Reg);
  }
  OperandMatchResultTy
  parseADDR32(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
    return parseRegister(Operands, 'r', GR32Regs, SystemZOperand::ADDR32Reg,
                         true);
  }
  OperandMatchResultTy
  parseADDR64(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
    return parseRegister(Operands, 'r', GR64Regs, SystemZOperand::ADDR64Reg,
                         true);
  }
  OperandMatchResultTy
  parseADDR128(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
    llvm_unreachable("Shouldn't be used as an operand");
  }
  OperandMatchResultTy
  parseFP32(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
    return parseRegister(Operands, 'f', FP32Regs, SystemZOperand::FP32Reg);
  }
  OperandMatchResultTy
  parseFP64(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
    return parseRegister(Operands, 'f', FP64Regs, SystemZOperand::FP64Reg);
  }
  OperandMatchResultTy
  parseFP128(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
    return parseRegister(Operands, 'f', FP128Regs, SystemZOperand::FP128Reg);
  }
  OperandMatchResultTy
  parseBDAddr32(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
    return parseAddress(Operands, GR32Regs, SystemZOperand::ADDR32Reg, false);
  }
  OperandMatchResultTy
  parseBDAddr64(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
    return parseAddress(Operands, GR64Regs, SystemZOperand::ADDR64Reg, false);
  }
  OperandMatchResultTy
  parseBDXAddr64(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
    return parseAddress(Operands, GR64Regs, SystemZOperand::ADDR64Reg, true);
  }
  OperandMatchResultTy
  parseAccessReg(SmallVectorImpl<MCParsedAsmOperand*> &Operands);
};
}

#define GET_REGISTER_MATCHER
#define GET_SUBTARGET_FEATURE_NAME
#define GET_MATCHER_IMPLEMENTATION
#include "SystemZGenAsmMatcher.inc"

void SystemZOperand::print(raw_ostream &OS) const {
  llvm_unreachable("Not implemented");
}

// Parse one register of the form %<prefix><number>.
bool SystemZAsmParser::parseRegister(Register &Reg) {
  Reg.StartLoc = Parser.getTok().getLoc();

  // Eat the % prefix.
  if (Parser.getTok().isNot(AsmToken::Percent))
    return true;
  Parser.Lex();

  // Expect a register name.
  if (Parser.getTok().isNot(AsmToken::Identifier))
    return true;

  // Check the prefix.
  StringRef Name = Parser.getTok().getString();
  if (Name.size() < 2)
    return true;
  Reg.Prefix = Name[0];

  // Treat the rest of the register name as a register number.
  if (Name.substr(1).getAsInteger(10, Reg.Number))
    return true;

  Reg.EndLoc = Parser.getTok().getLoc();
  Parser.Lex();
  return false;
}

// Parse a register with prefix Prefix and convert it to LLVM numbering.
// Regs maps asm register numbers to LLVM register numbers, with zero
// entries indicating an invalid register.  IsAddress says whether the
// register appears in an address context.
SystemZAsmParser::OperandMatchResultTy
SystemZAsmParser::parseRegister(Register &Reg, char Prefix,
                                const unsigned *Regs, bool IsAddress) {
  if (parseRegister(Reg))
    return MatchOperand_NoMatch;
  if (Reg.Prefix != Prefix || Reg.Number > 15 || Regs[Reg.Number] == 0) {
    Error(Reg.StartLoc, "invalid register");
    return MatchOperand_ParseFail;
  }
  if (Reg.Number == 0 && IsAddress) {
    Error(Reg.StartLoc, "%r0 used in an address");
    return MatchOperand_ParseFail;
  }
  Reg.Number = Regs[Reg.Number];
  return MatchOperand_Success;
}

// Parse a register and add it to Operands.  Prefix is 'r' for GPRs,
// 'f' for FPRs, etc.  Regs maps asm register numbers to LLVM register numbers,
// with zero entries indicating an invalid register.  Kind is the type of
// register represented by Regs and IsAddress says whether the register is
// being parsed in an address context, meaning that %r0 evaluates as 0.
SystemZAsmParser::OperandMatchResultTy
SystemZAsmParser::parseRegister(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                                char Prefix, const unsigned *Regs,
                                SystemZOperand::RegisterKind Kind,
                                bool IsAddress) {
  Register Reg;
  OperandMatchResultTy Result = parseRegister(Reg, Prefix, Regs, IsAddress);
  if (Result == MatchOperand_Success)
    Operands.push_back(SystemZOperand::createReg(Kind, Reg.Number,
                                                 Reg.StartLoc, Reg.EndLoc));
  return Result;
}

// Parse a memory operand and add it to Operands.  Regs maps asm register
// numbers to LLVM address registers and RegKind says what kind of address
// register we're using (ADDR32Reg or ADDR64Reg).  HasIndex says whether
// the address allows index registers.
SystemZAsmParser::OperandMatchResultTy
SystemZAsmParser::parseAddress(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                               const unsigned *Regs,
                               SystemZOperand::RegisterKind RegKind,
                               bool HasIndex) {
  SMLoc StartLoc = Parser.getTok().getLoc();

  // Parse the displacement, which must always be present.
  const MCExpr *Disp;
  if (getParser().parseExpression(Disp))
    return MatchOperand_NoMatch;

  // Parse the optional base and index.
  unsigned Index = 0;
  unsigned Base = 0;
  if (getLexer().is(AsmToken::LParen)) {
    Parser.Lex();

    // Parse the first register.
    Register Reg;
    OperandMatchResultTy Result = parseRegister(Reg, 'r', GR64Regs, true);
    if (Result != MatchOperand_Success)
      return Result;

    // Check whether there's a second register.  If so, the one that we
    // just parsed was the index.
    if (getLexer().is(AsmToken::Comma)) {
      Parser.Lex();

      if (!HasIndex) {
        Error(Reg.StartLoc, "invalid use of indexed addressing");
        return MatchOperand_ParseFail;
      }

      Index = Reg.Number;
      Result = parseRegister(Reg, 'r', GR64Regs, true);
      if (Result != MatchOperand_Success)
        return Result;
    }
    Base = Reg.Number;

    // Consume the closing bracket.
    if (getLexer().isNot(AsmToken::RParen))
      return MatchOperand_NoMatch;
    Parser.Lex();
  }

  SMLoc EndLoc =
    SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
  Operands.push_back(SystemZOperand::createMem(RegKind, Base, Disp, Index,
                                               StartLoc, EndLoc));
  return MatchOperand_Success;
}

bool SystemZAsmParser::ParseDirective(AsmToken DirectiveID) {
  return true;
}

bool SystemZAsmParser::ParseRegister(unsigned &RegNo, SMLoc &StartLoc,
                                     SMLoc &EndLoc) {
  Register Reg;
  if (parseRegister(Reg))
    return Error(Reg.StartLoc, "register expected");
  if (Reg.Prefix == 'r' && Reg.Number < 16)
    RegNo = GR64Regs[Reg.Number];
  else if (Reg.Prefix == 'f' && Reg.Number < 16)
    RegNo = FP64Regs[Reg.Number];
  else
    return Error(Reg.StartLoc, "invalid register");
  StartLoc = Reg.StartLoc;
  EndLoc = Reg.EndLoc;
  return false;
}

bool SystemZAsmParser::
ParseInstruction(ParseInstructionInfo &Info, StringRef Name, SMLoc NameLoc,
                 SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  Operands.push_back(SystemZOperand::createToken(Name, NameLoc));

  // Read the remaining operands.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    // Read the first operand.
    if (parseOperand(Operands, Name)) {
      Parser.eatToEndOfStatement();
      return true;
    }

    // Read any subsequent operands.
    while (getLexer().is(AsmToken::Comma)) {
      Parser.Lex();
      if (parseOperand(Operands, Name)) {
        Parser.eatToEndOfStatement();
        return true;
      }
    }
    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      SMLoc Loc = getLexer().getLoc();
      Parser.eatToEndOfStatement();
      return Error(Loc, "unexpected token in argument list");
    }
  }

  // Consume the EndOfStatement.
  Parser.Lex();
  return false;
}

bool SystemZAsmParser::
parseOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
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

  // The only other type of operand is an immediate.
  const MCExpr *Expr;
  SMLoc StartLoc = Parser.getTok().getLoc();
  if (getParser().parseExpression(Expr))
    return true;

  SMLoc EndLoc =
    SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
  Operands.push_back(SystemZOperand::createImm(Expr, StartLoc, EndLoc));
  return false;
}

bool SystemZAsmParser::
MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                        SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                        MCStreamer &Out, unsigned &ErrorInfo,
                        bool MatchingInlineAsm) {
  MCInst Inst;
  unsigned MatchResult;

  MatchResult = MatchInstructionImpl(Operands, Inst, ErrorInfo,
                                     MatchingInlineAsm);
  switch (MatchResult) {
  default: break;
  case Match_Success:
    Inst.setLoc(IDLoc);
    Out.EmitInstruction(Inst);
    return false;

  case Match_MissingFeature: {
    assert(ErrorInfo && "Unknown missing feature!");
    // Special case the error message for the very common case where only
    // a single subtarget feature is missing
    std::string Msg = "instruction requires:";
    unsigned Mask = 1;
    for (unsigned I = 0; I < sizeof(ErrorInfo) * 8 - 1; ++I) {
      if (ErrorInfo & Mask) {
        Msg += " ";
        Msg += getSubtargetFeatureName(ErrorInfo & Mask);
      }
      Mask <<= 1;
    }
    return Error(IDLoc, Msg);
  }

  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0U) {
      if (ErrorInfo >= Operands.size())
        return Error(IDLoc, "too few operands for instruction");

      ErrorLoc = ((SystemZOperand*)Operands[ErrorInfo])->getStartLoc();
      if (ErrorLoc == SMLoc())
        ErrorLoc = IDLoc;
    }
    return Error(ErrorLoc, "invalid operand for instruction");
  }

  case Match_MnemonicFail:
    return Error(IDLoc, "invalid instruction");
  }

  llvm_unreachable("Unexpected match type");
}

SystemZAsmParser::OperandMatchResultTy SystemZAsmParser::
parseAccessReg(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  Register Reg;
  if (parseRegister(Reg))
    return MatchOperand_NoMatch;
  if (Reg.Prefix != 'a' || Reg.Number > 15) {
    Error(Reg.StartLoc, "invalid register");
    return MatchOperand_ParseFail;
  }
  Operands.push_back(SystemZOperand::createAccessReg(Reg.Number,
                                                     Reg.StartLoc, Reg.EndLoc));
  return MatchOperand_Success;
}

// Force static initialization.
extern "C" void LLVMInitializeSystemZAsmParser() {
  RegisterMCAsmParser<SystemZAsmParser> X(TheSystemZTarget);
}
