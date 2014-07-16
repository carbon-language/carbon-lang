//===--- RuntimeDyldChecker.cpp - RuntimeDyld tester framework --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/RuntimeDyldChecker.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/StringRefMemoryObject.h"
#include "RuntimeDyldImpl.h"
#include <cctype>
#include <memory>

#define DEBUG_TYPE "rtdyld"

using namespace llvm;

namespace llvm {

  // Helper class that implements the language evaluated by RuntimeDyldChecker.
  class RuntimeDyldCheckerExprEval {
  public:

    RuntimeDyldCheckerExprEval(const RuntimeDyldChecker &Checker,
                               llvm::raw_ostream &ErrStream)
      : Checker(Checker), ErrStream(ErrStream) {}

    bool evaluate(StringRef Expr) const {
      // Expect equality expression of the form 'LHS = RHS'.
      Expr = Expr.trim();
      size_t EQIdx = Expr.find('=');

      // Evaluate LHS.
      StringRef LHSExpr = Expr.substr(0, EQIdx).rtrim();
      StringRef RemainingExpr;
      EvalResult LHSResult;
      std::tie(LHSResult, RemainingExpr) =
        evalComplexExpr(evalSimpleExpr(LHSExpr));
      if (LHSResult.hasError())
        return handleError(Expr, LHSResult);
      if (RemainingExpr != "")
        return handleError(Expr, unexpectedToken(RemainingExpr, LHSExpr, ""));

      // Evaluate RHS.
      StringRef RHSExpr = Expr.substr(EQIdx + 1).ltrim();
      EvalResult RHSResult;
      std::tie(RHSResult, RemainingExpr) =
        evalComplexExpr(evalSimpleExpr(RHSExpr));
      if (RHSResult.hasError())
        return handleError(Expr, RHSResult);
      if (RemainingExpr != "")
        return handleError(Expr, unexpectedToken(RemainingExpr, RHSExpr, ""));

      if (LHSResult.getValue() != RHSResult.getValue()) {
        ErrStream << "Expression '" << Expr << "' is false: "
                  << format("0x%lx", LHSResult.getValue()) << " != "
                  << format("0x%lx", RHSResult.getValue()) << "\n";
        return false;
      }
      return true;
    }

  private:
    const RuntimeDyldChecker &Checker;
    llvm::raw_ostream &ErrStream;

    enum class BinOpToken : unsigned { Invalid, Add, Sub, BitwiseAnd,
                                       BitwiseOr, ShiftLeft, ShiftRight };

    class EvalResult {
    public:
      EvalResult()
        : Value(0), ErrorMsg("") {}
      EvalResult(uint64_t Value)
        : Value(Value), ErrorMsg("") {}
      EvalResult(std::string ErrorMsg)
        : Value(0), ErrorMsg(ErrorMsg) {}
      uint64_t getValue() const { return Value; }
      bool hasError() const { return ErrorMsg != ""; }
      const std::string& getErrorMsg() const { return ErrorMsg; }
    private:
      uint64_t Value;
      std::string ErrorMsg;
    };

    StringRef getTokenForError(StringRef Expr) const {
      if (Expr.empty())
        return "";

      StringRef Token, Remaining;
      if (isalpha(Expr[0]))
        std::tie(Token, Remaining) = parseSymbol(Expr);
      else if (isdigit(Expr[0]))
        std::tie(Token, Remaining) = parseNumberString(Expr);
      else {
        unsigned TokLen = 1;
        if (Expr.startswith("<<") || Expr.startswith(">>"))
          TokLen = 2;
        Token = Expr.substr(0, TokLen);
      }
      return Token;
    }

    EvalResult unexpectedToken(StringRef TokenStart,
                               StringRef SubExpr,
                               StringRef ErrText) const {
      std::string ErrorMsg("Encountered unexpected token '");
      ErrorMsg += getTokenForError(TokenStart);
      if (SubExpr != "") {
        ErrorMsg += "' while parsing subexpression '";
        ErrorMsg += SubExpr;
      }
      ErrorMsg += "'";
      if (ErrText != "") {
        ErrorMsg += " ";
        ErrorMsg += ErrText;
      }
      return EvalResult(std::move(ErrorMsg));
    }

    bool handleError(StringRef Expr, const EvalResult &R) const {
      assert(R.hasError() && "Not an error result.");
      ErrStream << "Error evaluating expression '" << Expr << "': "
                << R.getErrorMsg() << "\n";
      return false;
    }

    std::pair<BinOpToken, StringRef> parseBinOpToken(StringRef Expr) const {
      if (Expr.empty())
        return std::make_pair(BinOpToken::Invalid, "");

      // Handle the two 2-character tokens.
      if (Expr.startswith("<<"))
        return std::make_pair(BinOpToken::ShiftLeft,
                              Expr.substr(2).ltrim());
      if (Expr.startswith(">>"))
        return std::make_pair(BinOpToken::ShiftRight,
                              Expr.substr(2).ltrim());

      // Handle one-character tokens.
      BinOpToken Op;
      switch (Expr[0]) {
        default: return std::make_pair(BinOpToken::Invalid, Expr);
        case '+': Op = BinOpToken::Add; break;
        case '-': Op = BinOpToken::Sub; break;
        case '&': Op = BinOpToken::BitwiseAnd; break;
        case '|': Op = BinOpToken::BitwiseOr; break;
      }

      return std::make_pair(Op, Expr.substr(1).ltrim());
    }

    EvalResult computeBinOpResult(BinOpToken Op, const EvalResult &LHSResult,
                                  const EvalResult &RHSResult) const {
      switch (Op) {
      default: llvm_unreachable("Tried to evaluate unrecognized operation.");
      case BinOpToken::Add:
        return EvalResult(LHSResult.getValue() + RHSResult.getValue());
      case BinOpToken::Sub:
        return EvalResult(LHSResult.getValue() - RHSResult.getValue());
      case BinOpToken::BitwiseAnd:
        return EvalResult(LHSResult.getValue() & RHSResult.getValue());
      case BinOpToken::BitwiseOr:
        return EvalResult(LHSResult.getValue() | RHSResult.getValue());
      case BinOpToken::ShiftLeft:
        return EvalResult(LHSResult.getValue() << RHSResult.getValue());
      case BinOpToken::ShiftRight:
        return EvalResult(LHSResult.getValue() >> RHSResult.getValue());
      }
    }

    // Parse a symbol and return a (string, string) pair representing the symbol
    // name and expression remaining to be parsed.
    std::pair<StringRef, StringRef> parseSymbol(StringRef Expr) const {
      size_t FirstNonSymbol =
        Expr.find_first_not_of("0123456789"
                               "abcdefghijklmnopqrstuvwxyz"
                               "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                               ":_");
      return std::make_pair(Expr.substr(0, FirstNonSymbol),
                            Expr.substr(FirstNonSymbol).ltrim());
    }

    // Evaluate a call to decode_operand. Decode the instruction operand at the
    // given symbol and get the value of the requested operand.
    // Returns an error if the instruction cannot be decoded, or the requested
    // operand is not an immediate.
    // On success, retuns a pair containing the value of the operand, plus
    // the expression remaining to be evaluated.
    std::pair<EvalResult, StringRef> evalDecodeOperand(StringRef Expr) const {
      if (!Expr.startswith("("))
        return std::make_pair(unexpectedToken(Expr, Expr, "expected '('"), "");
      StringRef RemainingExpr = Expr.substr(1).ltrim();
      StringRef Symbol;
      std::tie(Symbol, RemainingExpr) = parseSymbol(RemainingExpr);

      if (!Checker.isSymbolValid(Symbol))
        return std::make_pair(EvalResult(("Cannot decode unknown symbol '" +
                                          Symbol + "'").str()),
                              "");

      if (!RemainingExpr.startswith(","))
        return std::make_pair(unexpectedToken(RemainingExpr, RemainingExpr,
                                              "expected ','"),
                              "");
      RemainingExpr = RemainingExpr.substr(1).ltrim();

      EvalResult OpIdxExpr;
      std::tie(OpIdxExpr, RemainingExpr) = evalNumberExpr(RemainingExpr);
      if (OpIdxExpr.hasError())
        return std::make_pair(OpIdxExpr, "");

      if (!RemainingExpr.startswith(")"))
        return std::make_pair(unexpectedToken(RemainingExpr, RemainingExpr,
                                              "expected ')'"),
                              "");
      RemainingExpr = RemainingExpr.substr(1).ltrim();

      MCInst Inst;
      uint64_t Size;
      if (!decodeInst(Symbol, Inst, Size))
        return std::make_pair(EvalResult(("Couldn't decode instruction at '" +
                                          Symbol + "'").str()),
                              "");

      unsigned OpIdx = OpIdxExpr.getValue();
      if (OpIdx >= Inst.getNumOperands()) {
        std::string ErrMsg;
        raw_string_ostream ErrMsgStream(ErrMsg);
        ErrMsgStream << "Invalid operand index '" << format("%i", OpIdx)
                     << "' for instruction '" << Symbol
                     << "'. Instruction has only "
                     << format("%i", Inst.getNumOperands())
                     << " operands.\nInstruction is:\n  ";
        Inst.dump_pretty(ErrMsgStream,
                         Checker.Disassembler->getContext().getAsmInfo(),
                         Checker.InstPrinter);
        return std::make_pair(EvalResult(ErrMsgStream.str()), "");
      }

      const MCOperand &Op = Inst.getOperand(OpIdx);
      if (!Op.isImm()) {
        std::string ErrMsg;
        raw_string_ostream ErrMsgStream(ErrMsg);
        ErrMsgStream << "Operand '" << format("%i", OpIdx)
                     << "' of instruction '" << Symbol
                     << "' is not an immediate.\nInstruction is:\n  ";
        Inst.dump_pretty(ErrMsgStream,
                         Checker.Disassembler->getContext().getAsmInfo(),
                         Checker.InstPrinter);

        return std::make_pair(EvalResult(ErrMsgStream.str()), "");
      }

      return std::make_pair(EvalResult(Op.getImm()), RemainingExpr);
    }

    // Evaluate a call to next_pc. Decode the instruction at the given
    // symbol and return the following program counter..
    // Returns an error if the instruction cannot be decoded.
    // On success, returns a pair containing the next PC, plus the length of the
    // expression remaining to be evaluated.
    std::pair<EvalResult, StringRef> evalNextPC(StringRef Expr) const {
      if (!Expr.startswith("("))
        return std::make_pair(unexpectedToken(Expr, Expr, "expected '('"), "");
      StringRef RemainingExpr = Expr.substr(1).ltrim();
      StringRef Symbol;
      std::tie(Symbol, RemainingExpr) = parseSymbol(RemainingExpr);

      if (!Checker.isSymbolValid(Symbol))
        return std::make_pair(EvalResult(("Cannot decode unknown symbol '"
                                          + Symbol + "'").str()),
                              "");

      if (!RemainingExpr.startswith(")"))
        return std::make_pair(unexpectedToken(RemainingExpr, RemainingExpr,
                                              "expected ')'"),
                              "");
      RemainingExpr = RemainingExpr.substr(1).ltrim();

      MCInst Inst;
      uint64_t Size;
      if (!decodeInst(Symbol, Inst, Size))
        return std::make_pair(EvalResult(("Couldn't decode instruction at '" +
                                          Symbol + "'").str()),
                              "");
      uint64_t NextPC = Checker.getSymbolAddress(Symbol) + Size;

      return std::make_pair(EvalResult(NextPC), RemainingExpr);
    }

    // Evaluate an identiefer expr, which may be a symbol, or a call to
    // one of the builtin functions: get_insn_opcode or get_insn_length.
    // Return the result, plus the expression remaining to be parsed.
    std::pair<EvalResult, StringRef> evalIdentifierExpr(StringRef Expr) const {
      StringRef Symbol;
      StringRef RemainingExpr;
      std::tie(Symbol, RemainingExpr) = parseSymbol(Expr);

      // Check for builtin function calls.
      if (Symbol == "decode_operand")
        return evalDecodeOperand(RemainingExpr);
      else if (Symbol == "next_pc")
        return evalNextPC(RemainingExpr);

      if (!Checker.isSymbolValid(Symbol)) {
        std::string ErrMsg("No known address for symbol '");
        ErrMsg += Symbol;
        ErrMsg += "'";
        if (Symbol.startswith("L"))
          ErrMsg += " (this appears to be an assembler local label - "
                    " perhaps drop the 'L'?)";

        return std::make_pair(EvalResult(ErrMsg), "");
      }

      // Looks like a plain symbol reference.
      return std::make_pair(EvalResult(Checker.getSymbolAddress(Symbol)),
                            RemainingExpr);
    }

    // Parse a number (hexadecimal or decimal) and return a (string, string)
    // pair representing the number and the expression remaining to be parsed.
    std::pair<StringRef, StringRef> parseNumberString(StringRef Expr) const {
      size_t FirstNonDigit = StringRef::npos;
      if (Expr.startswith("0x")) {
        FirstNonDigit = Expr.find_first_not_of("0123456789abcdefABCDEF", 2);
        if (FirstNonDigit == StringRef::npos)
          FirstNonDigit = Expr.size();
      } else {
        FirstNonDigit = Expr.find_first_not_of("0123456789");
        if (FirstNonDigit == StringRef::npos)
          FirstNonDigit = Expr.size();
      }
      return std::make_pair(Expr.substr(0, FirstNonDigit),
                            Expr.substr(FirstNonDigit));
    }

    // Evaluate a constant numeric expression (hexidecimal or decimal) and
    // return a pair containing the result, and the expression remaining to be
    // evaluated.
    std::pair<EvalResult, StringRef> evalNumberExpr(StringRef Expr) const {
      StringRef ValueStr;
      StringRef RemainingExpr;
      std::tie(ValueStr, RemainingExpr) = parseNumberString(Expr);

      if (ValueStr.empty() || !isdigit(ValueStr[0]))
        return std::make_pair(unexpectedToken(RemainingExpr, RemainingExpr,
                                              "expected number"),
                              "");
      uint64_t Value;
      ValueStr.getAsInteger(0, Value);
      return std::make_pair(EvalResult(Value), RemainingExpr);
    }

    // Evaluate an expression of the form "(<expr>)" and return a pair
    // containing the result of evaluating <expr>, plus the expression
    // remaining to be parsed.
    std::pair<EvalResult, StringRef> evalParensExpr(StringRef Expr) const {
      assert(Expr.startswith("(") && "Not a parenthesized expression");
      EvalResult SubExprResult;
      StringRef RemainingExpr;
      std::tie(SubExprResult, RemainingExpr) =
        evalComplexExpr(evalSimpleExpr(Expr.substr(1).ltrim()));
      if (SubExprResult.hasError())
        return std::make_pair(SubExprResult, "");
      if (!RemainingExpr.startswith(")"))
        return std::make_pair(unexpectedToken(RemainingExpr, Expr,
                                              "expected ')'"),
                              "");
      RemainingExpr = RemainingExpr.substr(1).ltrim();
      return std::make_pair(SubExprResult, RemainingExpr);
    }

    // Evaluate an expression in one of the following forms:
    //   *{<number>}<symbol>
    //   *{<number>}(<symbol> + <number>)
    //   *{<number>}(<symbol> - <number>)
    // Return a pair containing the result, plus the expression remaining to be
    // parsed.
    std::pair<EvalResult, StringRef> evalLoadExpr(StringRef Expr) const {
      assert(Expr.startswith("*") && "Not a load expression");
      StringRef RemainingExpr = Expr.substr(1).ltrim();
      // Parse read size.
      if (!RemainingExpr.startswith("{"))
        return std::make_pair(EvalResult("Expected '{' following '*'."), "");
      RemainingExpr = RemainingExpr.substr(1).ltrim();
      EvalResult ReadSizeExpr;
      std::tie(ReadSizeExpr, RemainingExpr) = evalNumberExpr(RemainingExpr);
      if (ReadSizeExpr.hasError())
        return std::make_pair(ReadSizeExpr, RemainingExpr);
      uint64_t ReadSize = ReadSizeExpr.getValue();
      if (ReadSize < 1 || ReadSize > 8)
        return std::make_pair(EvalResult("Invalid size for dereference."), "");
      if (!RemainingExpr.startswith("}"))
        return std::make_pair(EvalResult("Missing '}' for dereference."), "");
      RemainingExpr = RemainingExpr.substr(1).ltrim();

      // Check for '(symbol +/- constant)' form.
      bool SymbolPlusConstant = false;
      if (RemainingExpr.startswith("(")) {
        SymbolPlusConstant = true;
        RemainingExpr = RemainingExpr.substr(1).ltrim();
      }

      // Read symbol.
      StringRef Symbol;
      std::tie(Symbol, RemainingExpr) = parseSymbol(RemainingExpr);

      if (!Checker.isSymbolValid(Symbol))
        return std::make_pair(EvalResult(("Cannot dereference unknown symbol '"
                                          + Symbol + "'").str()),
                              "");

      // Set up defaut offset.
      int64_t Offset = 0;

      // Handle "+/- constant)" portion if necessary.
      if (SymbolPlusConstant) {
        char OpChar = RemainingExpr[0];
        if (OpChar != '+' && OpChar != '-')
          return std::make_pair(EvalResult("Invalid operator in load address."),
                                "");
        RemainingExpr = RemainingExpr.substr(1).ltrim();

        EvalResult OffsetExpr;
        std::tie(OffsetExpr, RemainingExpr) = evalNumberExpr(RemainingExpr);

        Offset = (OpChar == '+') ?
                   OffsetExpr.getValue() : -1 * OffsetExpr.getValue();

        if (!RemainingExpr.startswith(")"))
          return std::make_pair(EvalResult("Missing ')' in load address."),
                                "");

        RemainingExpr = RemainingExpr.substr(1).ltrim();
      }

      return std::make_pair(
               EvalResult(Checker.readMemoryAtSymbol(Symbol, Offset, ReadSize)),
               RemainingExpr);
    }

    // Evaluate a "simple" expression. This is any expression that _isn't_ an
    // un-parenthesized binary expression.
    //
    // "Simple" expressions can be optionally bit-sliced. See evalSlicedExpr.
    //
    // Returns a pair containing the result of the evaluation, plus the
    // expression remaining to be parsed.
    std::pair<EvalResult, StringRef> evalSimpleExpr(StringRef Expr) const {
      EvalResult SubExprResult;
      StringRef RemainingExpr;

      if (Expr.empty())
        return std::make_pair(EvalResult("Unexpected end of expression"), "");

      if (Expr[0] == '(')
        std::tie(SubExprResult, RemainingExpr) = evalParensExpr(Expr);
      else if (Expr[0] == '*')
        std::tie(SubExprResult, RemainingExpr) = evalLoadExpr(Expr);
      else if (isalpha(Expr[0]))
        std::tie(SubExprResult, RemainingExpr) = evalIdentifierExpr(Expr);
      else if (isdigit(Expr[0]))
        std::tie(SubExprResult, RemainingExpr) = evalNumberExpr(Expr);

      if (SubExprResult.hasError())
        return std::make_pair(SubExprResult, RemainingExpr);

      // Evaluate bit-slice if present.
      if (RemainingExpr.startswith("["))
        std::tie(SubExprResult, RemainingExpr) =
          evalSliceExpr(std::make_pair(SubExprResult, RemainingExpr));

      return std::make_pair(SubExprResult, RemainingExpr);
    }

    // Evaluate a bit-slice of an expression.
    // A bit-slice has the form "<expr>[high:low]". The result of evaluating a
    // slice is the bits between high and low (inclusive) in the original
    // expression, right shifted so that the "low" bit is in position 0 in the
    // result.
    // Returns a pair containing the result of the slice operation, plus the
    // expression remaining to be parsed.
    std::pair<EvalResult, StringRef> evalSliceExpr(
                                    std::pair<EvalResult, StringRef> Ctx) const{
      EvalResult SubExprResult;
      StringRef RemainingExpr;
      std::tie(SubExprResult, RemainingExpr) = Ctx;

      assert(RemainingExpr.startswith("[") && "Not a slice expr.");
      RemainingExpr = RemainingExpr.substr(1).ltrim();

      EvalResult HighBitExpr;
      std::tie(HighBitExpr, RemainingExpr) = evalNumberExpr(RemainingExpr);

      if (HighBitExpr.hasError())
        return std::make_pair(HighBitExpr, RemainingExpr);

      if (!RemainingExpr.startswith(":"))
        return std::make_pair(unexpectedToken(RemainingExpr, RemainingExpr,
                                              "expected ':'"),
                              "");
      RemainingExpr = RemainingExpr.substr(1).ltrim();

      EvalResult LowBitExpr;
      std::tie(LowBitExpr, RemainingExpr) = evalNumberExpr(RemainingExpr);

      if (LowBitExpr.hasError())
        return std::make_pair(LowBitExpr, RemainingExpr);

      if (!RemainingExpr.startswith("]"))
        return std::make_pair(unexpectedToken(RemainingExpr, RemainingExpr,
                                              "expected ']'"),
                              "");
      RemainingExpr = RemainingExpr.substr(1).ltrim();

      unsigned HighBit = HighBitExpr.getValue();
      unsigned LowBit = LowBitExpr.getValue();
      uint64_t Mask = ((uint64_t)1 << (HighBit - LowBit + 1)) - 1;
      uint64_t SlicedValue = (SubExprResult.getValue() >> LowBit) & Mask;
      return std::make_pair(EvalResult(SlicedValue), RemainingExpr);
    }

    // Evaluate a "complex" expression.
    // Takes an already evaluated subexpression and checks for the presence of a
    // binary operator, computing the result of the binary operation if one is
    // found. Used to make arithmetic expressions left-associative.
    // Returns a pair containing the ultimate result of evaluating the
    // expression, plus the expression remaining to be evaluated.
    std::pair<EvalResult, StringRef> evalComplexExpr(
                                   std::pair<EvalResult, StringRef> Ctx) const {
      EvalResult LHSResult;
      StringRef RemainingExpr;
      std::tie(LHSResult, RemainingExpr) = Ctx;

      // If there was an error, or there's nothing left to evaluate, return the
      // result.
      if (LHSResult.hasError() || RemainingExpr == "")
        return std::make_pair(LHSResult, RemainingExpr);

      // Otherwise check if this is a binary expressioan.
      BinOpToken BinOp;
      std::tie(BinOp, RemainingExpr) = parseBinOpToken(RemainingExpr);

      // If this isn't a recognized expression just return.
      if (BinOp == BinOpToken::Invalid)
        return std::make_pair(LHSResult, RemainingExpr);

      // This is a recognized bin-op. Evaluate the RHS, then evaluate the binop.
      EvalResult RHSResult;
      std::tie(RHSResult, RemainingExpr) = evalSimpleExpr(RemainingExpr);

      // If there was an error evaluating the RHS, return it.
      if (RHSResult.hasError())
        return std::make_pair(RHSResult, RemainingExpr);

      // This is a binary expression - evaluate and try to continue as a
      // complex expr.
      EvalResult ThisResult(computeBinOpResult(BinOp, LHSResult, RHSResult));

      return evalComplexExpr(std::make_pair(ThisResult, RemainingExpr));
    }

    bool decodeInst(StringRef Symbol, MCInst &Inst, uint64_t &Size) const {
      MCDisassembler *Dis = Checker.Disassembler;
      StringRef SectionMem = Checker.getSubsectionStartingAt(Symbol);
      StringRefMemoryObject SectionBytes(SectionMem, 0);

      MCDisassembler::DecodeStatus S =
        Dis->getInstruction(Inst, Size, SectionBytes, 0, nulls(), nulls());

      return (S == MCDisassembler::Success);
    }

  };

}

bool RuntimeDyldChecker::check(StringRef CheckExpr) const {
  CheckExpr = CheckExpr.trim();
  DEBUG(llvm::dbgs() << "RuntimeDyldChecker: Checking '" << CheckExpr
                     << "'...\n");
  RuntimeDyldCheckerExprEval P(*this, ErrStream);
  bool Result = P.evaluate(CheckExpr);
  (void)Result;
  DEBUG(llvm::dbgs() << "RuntimeDyldChecker: '" << CheckExpr << "' "
                     << (Result ? "passed" : "FAILED") << ".\n");
  return Result;
}

bool RuntimeDyldChecker::checkAllRulesInBuffer(StringRef RulePrefix,
                                               MemoryBuffer* MemBuf) const {
  bool DidAllTestsPass = true;
  unsigned NumRules = 0;

  const char *LineStart = MemBuf->getBufferStart();

  // Eat whitespace.
  while (LineStart != MemBuf->getBufferEnd() &&
         std::isspace(*LineStart))
    ++LineStart;

  while (LineStart != MemBuf->getBufferEnd() && *LineStart != '\0') {
    const char *LineEnd = LineStart;
    while (LineEnd != MemBuf->getBufferEnd() &&
           *LineEnd != '\r' && *LineEnd != '\n')
      ++LineEnd;

    StringRef Line(LineStart, LineEnd - LineStart);
    if (Line.startswith(RulePrefix)) {
      DidAllTestsPass &= check(Line.substr(RulePrefix.size()));
      ++NumRules;
    }

    // Eat whitespace.
    LineStart = LineEnd;
    while (LineStart != MemBuf->getBufferEnd() &&
           std::isspace(*LineStart))
      ++LineStart;
  }
  return DidAllTestsPass && (NumRules != 0);
}

bool RuntimeDyldChecker::isSymbolValid(StringRef Symbol) const {
  return RTDyld.getSymbolAddress(Symbol) != nullptr;
}

uint64_t RuntimeDyldChecker::getSymbolAddress(StringRef Symbol) const {
  return RTDyld.getAnySymbolRemoteAddress(Symbol);
}

uint64_t RuntimeDyldChecker::readMemoryAtSymbol(StringRef Symbol,
                                                int64_t Offset,
                                                unsigned Size) const {
  uint8_t *Src = RTDyld.getSymbolAddress(Symbol);
  uint64_t Result = 0;
  memcpy(&Result, Src + Offset, Size);
  return Result;
}

StringRef RuntimeDyldChecker::getSubsectionStartingAt(StringRef Name) const {
  RuntimeDyldImpl::SymbolTableMap::const_iterator pos =
    RTDyld.GlobalSymbolTable.find(Name);
  if (pos == RTDyld.GlobalSymbolTable.end())
    return StringRef();
  RuntimeDyldImpl::SymbolLoc Loc = pos->second;
  uint8_t *SectionAddr = RTDyld.getSectionAddress(Loc.first);
  return StringRef(reinterpret_cast<const char*>(SectionAddr) + Loc.second,
                   RTDyld.Sections[Loc.first].Size - Loc.second);
}
