//===--- SemaStmtAsm.cpp - Semantic Analysis for Asm Statements -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for inline asm statements.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
using namespace clang;
using namespace sema;

/// CheckAsmLValue - GNU C has an extremely ugly extension whereby they silently
/// ignore "noop" casts in places where an lvalue is required by an inline asm.
/// We emulate this behavior when -fheinous-gnu-extensions is specified, but
/// provide a strong guidance to not use it.
///
/// This method checks to see if the argument is an acceptable l-value and
/// returns false if it is a case we can handle.
static bool CheckAsmLValue(const Expr *E, Sema &S) {
  // Type dependent expressions will be checked during instantiation.
  if (E->isTypeDependent())
    return false;

  if (E->isLValue())
    return false;  // Cool, this is an lvalue.

  // Okay, this is not an lvalue, but perhaps it is the result of a cast that we
  // are supposed to allow.
  const Expr *E2 = E->IgnoreParenNoopCasts(S.Context);
  if (E != E2 && E2->isLValue()) {
    if (!S.getLangOpts().HeinousExtensions)
      S.Diag(E2->getLocStart(), diag::err_invalid_asm_cast_lvalue)
        << E->getSourceRange();
    else
      S.Diag(E2->getLocStart(), diag::warn_invalid_asm_cast_lvalue)
        << E->getSourceRange();
    // Accept, even if we emitted an error diagnostic.
    return false;
  }

  // None of the above, just randomly invalid non-lvalue.
  return true;
}

/// isOperandMentioned - Return true if the specified operand # is mentioned
/// anywhere in the decomposed asm string.
static bool isOperandMentioned(unsigned OpNo,
                         ArrayRef<GCCAsmStmt::AsmStringPiece> AsmStrPieces) {
  for (unsigned p = 0, e = AsmStrPieces.size(); p != e; ++p) {
    const GCCAsmStmt::AsmStringPiece &Piece = AsmStrPieces[p];
    if (!Piece.isOperand()) continue;

    // If this is a reference to the input and if the input was the smaller
    // one, then we have to reject this asm.
    if (Piece.getOperandNo() == OpNo)
      return true;
  }
  return false;
}

StmtResult Sema::ActOnGCCAsmStmt(SourceLocation AsmLoc, bool IsSimple,
                                 bool IsVolatile, unsigned NumOutputs,
                                 unsigned NumInputs, IdentifierInfo **Names,
                                 MultiExprArg constraints, MultiExprArg exprs,
                                 Expr *asmString, MultiExprArg clobbers,
                                 SourceLocation RParenLoc) {
  unsigned NumClobbers = clobbers.size();
  StringLiteral **Constraints =
    reinterpret_cast<StringLiteral**>(constraints.data());
  Expr **Exprs = exprs.data();
  StringLiteral *AsmString = cast<StringLiteral>(asmString);
  StringLiteral **Clobbers = reinterpret_cast<StringLiteral**>(clobbers.data());

  SmallVector<TargetInfo::ConstraintInfo, 4> OutputConstraintInfos;

  // The parser verifies that there is a string literal here.
  if (!AsmString->isAscii())
    return StmtError(Diag(AsmString->getLocStart(),diag::err_asm_wide_character)
      << AsmString->getSourceRange());

  for (unsigned i = 0; i != NumOutputs; i++) {
    StringLiteral *Literal = Constraints[i];
    if (!Literal->isAscii())
      return StmtError(Diag(Literal->getLocStart(),diag::err_asm_wide_character)
        << Literal->getSourceRange());

    StringRef OutputName;
    if (Names[i])
      OutputName = Names[i]->getName();

    TargetInfo::ConstraintInfo Info(Literal->getString(), OutputName);
    if (!Context.getTargetInfo().validateOutputConstraint(Info))
      return StmtError(Diag(Literal->getLocStart(),
                            diag::err_asm_invalid_output_constraint)
                       << Info.getConstraintStr());

    // Check that the output exprs are valid lvalues.
    Expr *OutputExpr = Exprs[i];
    if (CheckAsmLValue(OutputExpr, *this)) {
      return StmtError(Diag(OutputExpr->getLocStart(),
                  diag::err_asm_invalid_lvalue_in_output)
        << OutputExpr->getSourceRange());
    }

    OutputConstraintInfos.push_back(Info);
  }

  SmallVector<TargetInfo::ConstraintInfo, 4> InputConstraintInfos;

  for (unsigned i = NumOutputs, e = NumOutputs + NumInputs; i != e; i++) {
    StringLiteral *Literal = Constraints[i];
    if (!Literal->isAscii())
      return StmtError(Diag(Literal->getLocStart(),diag::err_asm_wide_character)
        << Literal->getSourceRange());

    StringRef InputName;
    if (Names[i])
      InputName = Names[i]->getName();

    TargetInfo::ConstraintInfo Info(Literal->getString(), InputName);
    if (!Context.getTargetInfo().validateInputConstraint(OutputConstraintInfos.data(),
                                                NumOutputs, Info)) {
      return StmtError(Diag(Literal->getLocStart(),
                            diag::err_asm_invalid_input_constraint)
                       << Info.getConstraintStr());
    }

    Expr *InputExpr = Exprs[i];

    // Only allow void types for memory constraints.
    if (Info.allowsMemory() && !Info.allowsRegister()) {
      if (CheckAsmLValue(InputExpr, *this))
        return StmtError(Diag(InputExpr->getLocStart(),
                              diag::err_asm_invalid_lvalue_in_input)
                         << Info.getConstraintStr()
                         << InputExpr->getSourceRange());
    }

    if (Info.allowsRegister()) {
      if (InputExpr->getType()->isVoidType()) {
        return StmtError(Diag(InputExpr->getLocStart(),
                              diag::err_asm_invalid_type_in_input)
          << InputExpr->getType() << Info.getConstraintStr()
          << InputExpr->getSourceRange());
      }
    }

    ExprResult Result = DefaultFunctionArrayLvalueConversion(Exprs[i]);
    if (Result.isInvalid())
      return StmtError();

    Exprs[i] = Result.take();
    InputConstraintInfos.push_back(Info);
  }

  // Check that the clobbers are valid.
  for (unsigned i = 0; i != NumClobbers; i++) {
    StringLiteral *Literal = Clobbers[i];
    if (!Literal->isAscii())
      return StmtError(Diag(Literal->getLocStart(),diag::err_asm_wide_character)
        << Literal->getSourceRange());

    StringRef Clobber = Literal->getString();

    if (!Context.getTargetInfo().isValidClobber(Clobber))
      return StmtError(Diag(Literal->getLocStart(),
                  diag::err_asm_unknown_register_name) << Clobber);
  }

  GCCAsmStmt *NS =
    new (Context) GCCAsmStmt(Context, AsmLoc, IsSimple, IsVolatile, NumOutputs,
                             NumInputs, Names, Constraints, Exprs, AsmString,
                             NumClobbers, Clobbers, RParenLoc);
  // Validate the asm string, ensuring it makes sense given the operands we
  // have.
  SmallVector<GCCAsmStmt::AsmStringPiece, 8> Pieces;
  unsigned DiagOffs;
  if (unsigned DiagID = NS->AnalyzeAsmString(Pieces, Context, DiagOffs)) {
    Diag(getLocationOfStringLiteralByte(AsmString, DiagOffs), DiagID)
           << AsmString->getSourceRange();
    return StmtError();
  }

  // Validate tied input operands for type mismatches.
  for (unsigned i = 0, e = InputConstraintInfos.size(); i != e; ++i) {
    TargetInfo::ConstraintInfo &Info = InputConstraintInfos[i];

    // If this is a tied constraint, verify that the output and input have
    // either exactly the same type, or that they are int/ptr operands with the
    // same size (int/long, int*/long, are ok etc).
    if (!Info.hasTiedOperand()) continue;

    unsigned TiedTo = Info.getTiedOperand();
    unsigned InputOpNo = i+NumOutputs;
    Expr *OutputExpr = Exprs[TiedTo];
    Expr *InputExpr = Exprs[InputOpNo];

    if (OutputExpr->isTypeDependent() || InputExpr->isTypeDependent())
      continue;

    QualType InTy = InputExpr->getType();
    QualType OutTy = OutputExpr->getType();
    if (Context.hasSameType(InTy, OutTy))
      continue;  // All types can be tied to themselves.

    // Decide if the input and output are in the same domain (integer/ptr or
    // floating point.
    enum AsmDomain {
      AD_Int, AD_FP, AD_Other
    } InputDomain, OutputDomain;

    if (InTy->isIntegerType() || InTy->isPointerType())
      InputDomain = AD_Int;
    else if (InTy->isRealFloatingType())
      InputDomain = AD_FP;
    else
      InputDomain = AD_Other;

    if (OutTy->isIntegerType() || OutTy->isPointerType())
      OutputDomain = AD_Int;
    else if (OutTy->isRealFloatingType())
      OutputDomain = AD_FP;
    else
      OutputDomain = AD_Other;

    // They are ok if they are the same size and in the same domain.  This
    // allows tying things like:
    //   void* to int*
    //   void* to int            if they are the same size.
    //   double to long double   if they are the same size.
    //
    uint64_t OutSize = Context.getTypeSize(OutTy);
    uint64_t InSize = Context.getTypeSize(InTy);
    if (OutSize == InSize && InputDomain == OutputDomain &&
        InputDomain != AD_Other)
      continue;

    // If the smaller input/output operand is not mentioned in the asm string,
    // then we can promote the smaller one to a larger input and the asm string
    // won't notice.
    bool SmallerValueMentioned = false;

    // If this is a reference to the input and if the input was the smaller
    // one, then we have to reject this asm.
    if (isOperandMentioned(InputOpNo, Pieces)) {
      // This is a use in the asm string of the smaller operand.  Since we
      // codegen this by promoting to a wider value, the asm will get printed
      // "wrong".
      SmallerValueMentioned |= InSize < OutSize;
    }
    if (isOperandMentioned(TiedTo, Pieces)) {
      // If this is a reference to the output, and if the output is the larger
      // value, then it's ok because we'll promote the input to the larger type.
      SmallerValueMentioned |= OutSize < InSize;
    }

    // If the smaller value wasn't mentioned in the asm string, and if the
    // output was a register, just extend the shorter one to the size of the
    // larger one.
    if (!SmallerValueMentioned && InputDomain != AD_Other &&
        OutputConstraintInfos[TiedTo].allowsRegister())
      continue;

    // Either both of the operands were mentioned or the smaller one was
    // mentioned.  One more special case that we'll allow: if the tied input is
    // integer, unmentioned, and is a constant, then we'll allow truncating it
    // down to the size of the destination.
    if (InputDomain == AD_Int && OutputDomain == AD_Int &&
        !isOperandMentioned(InputOpNo, Pieces) &&
        InputExpr->isEvaluatable(Context)) {
      CastKind castKind =
        (OutTy->isBooleanType() ? CK_IntegralToBoolean : CK_IntegralCast);
      InputExpr = ImpCastExprToType(InputExpr, OutTy, castKind).take();
      Exprs[InputOpNo] = InputExpr;
      NS->setInputExpr(i, InputExpr);
      continue;
    }

    Diag(InputExpr->getLocStart(),
         diag::err_asm_tying_incompatible_types)
      << InTy << OutTy << OutputExpr->getSourceRange()
      << InputExpr->getSourceRange();
    return StmtError();
  }

  return Owned(NS);
}

// getSpelling - Get the spelling of the AsmTok token.
static StringRef getSpelling(Sema &SemaRef, Token AsmTok) {
  StringRef Asm;
  SmallString<512> TokenBuf;
  TokenBuf.resize(512);
  bool StringInvalid = false;
  Asm = SemaRef.PP.getSpelling(AsmTok, TokenBuf, &StringInvalid);
  assert (!StringInvalid && "Expected valid string!");
  return Asm;
}

// Determine if this is a simple MSAsm instruction.
static bool isSimpleMSAsm(std::vector<StringRef> &Pieces,
                          const TargetInfo &TI) {
  for (unsigned i = 1, e = Pieces.size(); i != e; ++i) {
    if (!TI.isValidGCCRegisterName(Pieces[i]))
      return false;
  }
  return true;
}

// Determine if this is a simple MSAsm block.
static bool isSimpleMSAsm(std::vector<std::vector<StringRef> > Pieces,
                          const TargetInfo &TI) {
  for (unsigned i = 0, e = Pieces.size(); i != e; ++i)
    if (!isSimpleMSAsm(Pieces[i], TI))
      return false;
  return true;
}

// Break the AsmString into pieces (i.e., mnemonic and operands).
static void buildMSAsmPieces(StringRef Asm, std::vector<StringRef> &Pieces) {
  std::pair<StringRef,StringRef> Split = Asm.split(' ');

  // Mnemonic
  Pieces.push_back(Split.first);
  Asm = Split.second;

  // Operands
  while (!Asm.empty()) {
    Split = Asm.split(", ");
    Pieces.push_back(Split.first);
    Asm = Split.second;
  }
}

static void buildMSAsmPieces(std::vector<std::string> &AsmStrings,
                             std::vector<std::vector<StringRef> > &Pieces) {
  for (unsigned i = 0, e = AsmStrings.size(); i != e; ++i)
    buildMSAsmPieces(AsmStrings[i], Pieces[i]);
}

// Build the individual assembly instruction(s) and place them in the AsmStrings
// vector.  These strings are fed to the AsmParser.  Returns true on error.
static bool buildMSAsmStrings(Sema &SemaRef,
                              SourceLocation AsmLoc,
                              ArrayRef<Token> AsmToks,
                              std::vector<std::string> &AsmStrings) {
  assert (!AsmToks.empty() && "Didn't expect an empty AsmToks!");

  SmallString<512> Asm;
  for (unsigned i = 0, e = AsmToks.size(); i < e; ++i) {
    bool isNewAsm = ((i == 0) ||
                     AsmToks[i].isAtStartOfLine() ||
                     AsmToks[i].is(tok::kw_asm));

    if (isNewAsm) {
      if (i) {
        AsmStrings.push_back(Asm.str());
        Asm.clear();
      }
      if (AsmToks[i].is(tok::kw_asm)) {
        i++; // Skip __asm
        if (i == e) {
          SemaRef.Diag(AsmLoc, diag::err_asm_empty);
          return true;
        }
      }
    }

    if (i && AsmToks[i].hasLeadingSpace() && !isNewAsm)
      Asm += ' ';

    StringRef Spelling = getSpelling(SemaRef, AsmToks[i]);
    Asm += Spelling;
  }
  AsmStrings.push_back(Asm.str());

  return false;
}

#define DEF_SIMPLE_MSASM(STR)                                                \
  MSAsmStmt *NS =                                                            \
    new (Context) MSAsmStmt(Context, AsmLoc, LBraceLoc, /*IsSimple*/ true,   \
                            /*IsVolatile*/ true, AsmToks, Inputs, Outputs,   \
                            InputExprs, OutputExprs, STR, Constraints,       \
                            Clobbers, EndLoc);

StmtResult Sema::ActOnMSAsmStmt(SourceLocation AsmLoc, SourceLocation LBraceLoc,
                                ArrayRef<Token> AsmToks,SourceLocation EndLoc) {
  SmallVector<StringRef, 4> Constraints;
  std::vector<std::string> InputConstraints;
  std::vector<std::string> OutputConstraints;
  SmallVector<StringRef, 4> Clobbers;
  std::set<std::string> ClobberRegs;

  // FIXME: Use a struct to hold the various expression information.
  SmallVector<IdentifierInfo*, 4> Inputs;
  SmallVector<IdentifierInfo*, 4> Outputs;
  SmallVector<Expr*, 4> InputExprs;
  SmallVector<Expr*, 4> OutputExprs;
  SmallVector<std::string, 4> InputExprNames;
  SmallVector<std::string, 4> OutputExprNames;
  SmallVector<unsigned, 4> InputExprStrIdx;
  SmallVector<unsigned, 4> OutputExprStrIdx;

  // Empty asm statements don't need to instantiate the AsmParser, etc.
  StringRef EmptyAsmStr;
  if (AsmToks.empty()) { DEF_SIMPLE_MSASM(EmptyAsmStr); return Owned(NS); }

  std::vector<std::string> AsmStrings;
  if (buildMSAsmStrings(*this, AsmLoc, AsmToks, AsmStrings))
    return StmtError();

  std::vector<std::vector<StringRef> > Pieces(AsmStrings.size());
  buildMSAsmPieces(AsmStrings, Pieces);

  bool IsSimple = isSimpleMSAsm(Pieces, Context.getTargetInfo());

  // Get the target specific parser.
  std::string Error;
  const std::string &TT = Context.getTargetInfo().getTriple().getTriple();
  const llvm::Target *TheTarget(llvm::TargetRegistry::lookupTarget(TT, Error));

  OwningPtr<llvm::MCAsmInfo> MAI(TheTarget->createMCAsmInfo(TT));
  OwningPtr<llvm::MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TT));
  OwningPtr<llvm::MCObjectFileInfo> MOFI(new llvm::MCObjectFileInfo());
  OwningPtr<llvm::MCSubtargetInfo>
    STI(TheTarget->createMCSubtargetInfo(TT, "", ""));

  for (unsigned StrIdx = 0, e = AsmStrings.size(); StrIdx != e; ++StrIdx) {
    llvm::SourceMgr SrcMgr;
    llvm::MCContext Ctx(*MAI, *MRI, MOFI.get(), &SrcMgr);
    llvm::MemoryBuffer *Buffer =
      llvm::MemoryBuffer::getMemBuffer(AsmStrings[StrIdx], "<inline asm>");

    // Tell SrcMgr about this buffer, which is what the parser will pick up.
    SrcMgr.AddNewSourceBuffer(Buffer, llvm::SMLoc());

    OwningPtr<llvm::MCStreamer> Str(createNullStreamer(Ctx));
    OwningPtr<llvm::MCAsmParser>
      Parser(createMCAsmParser(SrcMgr, Ctx, *Str.get(), *MAI));
    OwningPtr<llvm::MCTargetAsmParser>
      TargetParser(TheTarget->createMCAsmParser(*STI, *Parser));
    // Change to the Intel dialect.
    Parser->setAssemblerDialect(1);
    Parser->setTargetParser(*TargetParser.get());

    // Prime the lexer.
    Parser->Lex();

    // Parse the opcode.
    StringRef IDVal;
    Parser->ParseIdentifier(IDVal);

    // Canonicalize the opcode to lower case.
    SmallString<128> OpcodeStr;
    for (unsigned i = 0, e = IDVal.size(); i != e; ++i)
      OpcodeStr.push_back(tolower(IDVal[i]));
    // FIXME: Convert to a StmtError.
    assert(TargetParser->mnemonicIsValid(OpcodeStr) && "Invalid mnemonic!");

    // Parse the operands.
    llvm::SMLoc IDLoc;
    SmallVector<llvm::MCParsedAsmOperand*, 8> Operands;
    bool HadError = TargetParser->ParseInstruction(OpcodeStr.str(), IDLoc,
                                                   Operands);
    // If we had an error parsing the operands, fail gracefully.
    if (HadError) { DEF_SIMPLE_MSASM(EmptyAsmStr); return Owned(NS); }

    // Match the MCInstr.
    unsigned Kind;
    unsigned Opcode;
    unsigned ErrorInfo;
    SmallVector<std::pair< unsigned, std::string >, 4> MapAndConstraints;
    HadError = TargetParser->MatchInstruction(IDLoc, Operands, *Str.get(), Kind,
                                              Opcode, MapAndConstraints,
                                              ErrorInfo,
                                              /*matchingInlineAsm*/ true);
    // If we had an error parsing the operands, fail gracefully.
    if (HadError) { DEF_SIMPLE_MSASM(EmptyAsmStr); return Owned(NS); }

    // Get the instruction descriptor.
    const llvm::MCInstrInfo *MII = TheTarget->createMCInstrInfo();
    const llvm::MCInstrDesc &Desc = MII->get(Opcode);
    llvm::MCInstPrinter *IP =
      TheTarget->createMCInstPrinter(1, *MAI, *MII, *MRI, *STI);

    // Build the list of clobbers, outputs and inputs.
    unsigned NumDefs = Desc.getNumDefs();
    for (unsigned i = 1, e = Operands.size(); i != e; ++i) {
      // Skip immediates.
      if (Operands[i]->isImm())
        continue;

      // Register.
      if (Operands[i]->isReg()) {
        // Clobber.
        if (NumDefs && (MapAndConstraints[i-1].first < NumDefs)) {
          std::string Reg;
          llvm::raw_string_ostream OS(Reg);
          IP->printRegName(OS, Operands[i]->getReg());
          StringRef Clobber(OS.str());
          if (!Context.getTargetInfo().isValidClobber(Clobber))
            return StmtError(
              Diag(AsmLoc, diag::err_asm_unknown_register_name) << Clobber);
          ClobberRegs.insert(Reg);
        }
        continue;
      }

      // Expr/Input or Output.
      StringRef Name = Pieces[StrIdx][i];
      if (IdentifierInfo *II = &Context.Idents.get(Name)) {
        CXXScopeSpec SS;
        UnqualifiedId Id;
        SourceLocation Loc;
        Id.setIdentifier(II, AsmLoc);
        ExprResult Result = ActOnIdExpression(getCurScope(), SS, Loc, Id,
                                              false, false);
        if (!Result.isInvalid()) {
          // FIXME: Determine the proper constraints.
          bool isMemDef = (i == 1) && Desc.mayStore();
          if (isMemDef) {
            Outputs.push_back(II);
            OutputExprs.push_back(Result.take());
            OutputExprNames.push_back(Name.str());
            OutputExprStrIdx.push_back(StrIdx);

            std::string Constraint = "=" + MapAndConstraints[i-1].second;
            OutputConstraints.push_back(Constraint);
          } else {
            Inputs.push_back(II);
            InputExprs.push_back(Result.take());
            InputExprNames.push_back(Name.str());
            InputExprStrIdx.push_back(StrIdx);
            InputConstraints.push_back(MapAndConstraints[i-1].second);
          }
        }
      }
    }
  }
  for (std::set<std::string>::iterator I = ClobberRegs.begin(),
         E = ClobberRegs.end(); I != E; ++I)
    Clobbers.push_back(*I);

  // Merge the output and input constraints.  Output constraints are expected
  // first.
  for (std::vector<std::string>::iterator I = OutputConstraints.begin(),
         E = OutputConstraints.end(); I != E; ++I)
    Constraints.push_back(*I);

  for (std::vector<std::string>::iterator I = InputConstraints.begin(),
         E = InputConstraints.end(); I != E; ++I)
    Constraints.push_back(*I);

  // Enumerate the AsmString expressions.
  unsigned OpNum = 0;
  for (unsigned i = 0, e = OutputExprNames.size(); i != e; ++i, ++OpNum) {
    unsigned StrIdx = OutputExprStrIdx[i];
    // Iterate over the assembly instruction pieces, skipping the mnemonic.
    for (unsigned j = 1, f = Pieces[StrIdx].size(); j != f; ++j) {
      // If the operand and the expression name match, then rewrite the operand.
      if (OutputExprNames[i] == Pieces[StrIdx][j]) {
        SmallString<32> Res;
        llvm::raw_svector_ostream OS(Res);
        OS << '$' << OpNum;
        OutputExprNames[i] = OS.str();
        Pieces[StrIdx][j] = OutputExprNames[i];
        break;
      }
    }
  }
  for (unsigned i = 0, e = InputExprNames.size(); i != e; ++i, ++OpNum) {
    unsigned StrIdx = InputExprStrIdx[i];
    // Iterate over the assembly instruction pieces, skipping the mnemonic.
    for (unsigned j = 1, f = Pieces[StrIdx].size(); j != f; ++j) {
      // If the operand and the expression name match, then rewrite the operand.
      if (InputExprNames[i] == Pieces[StrIdx][j]) {
        SmallString<32> Res;
        llvm::raw_svector_ostream OS(Res);
        OS << '$' << OpNum;
        InputExprNames[i] = OS.str();
        Pieces[StrIdx][j] = InputExprNames[i];
        break;
      }
    }
  }

  // Emit the IR assembly string.
  std::string AsmString;
  for (unsigned i = 0, e = Pieces.size(); i != e; ++i) {
    // Skip empty asm stmts.
    if (Pieces[i].empty()) continue;

    if (i > 0)
      AsmString += "\n\t";

    // Emit the mnemonic.
    AsmString += Pieces[i][0];
    if (Pieces[i].size() > 1)
      AsmString += ' ';

    // Emit the operands adding $$ to constants.
    for (unsigned j = 1, f = Pieces[i].size(); j != f; ++j) {
      if (j > 1) AsmString += ", ";
      unsigned Val;
      if (!Pieces[i][j].getAsInteger(0, Val))
        AsmString += "$$";

      AsmString += Pieces[i][j];
    }
  }

  MSAsmStmt *NS =
    new (Context) MSAsmStmt(Context, AsmLoc, LBraceLoc, IsSimple,
                            /*IsVolatile*/ true, AsmToks, Inputs, Outputs,
                            InputExprs, OutputExprs, AsmString, Constraints,
                            Clobbers, EndLoc);
  return Owned(NS);
}
