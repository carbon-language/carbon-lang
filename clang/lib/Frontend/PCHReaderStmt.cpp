//===--- PCHReaderStmt.cpp - Stmt/Expr Deserialization ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Statement/expression deserialization.  This implements the
// PCHReader::ReadStmt method.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/PCHReader.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/StmtVisitor.h"
using namespace clang;

namespace {
  class PCHStmtReader : public StmtVisitor<PCHStmtReader, unsigned> {
    PCHReader &Reader;
    const PCHReader::RecordData &Record;
    unsigned &Idx;
    llvm::SmallVectorImpl<Stmt *> &StmtStack;

  public:
    PCHStmtReader(PCHReader &Reader, const PCHReader::RecordData &Record,
                  unsigned &Idx, llvm::SmallVectorImpl<Stmt *> &StmtStack)
      : Reader(Reader), Record(Record), Idx(Idx), StmtStack(StmtStack) { }

    /// \brief The number of record fields required for the Stmt class
    /// itself.
    static const unsigned NumStmtFields = 0;

    /// \brief The number of record fields required for the Expr class
    /// itself.
    static const unsigned NumExprFields = NumStmtFields + 3;

    // Each of the Visit* functions reads in part of the expression
    // from the given record and the current expression stack, then
    // return the total number of operands that it read from the
    // expression stack.

    unsigned VisitStmt(Stmt *S);
    unsigned VisitNullStmt(NullStmt *S);
    unsigned VisitCompoundStmt(CompoundStmt *S);
    unsigned VisitSwitchCase(SwitchCase *S);
    unsigned VisitCaseStmt(CaseStmt *S);
    unsigned VisitDefaultStmt(DefaultStmt *S);
    unsigned VisitLabelStmt(LabelStmt *S);
    unsigned VisitIfStmt(IfStmt *S);
    unsigned VisitSwitchStmt(SwitchStmt *S);
    unsigned VisitWhileStmt(WhileStmt *S);
    unsigned VisitDoStmt(DoStmt *S);
    unsigned VisitForStmt(ForStmt *S);
    unsigned VisitGotoStmt(GotoStmt *S);
    unsigned VisitIndirectGotoStmt(IndirectGotoStmt *S);
    unsigned VisitContinueStmt(ContinueStmt *S);
    unsigned VisitBreakStmt(BreakStmt *S);
    unsigned VisitReturnStmt(ReturnStmt *S);
    unsigned VisitDeclStmt(DeclStmt *S);
    unsigned VisitAsmStmt(AsmStmt *S);
    unsigned VisitExpr(Expr *E);
    unsigned VisitPredefinedExpr(PredefinedExpr *E);
    unsigned VisitDeclRefExpr(DeclRefExpr *E);
    unsigned VisitIntegerLiteral(IntegerLiteral *E);
    unsigned VisitFloatingLiteral(FloatingLiteral *E);
    unsigned VisitImaginaryLiteral(ImaginaryLiteral *E);
    unsigned VisitStringLiteral(StringLiteral *E);
    unsigned VisitCharacterLiteral(CharacterLiteral *E);
    unsigned VisitParenExpr(ParenExpr *E);
    unsigned VisitUnaryOperator(UnaryOperator *E);
    unsigned VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E);
    unsigned VisitArraySubscriptExpr(ArraySubscriptExpr *E);
    unsigned VisitCallExpr(CallExpr *E);
    unsigned VisitMemberExpr(MemberExpr *E);
    unsigned VisitCastExpr(CastExpr *E);
    unsigned VisitBinaryOperator(BinaryOperator *E);
    unsigned VisitCompoundAssignOperator(CompoundAssignOperator *E);
    unsigned VisitConditionalOperator(ConditionalOperator *E);
    unsigned VisitImplicitCastExpr(ImplicitCastExpr *E);
    unsigned VisitExplicitCastExpr(ExplicitCastExpr *E);
    unsigned VisitCStyleCastExpr(CStyleCastExpr *E);
    unsigned VisitCompoundLiteralExpr(CompoundLiteralExpr *E);
    unsigned VisitExtVectorElementExpr(ExtVectorElementExpr *E);
    unsigned VisitInitListExpr(InitListExpr *E);
    unsigned VisitDesignatedInitExpr(DesignatedInitExpr *E);
    unsigned VisitImplicitValueInitExpr(ImplicitValueInitExpr *E);
    unsigned VisitVAArgExpr(VAArgExpr *E);
    unsigned VisitAddrLabelExpr(AddrLabelExpr *E);
    unsigned VisitStmtExpr(StmtExpr *E);
    unsigned VisitTypesCompatibleExpr(TypesCompatibleExpr *E);
    unsigned VisitChooseExpr(ChooseExpr *E);
    unsigned VisitGNUNullExpr(GNUNullExpr *E);
    unsigned VisitShuffleVectorExpr(ShuffleVectorExpr *E);
    unsigned VisitBlockExpr(BlockExpr *E);
    unsigned VisitBlockDeclRefExpr(BlockDeclRefExpr *E);
    unsigned VisitObjCStringLiteral(ObjCStringLiteral *E);
    unsigned VisitObjCEncodeExpr(ObjCEncodeExpr *E);
    unsigned VisitObjCSelectorExpr(ObjCSelectorExpr *E);
    unsigned VisitObjCProtocolExpr(ObjCProtocolExpr *E);
    unsigned VisitObjCIvarRefExpr(ObjCIvarRefExpr *E);
    unsigned VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *E);
    unsigned VisitObjCImplicitSetterGetterRefExpr(
                            ObjCImplicitSetterGetterRefExpr *E);
    unsigned VisitObjCMessageExpr(ObjCMessageExpr *E);
    unsigned VisitObjCSuperExpr(ObjCSuperExpr *E);
    unsigned VisitObjCIsaExpr(ObjCIsaExpr *E);

    unsigned VisitObjCForCollectionStmt(ObjCForCollectionStmt *);
    unsigned VisitObjCAtCatchStmt(ObjCAtCatchStmt *);
    unsigned VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *);
    unsigned VisitObjCAtTryStmt(ObjCAtTryStmt *);
    unsigned VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *);
    unsigned VisitObjCAtThrowStmt(ObjCAtThrowStmt *);

    unsigned VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E);
    unsigned VisitCXXConstructExpr(CXXConstructExpr *E);
  };
}

unsigned PCHStmtReader::VisitStmt(Stmt *S) {
  assert(Idx == NumStmtFields && "Incorrect statement field count");
  return 0;
}

unsigned PCHStmtReader::VisitNullStmt(NullStmt *S) {
  VisitStmt(S);
  S->setSemiLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 0;
}

unsigned PCHStmtReader::VisitCompoundStmt(CompoundStmt *S) {
  VisitStmt(S);
  unsigned NumStmts = Record[Idx++];
  S->setStmts(*Reader.getContext(),
              StmtStack.data() + StmtStack.size() - NumStmts, NumStmts);
  S->setLBracLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setRBracLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return NumStmts;
}

unsigned PCHStmtReader::VisitSwitchCase(SwitchCase *S) {
  VisitStmt(S);
  Reader.RecordSwitchCaseID(S, Record[Idx++]);
  return 0;
}

unsigned PCHStmtReader::VisitCaseStmt(CaseStmt *S) {
  VisitSwitchCase(S);
  S->setLHS(cast<Expr>(StmtStack[StmtStack.size() - 3]));
  S->setRHS(cast_or_null<Expr>(StmtStack[StmtStack.size() - 2]));
  S->setSubStmt(StmtStack.back());
  S->setCaseLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setEllipsisLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setColonLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 3;
}

unsigned PCHStmtReader::VisitDefaultStmt(DefaultStmt *S) {
  VisitSwitchCase(S);
  S->setSubStmt(StmtStack.back());
  S->setDefaultLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setColonLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 1;
}

unsigned PCHStmtReader::VisitLabelStmt(LabelStmt *S) {
  VisitStmt(S);
  S->setID(Reader.GetIdentifierInfo(Record, Idx));
  S->setSubStmt(StmtStack.back());
  S->setIdentLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  Reader.RecordLabelStmt(S, Record[Idx++]);
  return 1;
}

unsigned PCHStmtReader::VisitIfStmt(IfStmt *S) {
  VisitStmt(S);
  S->setConditionVariable(cast_or_null<VarDecl>(Reader.GetDecl(Record[Idx++])));
  S->setCond(cast<Expr>(StmtStack[StmtStack.size() - 3]));
  S->setThen(StmtStack[StmtStack.size() - 2]);
  S->setElse(StmtStack[StmtStack.size() - 1]);
  S->setIfLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setElseLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 3;
}

unsigned PCHStmtReader::VisitSwitchStmt(SwitchStmt *S) {
  VisitStmt(S);
  S->setConditionVariable(cast_or_null<VarDecl>(Reader.GetDecl(Record[Idx++])));
  S->setCond(cast<Expr>(StmtStack[StmtStack.size() - 2]));
  S->setBody(StmtStack.back());
  S->setSwitchLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  SwitchCase *PrevSC = 0;
  for (unsigned N = Record.size(); Idx != N; ++Idx) {
    SwitchCase *SC = Reader.getSwitchCaseWithID(Record[Idx]);
    if (PrevSC)
      PrevSC->setNextSwitchCase(SC);
    else
      S->setSwitchCaseList(SC);

    // Retain this SwitchCase, since SwitchStmt::addSwitchCase() would
    // normally retain it (but we aren't calling addSwitchCase).
    SC->Retain();
    PrevSC = SC;
  }
  return 2;
}

unsigned PCHStmtReader::VisitWhileStmt(WhileStmt *S) {
  VisitStmt(S);
  S->setConditionVariable(cast_or_null<VarDecl>(Reader.GetDecl(Record[Idx++])));
  S->setCond(cast_or_null<Expr>(StmtStack[StmtStack.size() - 2]));
  S->setBody(StmtStack.back());
  S->setWhileLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 2;
}

unsigned PCHStmtReader::VisitDoStmt(DoStmt *S) {
  VisitStmt(S);
  S->setCond(cast_or_null<Expr>(StmtStack[StmtStack.size() - 2]));
  S->setBody(StmtStack.back());
  S->setDoLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setWhileLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 2;
}

unsigned PCHStmtReader::VisitForStmt(ForStmt *S) {
  VisitStmt(S);
  S->setInit(StmtStack[StmtStack.size() - 4]);
  S->setCond(cast_or_null<Expr>(StmtStack[StmtStack.size() - 3]));
  S->setConditionVariable(cast_or_null<VarDecl>(Reader.GetDecl(Record[Idx++])));
  S->setInc(cast_or_null<Expr>(StmtStack[StmtStack.size() - 2]));
  S->setBody(StmtStack.back());
  S->setForLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setLParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 4;
}

unsigned PCHStmtReader::VisitGotoStmt(GotoStmt *S) {
  VisitStmt(S);
  Reader.SetLabelOf(S, Record[Idx++]);
  S->setGotoLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setLabelLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 0;
}

unsigned PCHStmtReader::VisitIndirectGotoStmt(IndirectGotoStmt *S) {
  VisitStmt(S);
  S->setGotoLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setStarLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setTarget(cast_or_null<Expr>(StmtStack.back()));
  return 1;
}

unsigned PCHStmtReader::VisitContinueStmt(ContinueStmt *S) {
  VisitStmt(S);
  S->setContinueLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 0;
}

unsigned PCHStmtReader::VisitBreakStmt(BreakStmt *S) {
  VisitStmt(S);
  S->setBreakLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 0;
}

unsigned PCHStmtReader::VisitReturnStmt(ReturnStmt *S) {
  VisitStmt(S);
  S->setRetValue(cast_or_null<Expr>(StmtStack.back()));
  S->setReturnLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 1;
}

unsigned PCHStmtReader::VisitDeclStmt(DeclStmt *S) {
  VisitStmt(S);
  S->setStartLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setEndLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));

  if (Idx + 1 == Record.size()) {
    // Single declaration
    S->setDeclGroup(DeclGroupRef(Reader.GetDecl(Record[Idx++])));
  } else {
    llvm::SmallVector<Decl *, 16> Decls;
    Decls.reserve(Record.size() - Idx);
    for (unsigned N = Record.size(); Idx != N; ++Idx)
      Decls.push_back(Reader.GetDecl(Record[Idx]));
    S->setDeclGroup(DeclGroupRef(DeclGroup::Create(*Reader.getContext(),
                                                   Decls.data(),
                                                   Decls.size())));
  }
  return 0;
}

unsigned PCHStmtReader::VisitAsmStmt(AsmStmt *S) {
  VisitStmt(S);
  unsigned NumOutputs = Record[Idx++];
  unsigned NumInputs = Record[Idx++];
  unsigned NumClobbers = Record[Idx++];
  S->setAsmLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setVolatile(Record[Idx++]);
  S->setSimple(Record[Idx++]);
  S->setMSAsm(Record[Idx++]);

  unsigned StackIdx
    = StmtStack.size() - (NumOutputs*2 + NumInputs*2 + NumClobbers + 1);
  S->setAsmString(cast_or_null<StringLiteral>(StmtStack[StackIdx++]));

  // Outputs and inputs
  llvm::SmallVector<std::string, 16> Names;
  llvm::SmallVector<StringLiteral*, 16> Constraints;
  llvm::SmallVector<Stmt*, 16> Exprs;
  for (unsigned I = 0, N = NumOutputs + NumInputs; I != N; ++I) {
    Names.push_back(Reader.ReadString(Record, Idx));
    Constraints.push_back(cast_or_null<StringLiteral>(StmtStack[StackIdx++]));
    Exprs.push_back(StmtStack[StackIdx++]);
  }
  S->setOutputsAndInputs(NumOutputs, NumInputs,
                         Names.data(), Constraints.data(), Exprs.data());

  // Constraints
  llvm::SmallVector<StringLiteral*, 16> Clobbers;
  for (unsigned I = 0; I != NumClobbers; ++I)
    Clobbers.push_back(cast_or_null<StringLiteral>(StmtStack[StackIdx++]));
  S->setClobbers(Clobbers.data(), NumClobbers);

  assert(StackIdx == StmtStack.size() && "Error deserializing AsmStmt");
  return NumOutputs*2 + NumInputs*2 + NumClobbers + 1;
}

unsigned PCHStmtReader::VisitExpr(Expr *E) {
  VisitStmt(E);
  E->setType(Reader.GetType(Record[Idx++]));
  E->setTypeDependent(Record[Idx++]);
  E->setValueDependent(Record[Idx++]);
  assert(Idx == NumExprFields && "Incorrect expression field count");
  return 0;
}

unsigned PCHStmtReader::VisitPredefinedExpr(PredefinedExpr *E) {
  VisitExpr(E);
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setIdentType((PredefinedExpr::IdentType)Record[Idx++]);
  return 0;
}

unsigned PCHStmtReader::VisitDeclRefExpr(DeclRefExpr *E) {
  VisitExpr(E);
  E->setDecl(cast<ValueDecl>(Reader.GetDecl(Record[Idx++])));
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  // FIXME: read qualifier
  // FIXME: read explicit template arguments
  return 0;
}

unsigned PCHStmtReader::VisitIntegerLiteral(IntegerLiteral *E) {
  VisitExpr(E);
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setValue(Reader.ReadAPInt(Record, Idx));
  return 0;
}

unsigned PCHStmtReader::VisitFloatingLiteral(FloatingLiteral *E) {
  VisitExpr(E);
  E->setValue(Reader.ReadAPFloat(Record, Idx));
  E->setExact(Record[Idx++]);
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 0;
}

unsigned PCHStmtReader::VisitImaginaryLiteral(ImaginaryLiteral *E) {
  VisitExpr(E);
  E->setSubExpr(cast<Expr>(StmtStack.back()));
  return 1;
}

unsigned PCHStmtReader::VisitStringLiteral(StringLiteral *E) {
  VisitExpr(E);
  unsigned Len = Record[Idx++];
  assert(Record[Idx] == E->getNumConcatenated() &&
         "Wrong number of concatenated tokens!");
  ++Idx;
  E->setWide(Record[Idx++]);

  // Read string data
  llvm::SmallString<16> Str(&Record[Idx], &Record[Idx] + Len);
  E->setString(*Reader.getContext(), Str.str());
  Idx += Len;

  // Read source locations
  for (unsigned I = 0, N = E->getNumConcatenated(); I != N; ++I)
    E->setStrTokenLoc(I, SourceLocation::getFromRawEncoding(Record[Idx++]));

  return 0;
}

unsigned PCHStmtReader::VisitCharacterLiteral(CharacterLiteral *E) {
  VisitExpr(E);
  E->setValue(Record[Idx++]);
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setWide(Record[Idx++]);
  return 0;
}

unsigned PCHStmtReader::VisitParenExpr(ParenExpr *E) {
  VisitExpr(E);
  E->setLParen(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParen(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setSubExpr(cast<Expr>(StmtStack.back()));
  return 1;
}

unsigned PCHStmtReader::VisitUnaryOperator(UnaryOperator *E) {
  VisitExpr(E);
  E->setSubExpr(cast<Expr>(StmtStack.back()));
  E->setOpcode((UnaryOperator::Opcode)Record[Idx++]);
  E->setOperatorLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 1;
}

unsigned PCHStmtReader::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E) {
  VisitExpr(E);
  E->setSizeof(Record[Idx++]);
  if (Record[Idx] == 0) {
    E->setArgument(cast<Expr>(StmtStack.back()));
    ++Idx;
  } else {
    E->setArgument(Reader.GetTypeSourceInfo(Record, Idx));
  }
  E->setOperatorLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return E->isArgumentType()? 0 : 1;
}

unsigned PCHStmtReader::VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
  VisitExpr(E);
  E->setLHS(cast<Expr>(StmtStack[StmtStack.size() - 2]));
  E->setRHS(cast<Expr>(StmtStack[StmtStack.size() - 1]));
  E->setRBracketLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 2;
}

unsigned PCHStmtReader::VisitCallExpr(CallExpr *E) {
  VisitExpr(E);
  E->setNumArgs(*Reader.getContext(), Record[Idx++]);
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setCallee(cast<Expr>(StmtStack[StmtStack.size() - E->getNumArgs() - 1]));
  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I)
    E->setArg(I, cast<Expr>(StmtStack[StmtStack.size() - N + I]));
  return E->getNumArgs() + 1;
}

unsigned PCHStmtReader::VisitMemberExpr(MemberExpr *E) {
  VisitExpr(E);
  E->setBase(cast<Expr>(StmtStack.back()));
  E->setMemberDecl(cast<ValueDecl>(Reader.GetDecl(Record[Idx++])));
  E->setMemberLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setArrow(Record[Idx++]);
  return 1;
}

unsigned PCHStmtReader::VisitObjCIsaExpr(ObjCIsaExpr *E) {
  VisitExpr(E);
  E->setBase(cast<Expr>(StmtStack.back()));
  E->setIsaMemberLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setArrow(Record[Idx++]);
  return 1;
}

unsigned PCHStmtReader::VisitCastExpr(CastExpr *E) {
  VisitExpr(E);
  E->setSubExpr(cast<Expr>(StmtStack.back()));
  E->setCastKind((CastExpr::CastKind)Record[Idx++]);

  return 1;
}

unsigned PCHStmtReader::VisitBinaryOperator(BinaryOperator *E) {
  VisitExpr(E);
  E->setLHS(cast<Expr>(StmtStack.end()[-2]));
  E->setRHS(cast<Expr>(StmtStack.end()[-1]));
  E->setOpcode((BinaryOperator::Opcode)Record[Idx++]);
  E->setOperatorLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 2;
}

unsigned PCHStmtReader::VisitCompoundAssignOperator(CompoundAssignOperator *E) {
  VisitBinaryOperator(E);
  E->setComputationLHSType(Reader.GetType(Record[Idx++]));
  E->setComputationResultType(Reader.GetType(Record[Idx++]));
  return 2;
}

unsigned PCHStmtReader::VisitConditionalOperator(ConditionalOperator *E) {
  VisitExpr(E);
  E->setCond(cast<Expr>(StmtStack[StmtStack.size() - 3]));
  E->setLHS(cast_or_null<Expr>(StmtStack[StmtStack.size() - 2]));
  E->setRHS(cast_or_null<Expr>(StmtStack[StmtStack.size() - 1]));
  E->setQuestionLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setColonLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 3;
}

unsigned PCHStmtReader::VisitImplicitCastExpr(ImplicitCastExpr *E) {
  VisitCastExpr(E);
  E->setLvalueCast(Record[Idx++]);
  return 1;
}

unsigned PCHStmtReader::VisitExplicitCastExpr(ExplicitCastExpr *E) {
  VisitCastExpr(E);
  E->setTypeAsWritten(Reader.GetType(Record[Idx++]));
  return 1;
}

unsigned PCHStmtReader::VisitCStyleCastExpr(CStyleCastExpr *E) {
  VisitExplicitCastExpr(E);
  E->setLParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 1;
}

unsigned PCHStmtReader::VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
  VisitExpr(E);
  E->setLParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setInitializer(cast<Expr>(StmtStack.back()));
  E->setFileScope(Record[Idx++]);
  return 1;
}

unsigned PCHStmtReader::VisitExtVectorElementExpr(ExtVectorElementExpr *E) {
  VisitExpr(E);
  E->setBase(cast<Expr>(StmtStack.back()));
  E->setAccessor(Reader.GetIdentifierInfo(Record, Idx));
  E->setAccessorLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 1;
}

unsigned PCHStmtReader::VisitInitListExpr(InitListExpr *E) {
  VisitExpr(E);
  unsigned NumInits = Record[Idx++];
  E->reserveInits(NumInits);
  for (unsigned I = 0; I != NumInits; ++I)
    E->updateInit(I,
                  cast<Expr>(StmtStack[StmtStack.size() - NumInits - 1 + I]));
  E->setSyntacticForm(cast_or_null<InitListExpr>(StmtStack.back()));
  E->setLBraceLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRBraceLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setInitializedFieldInUnion(
                      cast_or_null<FieldDecl>(Reader.GetDecl(Record[Idx++])));
  E->sawArrayRangeDesignator(Record[Idx++]);
  return NumInits + 1;
}

unsigned PCHStmtReader::VisitDesignatedInitExpr(DesignatedInitExpr *E) {
  typedef DesignatedInitExpr::Designator Designator;

  VisitExpr(E);
  unsigned NumSubExprs = Record[Idx++];
  assert(NumSubExprs == E->getNumSubExprs() && "Wrong number of subexprs");
  for (unsigned I = 0; I != NumSubExprs; ++I)
    E->setSubExpr(I, cast<Expr>(StmtStack[StmtStack.size() - NumSubExprs + I]));
  E->setEqualOrColonLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setGNUSyntax(Record[Idx++]);

  llvm::SmallVector<Designator, 4> Designators;
  while (Idx < Record.size()) {
    switch ((pch::DesignatorTypes)Record[Idx++]) {
    case pch::DESIG_FIELD_DECL: {
      FieldDecl *Field = cast<FieldDecl>(Reader.GetDecl(Record[Idx++]));
      SourceLocation DotLoc
        = SourceLocation::getFromRawEncoding(Record[Idx++]);
      SourceLocation FieldLoc
        = SourceLocation::getFromRawEncoding(Record[Idx++]);
      Designators.push_back(Designator(Field->getIdentifier(), DotLoc,
                                       FieldLoc));
      Designators.back().setField(Field);
      break;
    }

    case pch::DESIG_FIELD_NAME: {
      const IdentifierInfo *Name = Reader.GetIdentifierInfo(Record, Idx);
      SourceLocation DotLoc
        = SourceLocation::getFromRawEncoding(Record[Idx++]);
      SourceLocation FieldLoc
        = SourceLocation::getFromRawEncoding(Record[Idx++]);
      Designators.push_back(Designator(Name, DotLoc, FieldLoc));
      break;
    }

    case pch::DESIG_ARRAY: {
      unsigned Index = Record[Idx++];
      SourceLocation LBracketLoc
        = SourceLocation::getFromRawEncoding(Record[Idx++]);
      SourceLocation RBracketLoc
        = SourceLocation::getFromRawEncoding(Record[Idx++]);
      Designators.push_back(Designator(Index, LBracketLoc, RBracketLoc));
      break;
    }

    case pch::DESIG_ARRAY_RANGE: {
      unsigned Index = Record[Idx++];
      SourceLocation LBracketLoc
        = SourceLocation::getFromRawEncoding(Record[Idx++]);
      SourceLocation EllipsisLoc
        = SourceLocation::getFromRawEncoding(Record[Idx++]);
      SourceLocation RBracketLoc
        = SourceLocation::getFromRawEncoding(Record[Idx++]);
      Designators.push_back(Designator(Index, LBracketLoc, EllipsisLoc,
                                       RBracketLoc));
      break;
    }
    }
  }
  E->setDesignators(*Reader.getContext(), 
                    Designators.data(), Designators.size());

  return NumSubExprs;
}

unsigned PCHStmtReader::VisitImplicitValueInitExpr(ImplicitValueInitExpr *E) {
  VisitExpr(E);
  return 0;
}

unsigned PCHStmtReader::VisitVAArgExpr(VAArgExpr *E) {
  VisitExpr(E);
  E->setSubExpr(cast<Expr>(StmtStack.back()));
  E->setBuiltinLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 1;
}

unsigned PCHStmtReader::VisitAddrLabelExpr(AddrLabelExpr *E) {
  VisitExpr(E);
  E->setAmpAmpLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setLabelLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  Reader.SetLabelOf(E, Record[Idx++]);
  return 0;
}

unsigned PCHStmtReader::VisitStmtExpr(StmtExpr *E) {
  VisitExpr(E);
  E->setLParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setSubStmt(cast_or_null<CompoundStmt>(StmtStack.back()));
  return 1;
}

unsigned PCHStmtReader::VisitTypesCompatibleExpr(TypesCompatibleExpr *E) {
  VisitExpr(E);
  E->setArgType1(Reader.GetType(Record[Idx++]));
  E->setArgType2(Reader.GetType(Record[Idx++]));
  E->setBuiltinLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 0;
}

unsigned PCHStmtReader::VisitChooseExpr(ChooseExpr *E) {
  VisitExpr(E);
  E->setCond(cast<Expr>(StmtStack[StmtStack.size() - 3]));
  E->setLHS(cast_or_null<Expr>(StmtStack[StmtStack.size() - 2]));
  E->setRHS(cast_or_null<Expr>(StmtStack[StmtStack.size() - 1]));
  E->setBuiltinLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 3;
}

unsigned PCHStmtReader::VisitGNUNullExpr(GNUNullExpr *E) {
  VisitExpr(E);
  E->setTokenLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 0;
}

unsigned PCHStmtReader::VisitShuffleVectorExpr(ShuffleVectorExpr *E) {
  VisitExpr(E);
  unsigned NumExprs = Record[Idx++];
  E->setExprs(*Reader.getContext(),
              (Expr **)&StmtStack[StmtStack.size() - NumExprs], NumExprs);
  E->setBuiltinLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return NumExprs;
}

unsigned PCHStmtReader::VisitBlockExpr(BlockExpr *E) {
  VisitExpr(E);
  E->setBlockDecl(cast_or_null<BlockDecl>(Reader.GetDecl(Record[Idx++])));
  E->setHasBlockDeclRefExprs(Record[Idx++]);
  return 0;
}

unsigned PCHStmtReader::VisitBlockDeclRefExpr(BlockDeclRefExpr *E) {
  VisitExpr(E);
  E->setDecl(cast<ValueDecl>(Reader.GetDecl(Record[Idx++])));
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setByRef(Record[Idx++]);
  E->setConstQualAdded(Record[Idx++]);
  return 0;
}

//===----------------------------------------------------------------------===//
// Objective-C Expressions and Statements

unsigned PCHStmtReader::VisitObjCStringLiteral(ObjCStringLiteral *E) {
  VisitExpr(E);
  E->setString(cast<StringLiteral>(StmtStack.back()));
  E->setAtLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 1;
}

unsigned PCHStmtReader::VisitObjCEncodeExpr(ObjCEncodeExpr *E) {
  VisitExpr(E);
  E->setEncodedType(Reader.GetType(Record[Idx++]));
  E->setAtLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 0;
}

unsigned PCHStmtReader::VisitObjCSelectorExpr(ObjCSelectorExpr *E) {
  VisitExpr(E);
  E->setSelector(Reader.GetSelector(Record, Idx));
  E->setAtLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 0;
}

unsigned PCHStmtReader::VisitObjCProtocolExpr(ObjCProtocolExpr *E) {
  VisitExpr(E);
  E->setProtocol(cast<ObjCProtocolDecl>(Reader.GetDecl(Record[Idx++])));
  E->setAtLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 0;
}

unsigned PCHStmtReader::VisitObjCIvarRefExpr(ObjCIvarRefExpr *E) {
  VisitExpr(E);
  E->setDecl(cast<ObjCIvarDecl>(Reader.GetDecl(Record[Idx++])));
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setBase(cast<Expr>(StmtStack.back()));
  E->setIsArrow(Record[Idx++]);
  E->setIsFreeIvar(Record[Idx++]);
  return 1;
}

unsigned PCHStmtReader::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *E) {
  VisitExpr(E);
  E->setProperty(cast<ObjCPropertyDecl>(Reader.GetDecl(Record[Idx++])));
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setBase(cast<Expr>(StmtStack.back()));
  return 1;
}

unsigned PCHStmtReader::VisitObjCImplicitSetterGetterRefExpr(
                                      ObjCImplicitSetterGetterRefExpr *E) {
  VisitExpr(E);
  E->setGetterMethod(
                 cast_or_null<ObjCMethodDecl>(Reader.GetDecl(Record[Idx++])));
  E->setSetterMethod(
                 cast_or_null<ObjCMethodDecl>(Reader.GetDecl(Record[Idx++])));
  E->setInterfaceDecl(
              cast_or_null<ObjCInterfaceDecl>(Reader.GetDecl(Record[Idx++])));
  E->setBase(cast_or_null<Expr>(StmtStack.back()));
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setClassLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 1;
}

unsigned PCHStmtReader::VisitObjCMessageExpr(ObjCMessageExpr *E) {
  VisitExpr(E);
  E->setNumArgs(Record[Idx++]);
  E->setLeftLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRightLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setSelector(Reader.GetSelector(Record, Idx));
  E->setMethodDecl(cast_or_null<ObjCMethodDecl>(Reader.GetDecl(Record[Idx++])));

  E->setReceiver(
         cast_or_null<Expr>(StmtStack[StmtStack.size() - E->getNumArgs() - 1]));
  if (!E->getReceiver()) {
    ObjCMessageExpr::ClassInfo CI;
    CI.first = cast_or_null<ObjCInterfaceDecl>(Reader.GetDecl(Record[Idx++]));
    CI.second = Reader.GetIdentifierInfo(Record, Idx);
    E->setClassInfo(CI);
  }

  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I)
    E->setArg(I, cast<Expr>(StmtStack[StmtStack.size() - N + I]));
  return E->getNumArgs() + 1;
}

unsigned PCHStmtReader::VisitObjCSuperExpr(ObjCSuperExpr *E) {
  VisitExpr(E);
  E->setLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 0;
}

unsigned PCHStmtReader::VisitObjCForCollectionStmt(ObjCForCollectionStmt *S) {
  VisitStmt(S);
  S->setElement(cast_or_null<Stmt>(StmtStack[StmtStack.size() - 3]));
  S->setCollection(cast_or_null<Expr>(StmtStack[StmtStack.size() - 2]));
  S->setBody(cast_or_null<Stmt>(StmtStack[StmtStack.size() - 1]));
  S->setForLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 3;
}

unsigned PCHStmtReader::VisitObjCAtCatchStmt(ObjCAtCatchStmt *S) {
  VisitStmt(S);
  S->setCatchBody(cast_or_null<Stmt>(StmtStack[StmtStack.size() - 2]));
  S->setNextCatchStmt(cast_or_null<Stmt>(StmtStack[StmtStack.size() - 1]));
  S->setCatchParamDecl(cast_or_null<ParmVarDecl>(Reader.GetDecl(Record[Idx++])));
  S->setAtCatchLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 2;
}

unsigned PCHStmtReader::VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *S) {
  VisitStmt(S);
  S->setFinallyBody(StmtStack.back());
  S->setAtFinallyLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 1;
}

unsigned PCHStmtReader::VisitObjCAtTryStmt(ObjCAtTryStmt *S) {
  VisitStmt(S);
  S->setTryBody(cast_or_null<Stmt>(StmtStack[StmtStack.size() - 3]));
  S->setCatchStmts(cast_or_null<Stmt>(StmtStack[StmtStack.size() - 2]));
  S->setFinallyStmt(cast_or_null<Stmt>(StmtStack[StmtStack.size() - 1]));
  S->setAtTryLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 3;
}

unsigned PCHStmtReader::VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *S) {
  VisitStmt(S);
  S->setSynchExpr(cast_or_null<Stmt>(StmtStack[StmtStack.size() - 2]));
  S->setSynchBody(cast_or_null<Stmt>(StmtStack[StmtStack.size() - 1]));
  S->setAtSynchronizedLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 2;
}

unsigned PCHStmtReader::VisitObjCAtThrowStmt(ObjCAtThrowStmt *S) {
  VisitStmt(S);
  S->setThrowExpr(StmtStack.back());
  S->setThrowLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  return 1;
}

//===----------------------------------------------------------------------===//
// C++ Expressions and Statements

unsigned PCHStmtReader::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
  unsigned num = VisitCallExpr(E);
  E->setOperator((OverloadedOperatorKind)Record[Idx++]);
  return num;
}

unsigned PCHStmtReader::VisitCXXConstructExpr(CXXConstructExpr *E) {
  VisitExpr(E);
  E->setConstructor(cast<CXXConstructorDecl>(Reader.GetDecl(Record[Idx++])));
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setElidable(Record[Idx++]);  
  E->setRequiresZeroInitialization(Record[Idx++]);
  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I)
    E->setArg(I, cast<Expr>(StmtStack[StmtStack.size() - N + I]));
  return E->getNumArgs();
}

// Within the bitstream, expressions are stored in Reverse Polish
// Notation, with each of the subexpressions preceding the
// expression they are stored in. To evaluate expressions, we
// continue reading expressions and placing them on the stack, with
// expressions having operands removing those operands from the
// stack. Evaluation terminates when we see a STMT_STOP record, and
// the single remaining expression on the stack is our result.
Stmt *PCHReader::ReadStmt(llvm::BitstreamCursor &Cursor) {
  RecordData Record;
  unsigned Idx;
  llvm::SmallVector<Stmt *, 16> StmtStack;
  PCHStmtReader Reader(*this, Record, Idx, StmtStack);
  Stmt::EmptyShell Empty;

  while (true) {
    unsigned Code = Cursor.ReadCode();
    if (Code == llvm::bitc::END_BLOCK) {
      if (Cursor.ReadBlockEnd()) {
        Error("error at end of block in PCH file");
        return 0;
      }
      break;
    }

    if (Code == llvm::bitc::ENTER_SUBBLOCK) {
      // No known subblocks, always skip them.
      Cursor.ReadSubBlockID();
      if (Cursor.SkipBlock()) {
        Error("malformed block record in PCH file");
        return 0;
      }
      continue;
    }

    if (Code == llvm::bitc::DEFINE_ABBREV) {
      Cursor.ReadAbbrevRecord();
      continue;
    }

    Stmt *S = 0;
    Idx = 0;
    Record.clear();
    bool Finished = false;
    switch ((pch::StmtCode)Cursor.ReadRecord(Code, Record)) {
    case pch::STMT_STOP:
      Finished = true;
      break;

    case pch::STMT_NULL_PTR:
      S = 0;
      break;

    case pch::STMT_NULL:
      S = new (Context) NullStmt(Empty);
      break;

    case pch::STMT_COMPOUND:
      S = new (Context) CompoundStmt(Empty);
      break;

    case pch::STMT_CASE:
      S = new (Context) CaseStmt(Empty);
      break;

    case pch::STMT_DEFAULT:
      S = new (Context) DefaultStmt(Empty);
      break;

    case pch::STMT_LABEL:
      S = new (Context) LabelStmt(Empty);
      break;

    case pch::STMT_IF:
      S = new (Context) IfStmt(Empty);
      break;

    case pch::STMT_SWITCH:
      S = new (Context) SwitchStmt(Empty);
      break;

    case pch::STMT_WHILE:
      S = new (Context) WhileStmt(Empty);
      break;

    case pch::STMT_DO:
      S = new (Context) DoStmt(Empty);
      break;

    case pch::STMT_FOR:
      S = new (Context) ForStmt(Empty);
      break;

    case pch::STMT_GOTO:
      S = new (Context) GotoStmt(Empty);
      break;

    case pch::STMT_INDIRECT_GOTO:
      S = new (Context) IndirectGotoStmt(Empty);
      break;

    case pch::STMT_CONTINUE:
      S = new (Context) ContinueStmt(Empty);
      break;

    case pch::STMT_BREAK:
      S = new (Context) BreakStmt(Empty);
      break;

    case pch::STMT_RETURN:
      S = new (Context) ReturnStmt(Empty);
      break;

    case pch::STMT_DECL:
      S = new (Context) DeclStmt(Empty);
      break;

    case pch::STMT_ASM:
      S = new (Context) AsmStmt(Empty);
      break;

    case pch::EXPR_PREDEFINED:
      S = new (Context) PredefinedExpr(Empty);
      break;

    case pch::EXPR_DECL_REF:
      S = new (Context) DeclRefExpr(Empty);
      break;

    case pch::EXPR_INTEGER_LITERAL:
      S = new (Context) IntegerLiteral(Empty);
      break;

    case pch::EXPR_FLOATING_LITERAL:
      S = new (Context) FloatingLiteral(Empty);
      break;

    case pch::EXPR_IMAGINARY_LITERAL:
      S = new (Context) ImaginaryLiteral(Empty);
      break;

    case pch::EXPR_STRING_LITERAL:
      S = StringLiteral::CreateEmpty(*Context,
                                     Record[PCHStmtReader::NumExprFields + 1]);
      break;

    case pch::EXPR_CHARACTER_LITERAL:
      S = new (Context) CharacterLiteral(Empty);
      break;

    case pch::EXPR_PAREN:
      S = new (Context) ParenExpr(Empty);
      break;

    case pch::EXPR_UNARY_OPERATOR:
      S = new (Context) UnaryOperator(Empty);
      break;

    case pch::EXPR_SIZEOF_ALIGN_OF:
      S = new (Context) SizeOfAlignOfExpr(Empty);
      break;

    case pch::EXPR_ARRAY_SUBSCRIPT:
      S = new (Context) ArraySubscriptExpr(Empty);
      break;

    case pch::EXPR_CALL:
      S = new (Context) CallExpr(*Context, Stmt::CallExprClass, Empty);
      break;

    case pch::EXPR_MEMBER:
      S = new (Context) MemberExpr(Empty);
      break;

    case pch::EXPR_BINARY_OPERATOR:
      S = new (Context) BinaryOperator(Empty);
      break;

    case pch::EXPR_COMPOUND_ASSIGN_OPERATOR:
      S = new (Context) CompoundAssignOperator(Empty);
      break;

    case pch::EXPR_CONDITIONAL_OPERATOR:
      S = new (Context) ConditionalOperator(Empty);
      break;

    case pch::EXPR_IMPLICIT_CAST:
      S = new (Context) ImplicitCastExpr(Empty);
      break;

    case pch::EXPR_CSTYLE_CAST:
      S = new (Context) CStyleCastExpr(Empty);
      break;

    case pch::EXPR_COMPOUND_LITERAL:
      S = new (Context) CompoundLiteralExpr(Empty);
      break;

    case pch::EXPR_EXT_VECTOR_ELEMENT:
      S = new (Context) ExtVectorElementExpr(Empty);
      break;

    case pch::EXPR_INIT_LIST:
      S = new (Context) InitListExpr(Empty);
      break;

    case pch::EXPR_DESIGNATED_INIT:
      S = DesignatedInitExpr::CreateEmpty(*Context,
                                     Record[PCHStmtReader::NumExprFields] - 1);

      break;

    case pch::EXPR_IMPLICIT_VALUE_INIT:
      S = new (Context) ImplicitValueInitExpr(Empty);
      break;

    case pch::EXPR_VA_ARG:
      S = new (Context) VAArgExpr(Empty);
      break;

    case pch::EXPR_ADDR_LABEL:
      S = new (Context) AddrLabelExpr(Empty);
      break;

    case pch::EXPR_STMT:
      S = new (Context) StmtExpr(Empty);
      break;

    case pch::EXPR_TYPES_COMPATIBLE:
      S = new (Context) TypesCompatibleExpr(Empty);
      break;

    case pch::EXPR_CHOOSE:
      S = new (Context) ChooseExpr(Empty);
      break;

    case pch::EXPR_GNU_NULL:
      S = new (Context) GNUNullExpr(Empty);
      break;

    case pch::EXPR_SHUFFLE_VECTOR:
      S = new (Context) ShuffleVectorExpr(Empty);
      break;

    case pch::EXPR_BLOCK:
      S = new (Context) BlockExpr(Empty);
      break;

    case pch::EXPR_BLOCK_DECL_REF:
      S = new (Context) BlockDeclRefExpr(Empty);
      break;

    case pch::EXPR_OBJC_STRING_LITERAL:
      S = new (Context) ObjCStringLiteral(Empty);
      break;
    case pch::EXPR_OBJC_ENCODE:
      S = new (Context) ObjCEncodeExpr(Empty);
      break;
    case pch::EXPR_OBJC_SELECTOR_EXPR:
      S = new (Context) ObjCSelectorExpr(Empty);
      break;
    case pch::EXPR_OBJC_PROTOCOL_EXPR:
      S = new (Context) ObjCProtocolExpr(Empty);
      break;
    case pch::EXPR_OBJC_IVAR_REF_EXPR:
      S = new (Context) ObjCIvarRefExpr(Empty);
      break;
    case pch::EXPR_OBJC_PROPERTY_REF_EXPR:
      S = new (Context) ObjCPropertyRefExpr(Empty);
      break;
    case pch::EXPR_OBJC_KVC_REF_EXPR:
      S = new (Context) ObjCImplicitSetterGetterRefExpr(Empty);
      break;
    case pch::EXPR_OBJC_MESSAGE_EXPR:
      S = new (Context) ObjCMessageExpr(Empty);
      break;
    case pch::EXPR_OBJC_SUPER_EXPR:
      S = new (Context) ObjCSuperExpr(Empty);
      break;
    case pch::EXPR_OBJC_ISA:
      S = new (Context) ObjCIsaExpr(Empty);
      break;
    case pch::STMT_OBJC_FOR_COLLECTION:
      S = new (Context) ObjCForCollectionStmt(Empty);
      break;
    case pch::STMT_OBJC_CATCH:
      S = new (Context) ObjCAtCatchStmt(Empty);
      break;
    case pch::STMT_OBJC_FINALLY:
      S = new (Context) ObjCAtFinallyStmt(Empty);
      break;
    case pch::STMT_OBJC_AT_TRY:
      S = new (Context) ObjCAtTryStmt(Empty);
      break;
    case pch::STMT_OBJC_AT_SYNCHRONIZED:
      S = new (Context) ObjCAtSynchronizedStmt(Empty);
      break;
    case pch::STMT_OBJC_AT_THROW:
      S = new (Context) ObjCAtThrowStmt(Empty);
      break;

    case pch::EXPR_CXX_OPERATOR_CALL:
      S = new (Context) CXXOperatorCallExpr(*Context, Empty);
      break;
        
    case pch::EXPR_CXX_CONSTRUCT:
      S = new (Context) CXXConstructExpr(Empty, *Context,
                                      Record[PCHStmtReader::NumExprFields + 2]);
      break;
    }

    // We hit a STMT_STOP, so we're done with this expression.
    if (Finished)
      break;

    ++NumStatementsRead;

    if (S) {
      unsigned NumSubStmts = Reader.Visit(S);
      while (NumSubStmts > 0) {
        StmtStack.pop_back();
        --NumSubStmts;
      }
    }

    assert(Idx == Record.size() && "Invalid deserialization of statement");
    StmtStack.push_back(S);
  }
  assert(StmtStack.size() == 1 && "Extra expressions on stack!");
  return StmtStack.back();
}
