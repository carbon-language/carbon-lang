//===--- ASTReaderStmt.cpp - Stmt/Expr Deserialization ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Statement/expression deserialization.  This implements the
// ASTReader::ReadStmt method.
//
//===----------------------------------------------------------------------===//

#include "clang/Serialization/ASTReader.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/StmtVisitor.h"
using namespace clang;
using namespace clang::serialization;

namespace clang {

  class ASTStmtReader : public StmtVisitor<ASTStmtReader> {
    ASTReader &Reader;
    llvm::BitstreamCursor &DeclsCursor;
    const ASTReader::RecordData &Record;
    unsigned &Idx;

  public:
    ASTStmtReader(ASTReader &Reader, llvm::BitstreamCursor &Cursor,
                  const ASTReader::RecordData &Record, unsigned &Idx)
      : Reader(Reader), DeclsCursor(Cursor), Record(Record), Idx(Idx) { }

    /// \brief The number of record fields required for the Stmt class
    /// itself.
    static const unsigned NumStmtFields = 0;

    /// \brief The number of record fields required for the Expr class
    /// itself.
    static const unsigned NumExprFields = NumStmtFields + 3;
    
    /// \brief Read and initialize a ExplicitTemplateArgumentList structure.
    void ReadExplicitTemplateArgumentList(ExplicitTemplateArgumentList &ArgList,
                                          unsigned NumTemplateArgs);

    void VisitStmt(Stmt *S);
    void VisitNullStmt(NullStmt *S);
    void VisitCompoundStmt(CompoundStmt *S);
    void VisitSwitchCase(SwitchCase *S);
    void VisitCaseStmt(CaseStmt *S);
    void VisitDefaultStmt(DefaultStmt *S);
    void VisitLabelStmt(LabelStmt *S);
    void VisitIfStmt(IfStmt *S);
    void VisitSwitchStmt(SwitchStmt *S);
    void VisitWhileStmt(WhileStmt *S);
    void VisitDoStmt(DoStmt *S);
    void VisitForStmt(ForStmt *S);
    void VisitGotoStmt(GotoStmt *S);
    void VisitIndirectGotoStmt(IndirectGotoStmt *S);
    void VisitContinueStmt(ContinueStmt *S);
    void VisitBreakStmt(BreakStmt *S);
    void VisitReturnStmt(ReturnStmt *S);
    void VisitDeclStmt(DeclStmt *S);
    void VisitAsmStmt(AsmStmt *S);
    void VisitExpr(Expr *E);
    void VisitPredefinedExpr(PredefinedExpr *E);
    void VisitDeclRefExpr(DeclRefExpr *E);
    void VisitIntegerLiteral(IntegerLiteral *E);
    void VisitFloatingLiteral(FloatingLiteral *E);
    void VisitImaginaryLiteral(ImaginaryLiteral *E);
    void VisitStringLiteral(StringLiteral *E);
    void VisitCharacterLiteral(CharacterLiteral *E);
    void VisitParenExpr(ParenExpr *E);
    void VisitParenListExpr(ParenListExpr *E);
    void VisitUnaryOperator(UnaryOperator *E);
    void VisitOffsetOfExpr(OffsetOfExpr *E);
    void VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E);
    void VisitArraySubscriptExpr(ArraySubscriptExpr *E);
    void VisitCallExpr(CallExpr *E);
    void VisitMemberExpr(MemberExpr *E);
    void VisitCastExpr(CastExpr *E);
    void VisitBinaryOperator(BinaryOperator *E);
    void VisitCompoundAssignOperator(CompoundAssignOperator *E);
    void VisitConditionalOperator(ConditionalOperator *E);
    void VisitImplicitCastExpr(ImplicitCastExpr *E);
    void VisitExplicitCastExpr(ExplicitCastExpr *E);
    void VisitCStyleCastExpr(CStyleCastExpr *E);
    void VisitCompoundLiteralExpr(CompoundLiteralExpr *E);
    void VisitExtVectorElementExpr(ExtVectorElementExpr *E);
    void VisitInitListExpr(InitListExpr *E);
    void VisitDesignatedInitExpr(DesignatedInitExpr *E);
    void VisitImplicitValueInitExpr(ImplicitValueInitExpr *E);
    void VisitVAArgExpr(VAArgExpr *E);
    void VisitAddrLabelExpr(AddrLabelExpr *E);
    void VisitStmtExpr(StmtExpr *E);
    void VisitTypesCompatibleExpr(TypesCompatibleExpr *E);
    void VisitChooseExpr(ChooseExpr *E);
    void VisitGNUNullExpr(GNUNullExpr *E);
    void VisitShuffleVectorExpr(ShuffleVectorExpr *E);
    void VisitBlockExpr(BlockExpr *E);
    void VisitBlockDeclRefExpr(BlockDeclRefExpr *E);
    void VisitObjCStringLiteral(ObjCStringLiteral *E);
    void VisitObjCEncodeExpr(ObjCEncodeExpr *E);
    void VisitObjCSelectorExpr(ObjCSelectorExpr *E);
    void VisitObjCProtocolExpr(ObjCProtocolExpr *E);
    void VisitObjCIvarRefExpr(ObjCIvarRefExpr *E);
    void VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *E);
    void VisitObjCImplicitSetterGetterRefExpr(
                            ObjCImplicitSetterGetterRefExpr *E);
    void VisitObjCMessageExpr(ObjCMessageExpr *E);
    void VisitObjCSuperExpr(ObjCSuperExpr *E);
    void VisitObjCIsaExpr(ObjCIsaExpr *E);

    void VisitObjCForCollectionStmt(ObjCForCollectionStmt *);
    void VisitObjCAtCatchStmt(ObjCAtCatchStmt *);
    void VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *);
    void VisitObjCAtTryStmt(ObjCAtTryStmt *);
    void VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *);
    void VisitObjCAtThrowStmt(ObjCAtThrowStmt *);

    // C++ Statements
    void VisitCXXCatchStmt(CXXCatchStmt *S);
    void VisitCXXTryStmt(CXXTryStmt *S);

    void VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E);
    void VisitCXXConstructExpr(CXXConstructExpr *E);
    void VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *E);
    void VisitCXXNamedCastExpr(CXXNamedCastExpr *E);
    void VisitCXXStaticCastExpr(CXXStaticCastExpr *E);
    void VisitCXXDynamicCastExpr(CXXDynamicCastExpr *E);
    void VisitCXXReinterpretCastExpr(CXXReinterpretCastExpr *E);
    void VisitCXXConstCastExpr(CXXConstCastExpr *E);
    void VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *E);
    void VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *E);
    void VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *E);
    void VisitCXXTypeidExpr(CXXTypeidExpr *E);
    void VisitCXXThisExpr(CXXThisExpr *E);
    void VisitCXXThrowExpr(CXXThrowExpr *E);
    void VisitCXXDefaultArgExpr(CXXDefaultArgExpr *E);
    void VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *E);
    void VisitCXXBindReferenceExpr(CXXBindReferenceExpr *E);
    
    void VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E);
    void VisitCXXNewExpr(CXXNewExpr *E);
    void VisitCXXDeleteExpr(CXXDeleteExpr *E);
    void VisitCXXPseudoDestructorExpr(CXXPseudoDestructorExpr *E);
    
    void VisitCXXExprWithTemporaries(CXXExprWithTemporaries *E);
    
    void VisitCXXDependentScopeMemberExpr(CXXDependentScopeMemberExpr *E);
    void VisitDependentScopeDeclRefExpr(DependentScopeDeclRefExpr *E);
    void VisitCXXUnresolvedConstructExpr(CXXUnresolvedConstructExpr *E);

    void VisitOverloadExpr(OverloadExpr *E);
    void VisitUnresolvedMemberExpr(UnresolvedMemberExpr *E);
    void VisitUnresolvedLookupExpr(UnresolvedLookupExpr *E);

    void VisitUnaryTypeTraitExpr(UnaryTypeTraitExpr *E);
  };
}

void ASTStmtReader::
ReadExplicitTemplateArgumentList(ExplicitTemplateArgumentList &ArgList,
                                 unsigned NumTemplateArgs) {
  TemplateArgumentListInfo ArgInfo;
  ArgInfo.setLAngleLoc(Reader.ReadSourceLocation(Record, Idx));
  ArgInfo.setRAngleLoc(Reader.ReadSourceLocation(Record, Idx));
  for (unsigned i = 0; i != NumTemplateArgs; ++i)
    ArgInfo.addArgument(
        Reader.ReadTemplateArgumentLoc(DeclsCursor, Record, Idx));
  ArgList.initializeFrom(ArgInfo);
}

void ASTStmtReader::VisitStmt(Stmt *S) {
  assert(Idx == NumStmtFields && "Incorrect statement field count");
}

void ASTStmtReader::VisitNullStmt(NullStmt *S) {
  VisitStmt(S);
  S->setSemiLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitCompoundStmt(CompoundStmt *S) {
  VisitStmt(S);
  llvm::SmallVector<Stmt *, 16> Stmts;
  unsigned NumStmts = Record[Idx++];
  while (NumStmts--)
    Stmts.push_back(Reader.ReadSubStmt());
  S->setStmts(*Reader.getContext(), Stmts.data(), Stmts.size());
  S->setLBracLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setRBracLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitSwitchCase(SwitchCase *S) {
  VisitStmt(S);
  Reader.RecordSwitchCaseID(S, Record[Idx++]);
}

void ASTStmtReader::VisitCaseStmt(CaseStmt *S) {
  VisitSwitchCase(S);
  S->setLHS(Reader.ReadSubExpr());
  S->setRHS(Reader.ReadSubExpr());
  S->setSubStmt(Reader.ReadSubStmt());
  S->setCaseLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setEllipsisLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setColonLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitDefaultStmt(DefaultStmt *S) {
  VisitSwitchCase(S);
  S->setSubStmt(Reader.ReadSubStmt());
  S->setDefaultLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setColonLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitLabelStmt(LabelStmt *S) {
  VisitStmt(S);
  S->setID(Reader.GetIdentifierInfo(Record, Idx));
  S->setSubStmt(Reader.ReadSubStmt());
  S->setIdentLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  Reader.RecordLabelStmt(S, Record[Idx++]);
}

void ASTStmtReader::VisitIfStmt(IfStmt *S) {
  VisitStmt(S);
  S->setConditionVariable(*Reader.getContext(),
                          cast_or_null<VarDecl>(Reader.GetDecl(Record[Idx++])));
  S->setCond(Reader.ReadSubExpr());
  S->setThen(Reader.ReadSubStmt());
  S->setElse(Reader.ReadSubStmt());
  S->setIfLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setElseLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitSwitchStmt(SwitchStmt *S) {
  VisitStmt(S);
  S->setConditionVariable(*Reader.getContext(),
                          cast_or_null<VarDecl>(Reader.GetDecl(Record[Idx++])));
  S->setCond(Reader.ReadSubExpr());
  S->setBody(Reader.ReadSubStmt());
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
}

void ASTStmtReader::VisitWhileStmt(WhileStmt *S) {
  VisitStmt(S);
  S->setConditionVariable(*Reader.getContext(),
                          cast_or_null<VarDecl>(Reader.GetDecl(Record[Idx++])));
  S->setCond(Reader.ReadSubExpr());
  S->setBody(Reader.ReadSubStmt());
  S->setWhileLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitDoStmt(DoStmt *S) {
  VisitStmt(S);
  S->setCond(Reader.ReadSubExpr());
  S->setBody(Reader.ReadSubStmt());
  S->setDoLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setWhileLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitForStmt(ForStmt *S) {
  VisitStmt(S);
  S->setInit(Reader.ReadSubStmt());
  S->setCond(Reader.ReadSubExpr());
  S->setConditionVariable(*Reader.getContext(),
                          cast_or_null<VarDecl>(Reader.GetDecl(Record[Idx++])));
  S->setInc(Reader.ReadSubExpr());
  S->setBody(Reader.ReadSubStmt());
  S->setForLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setLParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitGotoStmt(GotoStmt *S) {
  VisitStmt(S);
  Reader.SetLabelOf(S, Record[Idx++]);
  S->setGotoLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setLabelLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitIndirectGotoStmt(IndirectGotoStmt *S) {
  VisitStmt(S);
  S->setGotoLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setStarLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setTarget(Reader.ReadSubExpr());
}

void ASTStmtReader::VisitContinueStmt(ContinueStmt *S) {
  VisitStmt(S);
  S->setContinueLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitBreakStmt(BreakStmt *S) {
  VisitStmt(S);
  S->setBreakLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitReturnStmt(ReturnStmt *S) {
  VisitStmt(S);
  S->setRetValue(Reader.ReadSubExpr());
  S->setReturnLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setNRVOCandidate(cast_or_null<VarDecl>(Reader.GetDecl(Record[Idx++])));
}

void ASTStmtReader::VisitDeclStmt(DeclStmt *S) {
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
}

void ASTStmtReader::VisitAsmStmt(AsmStmt *S) {
  VisitStmt(S);
  unsigned NumOutputs = Record[Idx++];
  unsigned NumInputs = Record[Idx++];
  unsigned NumClobbers = Record[Idx++];
  S->setAsmLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setVolatile(Record[Idx++]);
  S->setSimple(Record[Idx++]);
  S->setMSAsm(Record[Idx++]);

  S->setAsmString(cast_or_null<StringLiteral>(Reader.ReadSubStmt()));

  // Outputs and inputs
  llvm::SmallVector<IdentifierInfo *, 16> Names;
  llvm::SmallVector<StringLiteral*, 16> Constraints;
  llvm::SmallVector<Stmt*, 16> Exprs;
  for (unsigned I = 0, N = NumOutputs + NumInputs; I != N; ++I) {
    Names.push_back(Reader.GetIdentifierInfo(Record, Idx));
    Constraints.push_back(cast_or_null<StringLiteral>(Reader.ReadSubStmt()));
    Exprs.push_back(Reader.ReadSubStmt());
  }

  // Constraints
  llvm::SmallVector<StringLiteral*, 16> Clobbers;
  for (unsigned I = 0; I != NumClobbers; ++I)
    Clobbers.push_back(cast_or_null<StringLiteral>(Reader.ReadSubStmt()));

  S->setOutputsAndInputsAndClobbers(*Reader.getContext(),
                                    Names.data(), Constraints.data(), 
                                    Exprs.data(), NumOutputs, NumInputs, 
                                    Clobbers.data(), NumClobbers);
}

void ASTStmtReader::VisitExpr(Expr *E) {
  VisitStmt(E);
  E->setType(Reader.GetType(Record[Idx++]));
  E->setTypeDependent(Record[Idx++]);
  E->setValueDependent(Record[Idx++]);
  assert(Idx == NumExprFields && "Incorrect expression field count");
}

void ASTStmtReader::VisitPredefinedExpr(PredefinedExpr *E) {
  VisitExpr(E);
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setIdentType((PredefinedExpr::IdentType)Record[Idx++]);
}

void ASTStmtReader::VisitDeclRefExpr(DeclRefExpr *E) {
  VisitExpr(E);

  bool HasQualifier = Record[Idx++];
  unsigned NumTemplateArgs = Record[Idx++];
  
  E->DecoratedD.setInt((HasQualifier? DeclRefExpr::HasQualifierFlag : 0) |
      (NumTemplateArgs ? DeclRefExpr::HasExplicitTemplateArgumentListFlag : 0));
  
  if (HasQualifier) {
    E->getNameQualifier()->NNS = Reader.ReadNestedNameSpecifier(Record, Idx);
    E->getNameQualifier()->Range = Reader.ReadSourceRange(Record, Idx);
  }

  if (NumTemplateArgs)
    ReadExplicitTemplateArgumentList(E->getExplicitTemplateArgs(),
                                     NumTemplateArgs);

  E->setDecl(cast<ValueDecl>(Reader.GetDecl(Record[Idx++])));
  // FIXME: read DeclarationNameLoc.
  E->setLocation(Reader.ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitIntegerLiteral(IntegerLiteral *E) {
  VisitExpr(E);
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setValue(Reader.ReadAPInt(Record, Idx));
}

void ASTStmtReader::VisitFloatingLiteral(FloatingLiteral *E) {
  VisitExpr(E);
  E->setValue(Reader.ReadAPFloat(Record, Idx));
  E->setExact(Record[Idx++]);
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitImaginaryLiteral(ImaginaryLiteral *E) {
  VisitExpr(E);
  E->setSubExpr(Reader.ReadSubExpr());
}

void ASTStmtReader::VisitStringLiteral(StringLiteral *E) {
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
}

void ASTStmtReader::VisitCharacterLiteral(CharacterLiteral *E) {
  VisitExpr(E);
  E->setValue(Record[Idx++]);
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setWide(Record[Idx++]);
}

void ASTStmtReader::VisitParenExpr(ParenExpr *E) {
  VisitExpr(E);
  E->setLParen(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParen(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setSubExpr(Reader.ReadSubExpr());
}

void ASTStmtReader::VisitParenListExpr(ParenListExpr *E) {
  VisitExpr(E);
  unsigned NumExprs = Record[Idx++];
  E->Exprs = new (*Reader.getContext()) Stmt*[NumExprs];
  for (unsigned i = 0; i != NumExprs; ++i)
    E->Exprs[i] = Reader.ReadSubStmt();
  E->NumExprs = NumExprs;
  E->LParenLoc = Reader.ReadSourceLocation(Record, Idx);
  E->RParenLoc = Reader.ReadSourceLocation(Record, Idx);
}

void ASTStmtReader::VisitUnaryOperator(UnaryOperator *E) {
  VisitExpr(E);
  E->setSubExpr(Reader.ReadSubExpr());
  E->setOpcode((UnaryOperator::Opcode)Record[Idx++]);
  E->setOperatorLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitOffsetOfExpr(OffsetOfExpr *E) {
  typedef OffsetOfExpr::OffsetOfNode Node;
  VisitExpr(E);
  assert(E->getNumComponents() == Record[Idx]);
  ++Idx;
  assert(E->getNumExpressions() == Record[Idx]);
  ++Idx;
  E->setOperatorLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setTypeSourceInfo(Reader.GetTypeSourceInfo(DeclsCursor, Record, Idx));
  for (unsigned I = 0, N = E->getNumComponents(); I != N; ++I) {
    Node::Kind Kind = static_cast<Node::Kind>(Record[Idx++]);
    SourceLocation Start = SourceLocation::getFromRawEncoding(Record[Idx++]);
    SourceLocation End = SourceLocation::getFromRawEncoding(Record[Idx++]);
    switch (Kind) {
    case Node::Array:
      E->setComponent(I, Node(Start, Record[Idx++], End));
      break;
        
    case Node::Field:
      E->setComponent(I, 
             Node(Start,  
                  dyn_cast_or_null<FieldDecl>(Reader.GetDecl(Record[Idx++])),
                  End));
      break;

    case Node::Identifier:
      E->setComponent(I, Node(Start, Reader.GetIdentifier(Record[Idx++]), End));
      break;
        
    case Node::Base: {
      CXXBaseSpecifier *Base = new (*Reader.getContext()) CXXBaseSpecifier();
      *Base = Reader.ReadCXXBaseSpecifier(DeclsCursor, Record, Idx);
      E->setComponent(I, Node(Base));
      break;
    }
    }
  }
  
  for (unsigned I = 0, N = E->getNumExpressions(); I != N; ++I)
    E->setIndexExpr(I, Reader.ReadSubExpr());
}

void ASTStmtReader::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E) {
  VisitExpr(E);
  E->setSizeof(Record[Idx++]);
  if (Record[Idx] == 0) {
    E->setArgument(Reader.ReadSubExpr());
    ++Idx;
  } else {
    E->setArgument(Reader.GetTypeSourceInfo(DeclsCursor, Record, Idx));
  }
  E->setOperatorLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
  VisitExpr(E);
  E->setLHS(Reader.ReadSubExpr());
  E->setRHS(Reader.ReadSubExpr());
  E->setRBracketLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitCallExpr(CallExpr *E) {
  VisitExpr(E);
  E->setNumArgs(*Reader.getContext(), Record[Idx++]);
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setCallee(Reader.ReadSubExpr());
  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I)
    E->setArg(I, Reader.ReadSubExpr());
}

void ASTStmtReader::VisitMemberExpr(MemberExpr *E) {
  // Don't call VisitExpr, this is fully initialized at creation.
  assert(E->getStmtClass() == Stmt::MemberExprClass &&
         "It's a subclass, we must advance Idx!");
}

void ASTStmtReader::VisitObjCIsaExpr(ObjCIsaExpr *E) {
  VisitExpr(E);
  E->setBase(Reader.ReadSubExpr());
  E->setIsaMemberLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setArrow(Record[Idx++]);
}

void ASTStmtReader::VisitCastExpr(CastExpr *E) {
  VisitExpr(E);
  unsigned NumBaseSpecs = Record[Idx++];
  assert(NumBaseSpecs == E->path_size());
  E->setSubExpr(Reader.ReadSubExpr());
  E->setCastKind((CastExpr::CastKind)Record[Idx++]);
  CastExpr::path_iterator BaseI = E->path_begin();
  while (NumBaseSpecs--) {
    CXXBaseSpecifier *BaseSpec = new (*Reader.getContext()) CXXBaseSpecifier;
    *BaseSpec = Reader.ReadCXXBaseSpecifier(DeclsCursor, Record, Idx);
    *BaseI++ = BaseSpec;
  }
}

void ASTStmtReader::VisitBinaryOperator(BinaryOperator *E) {
  VisitExpr(E);
  E->setLHS(Reader.ReadSubExpr());
  E->setRHS(Reader.ReadSubExpr());
  E->setOpcode((BinaryOperator::Opcode)Record[Idx++]);
  E->setOperatorLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitCompoundAssignOperator(CompoundAssignOperator *E) {
  VisitBinaryOperator(E);
  E->setComputationLHSType(Reader.GetType(Record[Idx++]));
  E->setComputationResultType(Reader.GetType(Record[Idx++]));
}

void ASTStmtReader::VisitConditionalOperator(ConditionalOperator *E) {
  VisitExpr(E);
  E->setCond(Reader.ReadSubExpr());
  E->setLHS(Reader.ReadSubExpr());
  E->setRHS(Reader.ReadSubExpr());
  E->setQuestionLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setColonLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitImplicitCastExpr(ImplicitCastExpr *E) {
  VisitCastExpr(E);
  E->setValueKind(static_cast<ExprValueKind>(Record[Idx++]));
}

void ASTStmtReader::VisitExplicitCastExpr(ExplicitCastExpr *E) {
  VisitCastExpr(E);
  E->setTypeInfoAsWritten(Reader.GetTypeSourceInfo(DeclsCursor, Record, Idx));
}

void ASTStmtReader::VisitCStyleCastExpr(CStyleCastExpr *E) {
  VisitExplicitCastExpr(E);
  E->setLParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
  VisitExpr(E);
  E->setLParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setTypeSourceInfo(Reader.GetTypeSourceInfo(DeclsCursor, Record, Idx));
  E->setInitializer(Reader.ReadSubExpr());
  E->setFileScope(Record[Idx++]);
}

void ASTStmtReader::VisitExtVectorElementExpr(ExtVectorElementExpr *E) {
  VisitExpr(E);
  E->setBase(Reader.ReadSubExpr());
  E->setAccessor(Reader.GetIdentifierInfo(Record, Idx));
  E->setAccessorLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitInitListExpr(InitListExpr *E) {
  VisitExpr(E);
  unsigned NumInits = Record[Idx++];
  E->reserveInits(*Reader.getContext(), NumInits);
  for (unsigned I = 0; I != NumInits; ++I)
    E->updateInit(*Reader.getContext(), I, Reader.ReadSubExpr());
  E->setSyntacticForm(cast_or_null<InitListExpr>(Reader.ReadSubStmt()));
  E->setLBraceLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRBraceLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setInitializedFieldInUnion(
                      cast_or_null<FieldDecl>(Reader.GetDecl(Record[Idx++])));
  E->sawArrayRangeDesignator(Record[Idx++]);
}

void ASTStmtReader::VisitDesignatedInitExpr(DesignatedInitExpr *E) {
  typedef DesignatedInitExpr::Designator Designator;

  VisitExpr(E);
  unsigned NumSubExprs = Record[Idx++];
  assert(NumSubExprs == E->getNumSubExprs() && "Wrong number of subexprs");
  for (unsigned I = 0; I != NumSubExprs; ++I)
    E->setSubExpr(I, Reader.ReadSubExpr());
  E->setEqualOrColonLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setGNUSyntax(Record[Idx++]);

  llvm::SmallVector<Designator, 4> Designators;
  while (Idx < Record.size()) {
    switch ((DesignatorTypes)Record[Idx++]) {
    case DESIG_FIELD_DECL: {
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

    case DESIG_FIELD_NAME: {
      const IdentifierInfo *Name = Reader.GetIdentifierInfo(Record, Idx);
      SourceLocation DotLoc
        = SourceLocation::getFromRawEncoding(Record[Idx++]);
      SourceLocation FieldLoc
        = SourceLocation::getFromRawEncoding(Record[Idx++]);
      Designators.push_back(Designator(Name, DotLoc, FieldLoc));
      break;
    }

    case DESIG_ARRAY: {
      unsigned Index = Record[Idx++];
      SourceLocation LBracketLoc
        = SourceLocation::getFromRawEncoding(Record[Idx++]);
      SourceLocation RBracketLoc
        = SourceLocation::getFromRawEncoding(Record[Idx++]);
      Designators.push_back(Designator(Index, LBracketLoc, RBracketLoc));
      break;
    }

    case DESIG_ARRAY_RANGE: {
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
}

void ASTStmtReader::VisitImplicitValueInitExpr(ImplicitValueInitExpr *E) {
  VisitExpr(E);
}

void ASTStmtReader::VisitVAArgExpr(VAArgExpr *E) {
  VisitExpr(E);
  E->setSubExpr(Reader.ReadSubExpr());
  E->setWrittenTypeInfo(Reader.GetTypeSourceInfo(DeclsCursor, Record, Idx));
  E->setBuiltinLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitAddrLabelExpr(AddrLabelExpr *E) {
  VisitExpr(E);
  E->setAmpAmpLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setLabelLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  Reader.SetLabelOf(E, Record[Idx++]);
}

void ASTStmtReader::VisitStmtExpr(StmtExpr *E) {
  VisitExpr(E);
  E->setLParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setSubStmt(cast_or_null<CompoundStmt>(Reader.ReadSubStmt()));
}

void ASTStmtReader::VisitTypesCompatibleExpr(TypesCompatibleExpr *E) {
  VisitExpr(E);
  E->setArgTInfo1(Reader.GetTypeSourceInfo(DeclsCursor, Record, Idx));
  E->setArgTInfo2(Reader.GetTypeSourceInfo(DeclsCursor, Record, Idx));
  E->setBuiltinLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitChooseExpr(ChooseExpr *E) {
  VisitExpr(E);
  E->setCond(Reader.ReadSubExpr());
  E->setLHS(Reader.ReadSubExpr());
  E->setRHS(Reader.ReadSubExpr());
  E->setBuiltinLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitGNUNullExpr(GNUNullExpr *E) {
  VisitExpr(E);
  E->setTokenLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitShuffleVectorExpr(ShuffleVectorExpr *E) {
  VisitExpr(E);
  llvm::SmallVector<Expr *, 16> Exprs;
  unsigned NumExprs = Record[Idx++];
  while (NumExprs--)
    Exprs.push_back(Reader.ReadSubExpr());
  E->setExprs(*Reader.getContext(), Exprs.data(), Exprs.size());
  E->setBuiltinLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitBlockExpr(BlockExpr *E) {
  VisitExpr(E);
  E->setBlockDecl(cast_or_null<BlockDecl>(Reader.GetDecl(Record[Idx++])));
  E->setHasBlockDeclRefExprs(Record[Idx++]);
}

void ASTStmtReader::VisitBlockDeclRefExpr(BlockDeclRefExpr *E) {
  VisitExpr(E);
  E->setDecl(cast<ValueDecl>(Reader.GetDecl(Record[Idx++])));
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setByRef(Record[Idx++]);
  E->setConstQualAdded(Record[Idx++]);
  E->setCopyConstructorExpr(Reader.ReadSubExpr());
}

//===----------------------------------------------------------------------===//
// Objective-C Expressions and Statements

void ASTStmtReader::VisitObjCStringLiteral(ObjCStringLiteral *E) {
  VisitExpr(E);
  E->setString(cast<StringLiteral>(Reader.ReadSubStmt()));
  E->setAtLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitObjCEncodeExpr(ObjCEncodeExpr *E) {
  VisitExpr(E);
  E->setEncodedTypeSourceInfo(Reader.GetTypeSourceInfo(DeclsCursor,Record,Idx));
  E->setAtLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitObjCSelectorExpr(ObjCSelectorExpr *E) {
  VisitExpr(E);
  E->setSelector(Reader.GetSelector(Record, Idx));
  E->setAtLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitObjCProtocolExpr(ObjCProtocolExpr *E) {
  VisitExpr(E);
  E->setProtocol(cast<ObjCProtocolDecl>(Reader.GetDecl(Record[Idx++])));
  E->setAtLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitObjCIvarRefExpr(ObjCIvarRefExpr *E) {
  VisitExpr(E);
  E->setDecl(cast<ObjCIvarDecl>(Reader.GetDecl(Record[Idx++])));
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setBase(Reader.ReadSubExpr());
  E->setIsArrow(Record[Idx++]);
  E->setIsFreeIvar(Record[Idx++]);
}

void ASTStmtReader::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *E) {
  VisitExpr(E);
  E->setProperty(cast<ObjCPropertyDecl>(Reader.GetDecl(Record[Idx++])));
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setBase(Reader.ReadSubExpr());
}

void ASTStmtReader::VisitObjCImplicitSetterGetterRefExpr(
                                      ObjCImplicitSetterGetterRefExpr *E) {
  VisitExpr(E);
  E->setGetterMethod(
                 cast_or_null<ObjCMethodDecl>(Reader.GetDecl(Record[Idx++])));
  E->setSetterMethod(
                 cast_or_null<ObjCMethodDecl>(Reader.GetDecl(Record[Idx++])));
  E->setInterfaceDecl(
              cast_or_null<ObjCInterfaceDecl>(Reader.GetDecl(Record[Idx++])));
  E->setBase(Reader.ReadSubExpr());
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setClassLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitObjCMessageExpr(ObjCMessageExpr *E) {
  VisitExpr(E);
  assert(Record[Idx] == E->getNumArgs());
  ++Idx;
  ObjCMessageExpr::ReceiverKind Kind
    = static_cast<ObjCMessageExpr::ReceiverKind>(Record[Idx++]);
  switch (Kind) {
  case ObjCMessageExpr::Instance:
    E->setInstanceReceiver(Reader.ReadSubExpr());
    break;

  case ObjCMessageExpr::Class:
    E->setClassReceiver(Reader.GetTypeSourceInfo(DeclsCursor, Record, Idx));
    break;

  case ObjCMessageExpr::SuperClass:
  case ObjCMessageExpr::SuperInstance: {
    QualType T = Reader.GetType(Record[Idx++]);
    SourceLocation SuperLoc = SourceLocation::getFromRawEncoding(Record[Idx++]);
    E->setSuper(SuperLoc, T, Kind == ObjCMessageExpr::SuperInstance);
    break;
  }
  }

  assert(Kind == E->getReceiverKind());

  if (Record[Idx++])
    E->setMethodDecl(cast_or_null<ObjCMethodDecl>(Reader.GetDecl(Record[Idx++])));
  else
    E->setSelector(Reader.GetSelector(Record, Idx));

  E->setLeftLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRightLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));

  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I)
    E->setArg(I, Reader.ReadSubExpr());
}

void ASTStmtReader::VisitObjCSuperExpr(ObjCSuperExpr *E) {
  VisitExpr(E);
  E->setLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitObjCForCollectionStmt(ObjCForCollectionStmt *S) {
  VisitStmt(S);
  S->setElement(Reader.ReadSubStmt());
  S->setCollection(Reader.ReadSubExpr());
  S->setBody(Reader.ReadSubStmt());
  S->setForLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitObjCAtCatchStmt(ObjCAtCatchStmt *S) {
  VisitStmt(S);
  S->setCatchBody(Reader.ReadSubStmt());
  S->setCatchParamDecl(cast_or_null<VarDecl>(Reader.GetDecl(Record[Idx++])));
  S->setAtCatchLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  S->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *S) {
  VisitStmt(S);
  S->setFinallyBody(Reader.ReadSubStmt());
  S->setAtFinallyLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitObjCAtTryStmt(ObjCAtTryStmt *S) {
  VisitStmt(S);
  assert(Record[Idx] == S->getNumCatchStmts());
  ++Idx;
  bool HasFinally = Record[Idx++];
  S->setTryBody(Reader.ReadSubStmt());
  for (unsigned I = 0, N = S->getNumCatchStmts(); I != N; ++I)
    S->setCatchStmt(I, cast_or_null<ObjCAtCatchStmt>(Reader.ReadSubStmt()));

  if (HasFinally)
    S->setFinallyStmt(Reader.ReadSubStmt());
  S->setAtTryLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *S) {
  VisitStmt(S);
  S->setSynchExpr(Reader.ReadSubStmt());
  S->setSynchBody(Reader.ReadSubStmt());
  S->setAtSynchronizedLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitObjCAtThrowStmt(ObjCAtThrowStmt *S) {
  VisitStmt(S);
  S->setThrowExpr(Reader.ReadSubStmt());
  S->setThrowLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

//===----------------------------------------------------------------------===//
// C++ Expressions and Statements
//===----------------------------------------------------------------------===//

void ASTStmtReader::VisitCXXCatchStmt(CXXCatchStmt *S) {
  VisitStmt(S);
  S->CatchLoc = Reader.ReadSourceLocation(Record, Idx);
  S->ExceptionDecl = cast_or_null<VarDecl>(Reader.GetDecl(Record[Idx++]));
  S->HandlerBlock = Reader.ReadSubStmt();
}

void ASTStmtReader::VisitCXXTryStmt(CXXTryStmt *S) {
  VisitStmt(S);
  assert(Record[Idx] == S->getNumHandlers() && "NumStmtFields is wrong ?");
  ++Idx;
  S->TryLoc = Reader.ReadSourceLocation(Record, Idx);
  S->getStmts()[0] = Reader.ReadSubStmt();
  for (unsigned i = 0, e = S->getNumHandlers(); i != e; ++i)
    S->getStmts()[i + 1] = Reader.ReadSubStmt();
}

void ASTStmtReader::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
  VisitCallExpr(E);
  E->setOperator((OverloadedOperatorKind)Record[Idx++]);
}

void ASTStmtReader::VisitCXXConstructExpr(CXXConstructExpr *E) {
  VisitExpr(E);
  E->NumArgs = Record[Idx++];
  if (E->NumArgs)
    E->Args = new (*Reader.getContext()) Stmt*[E->NumArgs];
  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I)
    E->setArg(I, Reader.ReadSubExpr());
  E->setConstructor(cast<CXXConstructorDecl>(Reader.GetDecl(Record[Idx++])));
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setElidable(Record[Idx++]);  
  E->setRequiresZeroInitialization(Record[Idx++]);
  E->setConstructionKind((CXXConstructExpr::ConstructionKind)Record[Idx++]);
}

void ASTStmtReader::VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *E) {
  VisitCXXConstructExpr(E);
  E->TyBeginLoc = Reader.ReadSourceLocation(Record, Idx);
  E->RParenLoc = Reader.ReadSourceLocation(Record, Idx);
}

void ASTStmtReader::VisitCXXNamedCastExpr(CXXNamedCastExpr *E) {
  VisitExplicitCastExpr(E);
  E->setOperatorLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitCXXStaticCastExpr(CXXStaticCastExpr *E) {
  return VisitCXXNamedCastExpr(E);
}

void ASTStmtReader::VisitCXXDynamicCastExpr(CXXDynamicCastExpr *E) {
  return VisitCXXNamedCastExpr(E);
}

void ASTStmtReader::VisitCXXReinterpretCastExpr(CXXReinterpretCastExpr *E) {
  return VisitCXXNamedCastExpr(E);
}

void ASTStmtReader::VisitCXXConstCastExpr(CXXConstCastExpr *E) {
  return VisitCXXNamedCastExpr(E);
}

void ASTStmtReader::VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *E) {
  VisitExplicitCastExpr(E);
  E->setTypeBeginLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *E) {
  VisitExpr(E);
  E->setValue(Record[Idx++]);
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *E) {
  VisitExpr(E);
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitCXXTypeidExpr(CXXTypeidExpr *E) {
  VisitExpr(E);
  E->setSourceRange(Reader.ReadSourceRange(Record, Idx));
  if (E->isTypeOperand()) { // typeid(int)
    E->setTypeOperandSourceInfo(
        Reader.GetTypeSourceInfo(DeclsCursor, Record, Idx));
    return;
  }
  
  // typeid(42+2)
  E->setExprOperand(Reader.ReadSubExpr());
}

void ASTStmtReader::VisitCXXThisExpr(CXXThisExpr *E) {
  VisitExpr(E);
  E->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setImplicit(Record[Idx++]);
}

void ASTStmtReader::VisitCXXThrowExpr(CXXThrowExpr *E) {
  VisitExpr(E);
  E->setThrowLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setSubExpr(Reader.ReadSubExpr());
}

void ASTStmtReader::VisitCXXDefaultArgExpr(CXXDefaultArgExpr *E) {
  VisitExpr(E);

  assert(Record[Idx] == E->Param.getInt() && "We messed up at creation ?");
  ++Idx; // HasOtherExprStored and SubExpr was handled during creation.
  E->Param.setPointer(cast<ParmVarDecl>(Reader.GetDecl(Record[Idx++])));
  E->Loc = Reader.ReadSourceLocation(Record, Idx);
}

void ASTStmtReader::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *E) {
  VisitExpr(E);
  E->setTemporary(Reader.ReadCXXTemporary(Record, Idx));
  E->setSubExpr(Reader.ReadSubExpr());
}

void ASTStmtReader::VisitCXXBindReferenceExpr(CXXBindReferenceExpr *E) {
  VisitExpr(E);
  E->SubExpr = Reader.ReadSubExpr();
  E->ExtendsLifetime = Record[Idx++];
  E->RequiresTemporaryCopy = Record[Idx++];
}

void ASTStmtReader::VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E) {
  VisitExpr(E);
  E->setTypeBeginLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setRParenLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitCXXNewExpr(CXXNewExpr *E) {
  VisitExpr(E);
  E->setGlobalNew(Record[Idx++]);
  E->setHasInitializer(Record[Idx++]);
  bool isArray = Record[Idx++];
  unsigned NumPlacementArgs = Record[Idx++];
  unsigned NumCtorArgs = Record[Idx++];
  E->setOperatorNew(cast_or_null<FunctionDecl>(Reader.GetDecl(Record[Idx++])));
  E->setOperatorDelete(
                    cast_or_null<FunctionDecl>(Reader.GetDecl(Record[Idx++])));
  E->setConstructor(
               cast_or_null<CXXConstructorDecl>(Reader.GetDecl(Record[Idx++])));
  SourceRange TypeIdParens;
  TypeIdParens.setBegin(SourceLocation::getFromRawEncoding(Record[Idx++]));
  TypeIdParens.setEnd(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->TypeIdParens = TypeIdParens;
  E->setStartLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  E->setEndLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  
  E->AllocateArgsArray(*Reader.getContext(), isArray, NumPlacementArgs,
                       NumCtorArgs);

  // Install all the subexpressions.
  for (CXXNewExpr::raw_arg_iterator I = E->raw_arg_begin(),e = E->raw_arg_end();
       I != e; ++I)
    *I = Reader.ReadSubStmt();
}

void ASTStmtReader::VisitCXXDeleteExpr(CXXDeleteExpr *E) {
  VisitExpr(E);
  E->setGlobalDelete(Record[Idx++]);
  E->setArrayForm(Record[Idx++]);
  E->setOperatorDelete(
                     cast_or_null<FunctionDecl>(Reader.GetDecl(Record[Idx++])));
  E->setArgument(Reader.ReadSubExpr());
  E->setStartLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void ASTStmtReader::VisitCXXPseudoDestructorExpr(CXXPseudoDestructorExpr *E) {
  VisitExpr(E);

  E->setBase(Reader.ReadSubExpr());
  E->setArrow(Record[Idx++]);
  E->setOperatorLoc(Reader.ReadSourceLocation(Record, Idx));
  E->setQualifier(Reader.ReadNestedNameSpecifier(Record, Idx));
  E->setQualifierRange(Reader.ReadSourceRange(Record, Idx));
  E->setScopeTypeInfo(Reader.GetTypeSourceInfo(DeclsCursor, Record, Idx));
  E->setColonColonLoc(Reader.ReadSourceLocation(Record, Idx));
  E->setTildeLoc(Reader.ReadSourceLocation(Record, Idx));
  
  IdentifierInfo *II = Reader.GetIdentifierInfo(Record, Idx);
  if (II)
    E->setDestroyedType(II, Reader.ReadSourceLocation(Record, Idx));
  else
    E->setDestroyedType(Reader.GetTypeSourceInfo(DeclsCursor, Record, Idx));
}

void ASTStmtReader::VisitCXXExprWithTemporaries(CXXExprWithTemporaries *E) {
  VisitExpr(E);
  unsigned NumTemps = Record[Idx++];
  if (NumTemps) {
    E->setNumTemporaries(*Reader.getContext(), NumTemps);
    for (unsigned i = 0; i != NumTemps; ++i)
      E->setTemporary(i, Reader.ReadCXXTemporary(Record, Idx));
  }
  E->setSubExpr(Reader.ReadSubExpr());
}

void
ASTStmtReader::VisitCXXDependentScopeMemberExpr(CXXDependentScopeMemberExpr *E){
  VisitExpr(E);
  
  unsigned NumTemplateArgs = Record[Idx++];
  assert((NumTemplateArgs != 0) == E->hasExplicitTemplateArgs() &&
         "Read wrong record during creation ?");
  if (E->hasExplicitTemplateArgs())
    ReadExplicitTemplateArgumentList(E->getExplicitTemplateArgs(),
                                     NumTemplateArgs);

  E->setBase(Reader.ReadSubExpr());
  E->setBaseType(Reader.GetType(Record[Idx++]));
  E->setArrow(Record[Idx++]);
  E->setOperatorLoc(Reader.ReadSourceLocation(Record, Idx));
  E->setQualifier(Reader.ReadNestedNameSpecifier(Record, Idx));
  E->setQualifierRange(Reader.ReadSourceRange(Record, Idx));
  E->setFirstQualifierFoundInScope(
                        cast_or_null<NamedDecl>(Reader.GetDecl(Record[Idx++])));
  // FIXME: read whole DeclarationNameInfo.
  E->setMember(Reader.ReadDeclarationName(Record, Idx));
  E->setMemberLoc(Reader.ReadSourceLocation(Record, Idx));
}

void
ASTStmtReader::VisitDependentScopeDeclRefExpr(DependentScopeDeclRefExpr *E) {
  VisitExpr(E);
  
  unsigned NumTemplateArgs = Record[Idx++];
  assert((NumTemplateArgs != 0) == E->hasExplicitTemplateArgs() &&
         "Read wrong record during creation ?");
  if (E->hasExplicitTemplateArgs())
    ReadExplicitTemplateArgumentList(E->getExplicitTemplateArgs(),
                                     NumTemplateArgs);

  // FIXME: read whole DeclarationNameInfo.
  E->setDeclName(Reader.ReadDeclarationName(Record, Idx));
  E->setLocation(Reader.ReadSourceLocation(Record, Idx));
  E->setQualifierRange(Reader.ReadSourceRange(Record, Idx));
  E->setQualifier(Reader.ReadNestedNameSpecifier(Record, Idx));
}

void
ASTStmtReader::VisitCXXUnresolvedConstructExpr(CXXUnresolvedConstructExpr *E) {
  VisitExpr(E);
  assert(Record[Idx] == E->arg_size() && "Read wrong record during creation ?");
  ++Idx; // NumArgs;
  for (unsigned I = 0, N = E->arg_size(); I != N; ++I)
    E->setArg(I, Reader.ReadSubExpr());
  E->setTypeBeginLoc(Reader.ReadSourceLocation(Record, Idx));
  E->setTypeAsWritten(Reader.GetType(Record[Idx++]));
  E->setLParenLoc(Reader.ReadSourceLocation(Record, Idx));
  E->setRParenLoc(Reader.ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitOverloadExpr(OverloadExpr *E) {
  VisitExpr(E);
  
  unsigned NumTemplateArgs = Record[Idx++];
  assert((NumTemplateArgs != 0) == E->hasExplicitTemplateArgs() &&
         "Read wrong record during creation ?");
  if (E->hasExplicitTemplateArgs())
    ReadExplicitTemplateArgumentList(E->getExplicitTemplateArgs(),
                                     NumTemplateArgs);

  unsigned NumDecls = Record[Idx++];
  UnresolvedSet<8> Decls;
  for (unsigned i = 0; i != NumDecls; ++i) {
    NamedDecl *D = cast<NamedDecl>(Reader.GetDecl(Record[Idx++]));
    AccessSpecifier AS = (AccessSpecifier)Record[Idx++];
    Decls.addDecl(D, AS);
  }
  E->initializeResults(*Reader.getContext(), Decls.begin(), Decls.end());

  // FIXME: read whole DeclarationNameInfo.
  E->setName(Reader.ReadDeclarationName(Record, Idx));
  E->setQualifier(Reader.ReadNestedNameSpecifier(Record, Idx));
  E->setQualifierRange(Reader.ReadSourceRange(Record, Idx));
  E->setNameLoc(Reader.ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitUnresolvedMemberExpr(UnresolvedMemberExpr *E) {
  VisitOverloadExpr(E);
  E->setArrow(Record[Idx++]);
  E->setHasUnresolvedUsing(Record[Idx++]);
  E->setBase(Reader.ReadSubExpr());
  E->setBaseType(Reader.GetType(Record[Idx++]));
  E->setOperatorLoc(Reader.ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitUnresolvedLookupExpr(UnresolvedLookupExpr *E) {
  VisitOverloadExpr(E);
  E->setRequiresADL(Record[Idx++]);
  E->setOverloaded(Record[Idx++]);
  E->setNamingClass(cast_or_null<CXXRecordDecl>(Reader.GetDecl(Record[Idx++])));
}

void ASTStmtReader::VisitUnaryTypeTraitExpr(UnaryTypeTraitExpr *E) {
  VisitExpr(E);
  E->UTT = (UnaryTypeTrait)Record[Idx++];
  SourceRange Range = Reader.ReadSourceRange(Record, Idx);
  E->Loc = Range.getBegin();
  E->RParen = Range.getEnd();
  E->QueriedType = Reader.GetType(Record[Idx++]);
}

Stmt *ASTReader::ReadStmt(llvm::BitstreamCursor &Cursor) {
  switch (ReadingKind) {
  case Read_Decl:
  case Read_Type:
    return ReadStmtFromStream(Cursor);
  case Read_Stmt:
    return ReadSubStmt();
  }

  llvm_unreachable("ReadingKind not set ?");
  return 0;
}

Expr *ASTReader::ReadExpr(llvm::BitstreamCursor &Cursor) {
  return cast_or_null<Expr>(ReadStmt(Cursor));
}

Expr *ASTReader::ReadSubExpr() {
  return cast_or_null<Expr>(ReadSubStmt());
}

// Within the bitstream, expressions are stored in Reverse Polish
// Notation, with each of the subexpressions preceding the
// expression they are stored in. Subexpressions are stored from last to first.
// To evaluate expressions, we continue reading expressions and placing them on
// the stack, with expressions having operands removing those operands from the
// stack. Evaluation terminates when we see a STMT_STOP record, and
// the single remaining expression on the stack is our result.
Stmt *ASTReader::ReadStmtFromStream(llvm::BitstreamCursor &Cursor) {

  ReadingKindTracker ReadingKind(Read_Stmt, *this);
  
#ifndef NDEBUG
  unsigned PrevNumStmts = StmtStack.size();
#endif

  RecordData Record;
  unsigned Idx;
  ASTStmtReader Reader(*this, Cursor, Record, Idx);
  Stmt::EmptyShell Empty;

  while (true) {
    unsigned Code = Cursor.ReadCode();
    if (Code == llvm::bitc::END_BLOCK) {
      if (Cursor.ReadBlockEnd()) {
        Error("error at end of block in AST file");
        return 0;
      }
      break;
    }

    if (Code == llvm::bitc::ENTER_SUBBLOCK) {
      // No known subblocks, always skip them.
      Cursor.ReadSubBlockID();
      if (Cursor.SkipBlock()) {
        Error("malformed block record in AST file");
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
    switch ((StmtCode)Cursor.ReadRecord(Code, Record)) {
    case STMT_STOP:
      Finished = true;
      break;

    case STMT_NULL_PTR:
      S = 0;
      break;

    case STMT_NULL:
      S = new (Context) NullStmt(Empty);
      break;

    case STMT_COMPOUND:
      S = new (Context) CompoundStmt(Empty);
      break;

    case STMT_CASE:
      S = new (Context) CaseStmt(Empty);
      break;

    case STMT_DEFAULT:
      S = new (Context) DefaultStmt(Empty);
      break;

    case STMT_LABEL:
      S = new (Context) LabelStmt(Empty);
      break;

    case STMT_IF:
      S = new (Context) IfStmt(Empty);
      break;

    case STMT_SWITCH:
      S = new (Context) SwitchStmt(Empty);
      break;

    case STMT_WHILE:
      S = new (Context) WhileStmt(Empty);
      break;

    case STMT_DO:
      S = new (Context) DoStmt(Empty);
      break;

    case STMT_FOR:
      S = new (Context) ForStmt(Empty);
      break;

    case STMT_GOTO:
      S = new (Context) GotoStmt(Empty);
      break;

    case STMT_INDIRECT_GOTO:
      S = new (Context) IndirectGotoStmt(Empty);
      break;

    case STMT_CONTINUE:
      S = new (Context) ContinueStmt(Empty);
      break;

    case STMT_BREAK:
      S = new (Context) BreakStmt(Empty);
      break;

    case STMT_RETURN:
      S = new (Context) ReturnStmt(Empty);
      break;

    case STMT_DECL:
      S = new (Context) DeclStmt(Empty);
      break;

    case STMT_ASM:
      S = new (Context) AsmStmt(Empty);
      break;

    case EXPR_PREDEFINED:
      S = new (Context) PredefinedExpr(Empty);
      break;

    case EXPR_DECL_REF:
      S = DeclRefExpr::CreateEmpty(*Context,
                         /*HasQualifier=*/Record[ASTStmtReader::NumExprFields],
                  /*NumTemplateArgs=*/Record[ASTStmtReader::NumExprFields + 1]);
      break;

    case EXPR_INTEGER_LITERAL:
      S = new (Context) IntegerLiteral(Empty);
      break;

    case EXPR_FLOATING_LITERAL:
      S = new (Context) FloatingLiteral(Empty);
      break;

    case EXPR_IMAGINARY_LITERAL:
      S = new (Context) ImaginaryLiteral(Empty);
      break;

    case EXPR_STRING_LITERAL:
      S = StringLiteral::CreateEmpty(*Context,
                                     Record[ASTStmtReader::NumExprFields + 1]);
      break;

    case EXPR_CHARACTER_LITERAL:
      S = new (Context) CharacterLiteral(Empty);
      break;

    case EXPR_PAREN:
      S = new (Context) ParenExpr(Empty);
      break;

    case EXPR_PAREN_LIST:
      S = new (Context) ParenListExpr(Empty);
      break;

    case EXPR_UNARY_OPERATOR:
      S = new (Context) UnaryOperator(Empty);
      break;

    case EXPR_OFFSETOF:
      S = OffsetOfExpr::CreateEmpty(*Context, 
                                    Record[ASTStmtReader::NumExprFields],
                                    Record[ASTStmtReader::NumExprFields + 1]);
      break;
        
    case EXPR_SIZEOF_ALIGN_OF:
      S = new (Context) SizeOfAlignOfExpr(Empty);
      break;

    case EXPR_ARRAY_SUBSCRIPT:
      S = new (Context) ArraySubscriptExpr(Empty);
      break;

    case EXPR_CALL:
      S = new (Context) CallExpr(*Context, Stmt::CallExprClass, Empty);
      break;

    case EXPR_MEMBER: {
      // We load everything here and fully initialize it at creation.
      // That way we can use MemberExpr::Create and don't have to duplicate its
      // logic with a MemberExpr::CreateEmpty.

      assert(Idx == 0);
      NestedNameSpecifier *NNS = 0;
      SourceRange QualifierRange;
      if (Record[Idx++]) { // HasQualifier.
        NNS = ReadNestedNameSpecifier(Record, Idx);
        QualifierRange = ReadSourceRange(Record, Idx);
      }

      TemplateArgumentListInfo ArgInfo;
      unsigned NumTemplateArgs = Record[Idx++];
      if (NumTemplateArgs) {
        ArgInfo.setLAngleLoc(ReadSourceLocation(Record, Idx));
        ArgInfo.setRAngleLoc(ReadSourceLocation(Record, Idx));
        for (unsigned i = 0; i != NumTemplateArgs; ++i)
          ArgInfo.addArgument(ReadTemplateArgumentLoc(Cursor, Record, Idx));
      }
      
      NamedDecl *FoundD = cast_or_null<NamedDecl>(GetDecl(Record[Idx++]));
      AccessSpecifier AS = (AccessSpecifier)Record[Idx++];
      DeclAccessPair FoundDecl = DeclAccessPair::make(FoundD, AS);

      QualType T = GetType(Record[Idx++]);
      Expr *Base = ReadSubExpr();
      ValueDecl *MemberD = cast<ValueDecl>(GetDecl(Record[Idx++]));
      // FIXME: read DeclarationNameLoc.
      SourceLocation MemberLoc = ReadSourceLocation(Record, Idx);
      DeclarationNameInfo MemberNameInfo(MemberD->getDeclName(), MemberLoc);
      bool IsArrow = Record[Idx++];

      S = MemberExpr::Create(*Context, Base, IsArrow, NNS, QualifierRange,
                             MemberD, FoundDecl, MemberNameInfo,
                             NumTemplateArgs ? &ArgInfo : 0, T);
      break;
    }

    case EXPR_BINARY_OPERATOR:
      S = new (Context) BinaryOperator(Empty);
      break;

    case EXPR_COMPOUND_ASSIGN_OPERATOR:
      S = new (Context) CompoundAssignOperator(Empty);
      break;

    case EXPR_CONDITIONAL_OPERATOR:
      S = new (Context) ConditionalOperator(Empty);
      break;

    case EXPR_IMPLICIT_CAST:
      S = ImplicitCastExpr::CreateEmpty(*Context,
                       /*PathSize*/ Record[ASTStmtReader::NumExprFields]);
      break;

    case EXPR_CSTYLE_CAST:
      S = CStyleCastExpr::CreateEmpty(*Context,
                       /*PathSize*/ Record[ASTStmtReader::NumExprFields]);
      break;

    case EXPR_COMPOUND_LITERAL:
      S = new (Context) CompoundLiteralExpr(Empty);
      break;

    case EXPR_EXT_VECTOR_ELEMENT:
      S = new (Context) ExtVectorElementExpr(Empty);
      break;

    case EXPR_INIT_LIST:
      S = new (Context) InitListExpr(*getContext(), Empty);
      break;

    case EXPR_DESIGNATED_INIT:
      S = DesignatedInitExpr::CreateEmpty(*Context,
                                     Record[ASTStmtReader::NumExprFields] - 1);

      break;

    case EXPR_IMPLICIT_VALUE_INIT:
      S = new (Context) ImplicitValueInitExpr(Empty);
      break;

    case EXPR_VA_ARG:
      S = new (Context) VAArgExpr(Empty);
      break;

    case EXPR_ADDR_LABEL:
      S = new (Context) AddrLabelExpr(Empty);
      break;

    case EXPR_STMT:
      S = new (Context) StmtExpr(Empty);
      break;

    case EXPR_TYPES_COMPATIBLE:
      S = new (Context) TypesCompatibleExpr(Empty);
      break;

    case EXPR_CHOOSE:
      S = new (Context) ChooseExpr(Empty);
      break;

    case EXPR_GNU_NULL:
      S = new (Context) GNUNullExpr(Empty);
      break;

    case EXPR_SHUFFLE_VECTOR:
      S = new (Context) ShuffleVectorExpr(Empty);
      break;

    case EXPR_BLOCK:
      S = new (Context) BlockExpr(Empty);
      break;

    case EXPR_BLOCK_DECL_REF:
      S = new (Context) BlockDeclRefExpr(Empty);
      break;

    case EXPR_OBJC_STRING_LITERAL:
      S = new (Context) ObjCStringLiteral(Empty);
      break;
    case EXPR_OBJC_ENCODE:
      S = new (Context) ObjCEncodeExpr(Empty);
      break;
    case EXPR_OBJC_SELECTOR_EXPR:
      S = new (Context) ObjCSelectorExpr(Empty);
      break;
    case EXPR_OBJC_PROTOCOL_EXPR:
      S = new (Context) ObjCProtocolExpr(Empty);
      break;
    case EXPR_OBJC_IVAR_REF_EXPR:
      S = new (Context) ObjCIvarRefExpr(Empty);
      break;
    case EXPR_OBJC_PROPERTY_REF_EXPR:
      S = new (Context) ObjCPropertyRefExpr(Empty);
      break;
    case EXPR_OBJC_KVC_REF_EXPR:
      S = new (Context) ObjCImplicitSetterGetterRefExpr(Empty);
      break;
    case EXPR_OBJC_MESSAGE_EXPR:
      S = ObjCMessageExpr::CreateEmpty(*Context,
                                     Record[ASTStmtReader::NumExprFields]);
      break;
    case EXPR_OBJC_SUPER_EXPR:
      S = new (Context) ObjCSuperExpr(Empty);
      break;
    case EXPR_OBJC_ISA:
      S = new (Context) ObjCIsaExpr(Empty);
      break;
    case STMT_OBJC_FOR_COLLECTION:
      S = new (Context) ObjCForCollectionStmt(Empty);
      break;
    case STMT_OBJC_CATCH:
      S = new (Context) ObjCAtCatchStmt(Empty);
      break;
    case STMT_OBJC_FINALLY:
      S = new (Context) ObjCAtFinallyStmt(Empty);
      break;
    case STMT_OBJC_AT_TRY:
      S = ObjCAtTryStmt::CreateEmpty(*Context, 
                                     Record[ASTStmtReader::NumStmtFields],
                                     Record[ASTStmtReader::NumStmtFields + 1]);
      break;
    case STMT_OBJC_AT_SYNCHRONIZED:
      S = new (Context) ObjCAtSynchronizedStmt(Empty);
      break;
    case STMT_OBJC_AT_THROW:
      S = new (Context) ObjCAtThrowStmt(Empty);
      break;

    case STMT_CXX_CATCH:
      S = new (Context) CXXCatchStmt(Empty);
      break;

    case STMT_CXX_TRY:
      S = CXXTryStmt::Create(*Context, Empty,
             /*NumHandlers=*/Record[ASTStmtReader::NumStmtFields]);
      break;

    case EXPR_CXX_OPERATOR_CALL:
      S = new (Context) CXXOperatorCallExpr(*Context, Empty);
      break;

    case EXPR_CXX_MEMBER_CALL:
      S = new (Context) CXXMemberCallExpr(*Context, Empty);
      break;
        
    case EXPR_CXX_CONSTRUCT:
      S = new (Context) CXXConstructExpr(Empty);
      break;
      
    case EXPR_CXX_TEMPORARY_OBJECT:
      S = new (Context) CXXTemporaryObjectExpr(Empty);
      break;

    case EXPR_CXX_STATIC_CAST:
      S = CXXStaticCastExpr::CreateEmpty(*Context,
                       /*PathSize*/ Record[ASTStmtReader::NumExprFields]);
      break;

    case EXPR_CXX_DYNAMIC_CAST:
      S = CXXDynamicCastExpr::CreateEmpty(*Context,
                       /*PathSize*/ Record[ASTStmtReader::NumExprFields]);
      break;

    case EXPR_CXX_REINTERPRET_CAST:
      S = CXXReinterpretCastExpr::CreateEmpty(*Context,
                       /*PathSize*/ Record[ASTStmtReader::NumExprFields]);
      break;

    case EXPR_CXX_CONST_CAST:
      S = CXXConstCastExpr::CreateEmpty(*Context);
      break;

    case EXPR_CXX_FUNCTIONAL_CAST:
      S = CXXFunctionalCastExpr::CreateEmpty(*Context,
                       /*PathSize*/ Record[ASTStmtReader::NumExprFields]);
      break;

    case EXPR_CXX_BOOL_LITERAL:
      S = new (Context) CXXBoolLiteralExpr(Empty);
      break;

    case EXPR_CXX_NULL_PTR_LITERAL:
      S = new (Context) CXXNullPtrLiteralExpr(Empty);
      break;
    case EXPR_CXX_TYPEID_EXPR:
      S = new (Context) CXXTypeidExpr(Empty, true);
      break;
    case EXPR_CXX_TYPEID_TYPE:
      S = new (Context) CXXTypeidExpr(Empty, false);
      break;
    case EXPR_CXX_THIS:
      S = new (Context) CXXThisExpr(Empty);
      break;
    case EXPR_CXX_THROW:
      S = new (Context) CXXThrowExpr(Empty);
      break;
    case EXPR_CXX_DEFAULT_ARG: {
      bool HasOtherExprStored = Record[ASTStmtReader::NumExprFields];
      if (HasOtherExprStored) {
        Expr *SubExpr = ReadSubExpr();
        S = CXXDefaultArgExpr::Create(*Context, SourceLocation(), 0, SubExpr);
      } else
        S = new (Context) CXXDefaultArgExpr(Empty);
      break;
    }
    case EXPR_CXX_BIND_TEMPORARY:
      S = new (Context) CXXBindTemporaryExpr(Empty);
      break;
    case EXPR_CXX_BIND_REFERENCE:
      S = new (Context) CXXBindReferenceExpr(Empty);
      break;

    case EXPR_CXX_SCALAR_VALUE_INIT:
      S = new (Context) CXXScalarValueInitExpr(Empty);
      break;
    case EXPR_CXX_NEW:
      S = new (Context) CXXNewExpr(Empty);
      break;
    case EXPR_CXX_DELETE:
      S = new (Context) CXXDeleteExpr(Empty);
      break;
    case EXPR_CXX_PSEUDO_DESTRUCTOR:
      S = new (Context) CXXPseudoDestructorExpr(Empty);
      break;
        
    case EXPR_CXX_EXPR_WITH_TEMPORARIES:
      S = new (Context) CXXExprWithTemporaries(Empty);
      break;
      
    case EXPR_CXX_DEPENDENT_SCOPE_MEMBER:
      S = CXXDependentScopeMemberExpr::CreateEmpty(*Context,
                      /*NumTemplateArgs=*/Record[ASTStmtReader::NumExprFields]);
      break;
      
    case EXPR_CXX_DEPENDENT_SCOPE_DECL_REF:
      S = DependentScopeDeclRefExpr::CreateEmpty(*Context,
                      /*NumTemplateArgs=*/Record[ASTStmtReader::NumExprFields]);
      break;
      
    case EXPR_CXX_UNRESOLVED_CONSTRUCT:
      S = CXXUnresolvedConstructExpr::CreateEmpty(*Context,
                              /*NumArgs=*/Record[ASTStmtReader::NumExprFields]);
      break;
      
    case EXPR_CXX_UNRESOLVED_MEMBER:
      S = UnresolvedMemberExpr::CreateEmpty(*Context,
                      /*NumTemplateArgs=*/Record[ASTStmtReader::NumExprFields]);
      break;
      
    case EXPR_CXX_UNRESOLVED_LOOKUP:
      S = UnresolvedLookupExpr::CreateEmpty(*Context,
                      /*NumTemplateArgs=*/Record[ASTStmtReader::NumExprFields]);
      break;
      
    case EXPR_CXX_UNARY_TYPE_TRAIT:
      S = new (Context) UnaryTypeTraitExpr(Empty);
      break;
    }
    
    // We hit a STMT_STOP, so we're done with this expression.
    if (Finished)
      break;

    ++NumStatementsRead;

    if (S)
      Reader.Visit(S);

    assert(Idx == Record.size() && "Invalid deserialization of statement");
    StmtStack.push_back(S);
  }

#ifndef NDEBUG
  assert(StmtStack.size() > PrevNumStmts && "Read too many sub stmts!");
  assert(StmtStack.size() == PrevNumStmts + 1 && "Extra expressions on stack!");
#endif

  return StmtStack.pop_back_val();
}
