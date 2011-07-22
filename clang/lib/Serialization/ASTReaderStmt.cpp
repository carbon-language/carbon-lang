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
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/StmtVisitor.h"
using namespace clang;
using namespace clang::serialization;

namespace clang {

  class ASTStmtReader : public StmtVisitor<ASTStmtReader> {
    typedef ASTReader::RecordData RecordData;
    
    ASTReader &Reader;
    ASTReader::PerFileData &F;
    llvm::BitstreamCursor &DeclsCursor;
    const ASTReader::RecordData &Record;
    unsigned &Idx;

    SourceLocation ReadSourceLocation(const RecordData &R, unsigned &I) {
      return Reader.ReadSourceLocation(F, R, I);
    }
    
    SourceRange ReadSourceRange(const RecordData &R, unsigned &I) {
      return Reader.ReadSourceRange(F, R, I);
    }
    
    TypeSourceInfo *GetTypeSourceInfo(const RecordData &R, unsigned &I) {
      return Reader.GetTypeSourceInfo(F, R, I);
    }
    
    serialization::DeclID ReadDeclID(const RecordData &R, unsigned &I) {
      return Reader.ReadDeclID(F, R, I);
    }
    
    Decl *ReadDecl(const RecordData &R, unsigned &I) {
      return Reader.ReadDecl(F, R, I);
    }
    
    template<typename T>
    T *ReadDeclAs(const RecordData &R, unsigned &I) {
      return Reader.ReadDeclAs<T>(F, R, I);
    }

    void ReadDeclarationNameLoc(DeclarationNameLoc &DNLoc, DeclarationName Name,
                                const ASTReader::RecordData &R, unsigned &I) {
      Reader.ReadDeclarationNameLoc(F, DNLoc, Name, R, I);
    }
    
    void ReadDeclarationNameInfo(DeclarationNameInfo &NameInfo,
                                const ASTReader::RecordData &R, unsigned &I) {
      Reader.ReadDeclarationNameInfo(F, NameInfo, R, I);
    }

  public:
    ASTStmtReader(ASTReader &Reader, ASTReader::PerFileData &F,
                  llvm::BitstreamCursor &Cursor,
                  const ASTReader::RecordData &Record, unsigned &Idx)
      : Reader(Reader), F(F), DeclsCursor(Cursor), Record(Record), Idx(Idx) { }

    /// \brief The number of record fields required for the Stmt class
    /// itself.
    static const unsigned NumStmtFields = 0;

    /// \brief The number of record fields required for the Expr class
    /// itself.
    static const unsigned NumExprFields = NumStmtFields + 7;
    
    /// \brief Read and initialize a ExplicitTemplateArgumentList structure.
    void ReadExplicitTemplateArgumentList(ExplicitTemplateArgumentList &ArgList,
                                          unsigned NumTemplateArgs);

    void VisitStmt(Stmt *S);
#define STMT(Type, Base) \
    void Visit##Type(Type *);
#include "clang/AST/StmtNodes.inc"
  };
}

void ASTStmtReader::
ReadExplicitTemplateArgumentList(ExplicitTemplateArgumentList &ArgList,
                                 unsigned NumTemplateArgs) {
  TemplateArgumentListInfo ArgInfo;
  ArgInfo.setLAngleLoc(ReadSourceLocation(Record, Idx));
  ArgInfo.setRAngleLoc(ReadSourceLocation(Record, Idx));
  for (unsigned i = 0; i != NumTemplateArgs; ++i)
    ArgInfo.addArgument(
        Reader.ReadTemplateArgumentLoc(F, Record, Idx));
  ArgList.initializeFrom(ArgInfo);
}

void ASTStmtReader::VisitStmt(Stmt *S) {
  assert(Idx == NumStmtFields && "Incorrect statement field count");
}

void ASTStmtReader::VisitNullStmt(NullStmt *S) {
  VisitStmt(S);
  S->setSemiLoc(ReadSourceLocation(Record, Idx));
  S->LeadingEmptyMacro = ReadSourceLocation(Record, Idx);
}

void ASTStmtReader::VisitCompoundStmt(CompoundStmt *S) {
  VisitStmt(S);
  llvm::SmallVector<Stmt *, 16> Stmts;
  unsigned NumStmts = Record[Idx++];
  while (NumStmts--)
    Stmts.push_back(Reader.ReadSubStmt());
  S->setStmts(*Reader.getContext(), Stmts.data(), Stmts.size());
  S->setLBracLoc(ReadSourceLocation(Record, Idx));
  S->setRBracLoc(ReadSourceLocation(Record, Idx));
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
  S->setCaseLoc(ReadSourceLocation(Record, Idx));
  S->setEllipsisLoc(ReadSourceLocation(Record, Idx));
  S->setColonLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitDefaultStmt(DefaultStmt *S) {
  VisitSwitchCase(S);
  S->setSubStmt(Reader.ReadSubStmt());
  S->setDefaultLoc(ReadSourceLocation(Record, Idx));
  S->setColonLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitLabelStmt(LabelStmt *S) {
  VisitStmt(S);
  LabelDecl *LD = ReadDeclAs<LabelDecl>(Record, Idx);
  LD->setStmt(S);
  S->setDecl(LD);
  S->setSubStmt(Reader.ReadSubStmt());
  S->setIdentLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitIfStmt(IfStmt *S) {
  VisitStmt(S);
  S->setConditionVariable(*Reader.getContext(), 
                          ReadDeclAs<VarDecl>(Record, Idx));
  S->setCond(Reader.ReadSubExpr());
  S->setThen(Reader.ReadSubStmt());
  S->setElse(Reader.ReadSubStmt());
  S->setIfLoc(ReadSourceLocation(Record, Idx));
  S->setElseLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitSwitchStmt(SwitchStmt *S) {
  VisitStmt(S);
  S->setConditionVariable(*Reader.getContext(),
                          ReadDeclAs<VarDecl>(Record, Idx));
  S->setCond(Reader.ReadSubExpr());
  S->setBody(Reader.ReadSubStmt());
  S->setSwitchLoc(ReadSourceLocation(Record, Idx));
  if (Record[Idx++])
    S->setAllEnumCasesCovered();

  SwitchCase *PrevSC = 0;
  for (unsigned N = Record.size(); Idx != N; ++Idx) {
    SwitchCase *SC = Reader.getSwitchCaseWithID(Record[Idx]);
    if (PrevSC)
      PrevSC->setNextSwitchCase(SC);
    else
      S->setSwitchCaseList(SC);

    PrevSC = SC;
  }
}

void ASTStmtReader::VisitWhileStmt(WhileStmt *S) {
  VisitStmt(S);
  S->setConditionVariable(*Reader.getContext(),
                          ReadDeclAs<VarDecl>(Record, Idx));

  S->setCond(Reader.ReadSubExpr());
  S->setBody(Reader.ReadSubStmt());
  S->setWhileLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitDoStmt(DoStmt *S) {
  VisitStmt(S);
  S->setCond(Reader.ReadSubExpr());
  S->setBody(Reader.ReadSubStmt());
  S->setDoLoc(ReadSourceLocation(Record, Idx));
  S->setWhileLoc(ReadSourceLocation(Record, Idx));
  S->setRParenLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitForStmt(ForStmt *S) {
  VisitStmt(S);
  S->setInit(Reader.ReadSubStmt());
  S->setCond(Reader.ReadSubExpr());
  S->setConditionVariable(*Reader.getContext(),
                          ReadDeclAs<VarDecl>(Record, Idx));
  S->setInc(Reader.ReadSubExpr());
  S->setBody(Reader.ReadSubStmt());
  S->setForLoc(ReadSourceLocation(Record, Idx));
  S->setLParenLoc(ReadSourceLocation(Record, Idx));
  S->setRParenLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitGotoStmt(GotoStmt *S) {
  VisitStmt(S);
  S->setLabel(ReadDeclAs<LabelDecl>(Record, Idx));
  S->setGotoLoc(ReadSourceLocation(Record, Idx));
  S->setLabelLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitIndirectGotoStmt(IndirectGotoStmt *S) {
  VisitStmt(S);
  S->setGotoLoc(ReadSourceLocation(Record, Idx));
  S->setStarLoc(ReadSourceLocation(Record, Idx));
  S->setTarget(Reader.ReadSubExpr());
}

void ASTStmtReader::VisitContinueStmt(ContinueStmt *S) {
  VisitStmt(S);
  S->setContinueLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitBreakStmt(BreakStmt *S) {
  VisitStmt(S);
  S->setBreakLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitReturnStmt(ReturnStmt *S) {
  VisitStmt(S);
  S->setRetValue(Reader.ReadSubExpr());
  S->setReturnLoc(ReadSourceLocation(Record, Idx));
  S->setNRVOCandidate(ReadDeclAs<VarDecl>(Record, Idx));
}

void ASTStmtReader::VisitDeclStmt(DeclStmt *S) {
  VisitStmt(S);
  S->setStartLoc(ReadSourceLocation(Record, Idx));
  S->setEndLoc(ReadSourceLocation(Record, Idx));

  if (Idx + 1 == Record.size()) {
    // Single declaration
    S->setDeclGroup(DeclGroupRef(ReadDecl(Record, Idx)));
  } else {
    llvm::SmallVector<Decl *, 16> Decls;
    Decls.reserve(Record.size() - Idx);    
    for (unsigned N = Record.size(); Idx != N; )
      Decls.push_back(ReadDecl(Record, Idx));
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
  S->setAsmLoc(ReadSourceLocation(Record, Idx));
  S->setRParenLoc(ReadSourceLocation(Record, Idx));
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
  E->setType(Reader.readType(F, Record, Idx));
  E->setTypeDependent(Record[Idx++]);
  E->setValueDependent(Record[Idx++]);
  E->setInstantiationDependent(Record[Idx++]);
  E->ExprBits.ContainsUnexpandedParameterPack = Record[Idx++];
  E->setValueKind(static_cast<ExprValueKind>(Record[Idx++]));
  E->setObjectKind(static_cast<ExprObjectKind>(Record[Idx++]));
  assert(Idx == NumExprFields && "Incorrect expression field count");
}

void ASTStmtReader::VisitPredefinedExpr(PredefinedExpr *E) {
  VisitExpr(E);
  E->setLocation(ReadSourceLocation(Record, Idx));
  E->setIdentType((PredefinedExpr::IdentType)Record[Idx++]);
}

void ASTStmtReader::VisitDeclRefExpr(DeclRefExpr *E) {
  VisitExpr(E);

  E->DeclRefExprBits.HasQualifier = Record[Idx++];
  E->DeclRefExprBits.HasFoundDecl = Record[Idx++];
  E->DeclRefExprBits.HasExplicitTemplateArgs = Record[Idx++];
  unsigned NumTemplateArgs = 0;
  if (E->hasExplicitTemplateArgs())
    NumTemplateArgs = Record[Idx++];

  if (E->hasQualifier())
    E->getInternalQualifierLoc()
      = Reader.ReadNestedNameSpecifierLoc(F, Record, Idx);

  if (E->hasFoundDecl())
    E->getInternalFoundDecl() = ReadDeclAs<NamedDecl>(Record, Idx);

  if (E->hasExplicitTemplateArgs())
    ReadExplicitTemplateArgumentList(E->getExplicitTemplateArgs(),
                                     NumTemplateArgs);

  E->setDecl(ReadDeclAs<ValueDecl>(Record, Idx));
  E->setLocation(ReadSourceLocation(Record, Idx));
  ReadDeclarationNameLoc(E->DNLoc, E->getDecl()->getDeclName(), Record, Idx);
}

void ASTStmtReader::VisitIntegerLiteral(IntegerLiteral *E) {
  VisitExpr(E);
  E->setLocation(ReadSourceLocation(Record, Idx));
  E->setValue(*Reader.getContext(), Reader.ReadAPInt(Record, Idx));
}

void ASTStmtReader::VisitFloatingLiteral(FloatingLiteral *E) {
  VisitExpr(E);
  E->setValue(*Reader.getContext(), Reader.ReadAPFloat(Record, Idx));
  E->setExact(Record[Idx++]);
  E->setLocation(ReadSourceLocation(Record, Idx));
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
  E->IsWide = Record[Idx++];
  E->IsPascal = Record[Idx++];

  // Read string data
  llvm::SmallString<16> Str(&Record[Idx], &Record[Idx] + Len);
  E->setString(*Reader.getContext(), Str.str());
  Idx += Len;

  // Read source locations
  for (unsigned I = 0, N = E->getNumConcatenated(); I != N; ++I)
    E->setStrTokenLoc(I, ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitCharacterLiteral(CharacterLiteral *E) {
  VisitExpr(E);
  E->setValue(Record[Idx++]);
  E->setLocation(ReadSourceLocation(Record, Idx));
  E->setWide(Record[Idx++]);
}

void ASTStmtReader::VisitParenExpr(ParenExpr *E) {
  VisitExpr(E);
  E->setLParen(ReadSourceLocation(Record, Idx));
  E->setRParen(ReadSourceLocation(Record, Idx));
  E->setSubExpr(Reader.ReadSubExpr());
}

void ASTStmtReader::VisitParenListExpr(ParenListExpr *E) {
  VisitExpr(E);
  unsigned NumExprs = Record[Idx++];
  E->Exprs = new (*Reader.getContext()) Stmt*[NumExprs];
  for (unsigned i = 0; i != NumExprs; ++i)
    E->Exprs[i] = Reader.ReadSubStmt();
  E->NumExprs = NumExprs;
  E->LParenLoc = ReadSourceLocation(Record, Idx);
  E->RParenLoc = ReadSourceLocation(Record, Idx);
}

void ASTStmtReader::VisitUnaryOperator(UnaryOperator *E) {
  VisitExpr(E);
  E->setSubExpr(Reader.ReadSubExpr());
  E->setOpcode((UnaryOperator::Opcode)Record[Idx++]);
  E->setOperatorLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitOffsetOfExpr(OffsetOfExpr *E) {
  typedef OffsetOfExpr::OffsetOfNode Node;
  VisitExpr(E);
  assert(E->getNumComponents() == Record[Idx]);
  ++Idx;
  assert(E->getNumExpressions() == Record[Idx]);
  ++Idx;
  E->setOperatorLoc(ReadSourceLocation(Record, Idx));
  E->setRParenLoc(ReadSourceLocation(Record, Idx));
  E->setTypeSourceInfo(GetTypeSourceInfo(Record, Idx));
  for (unsigned I = 0, N = E->getNumComponents(); I != N; ++I) {
    Node::Kind Kind = static_cast<Node::Kind>(Record[Idx++]);
    SourceLocation Start = ReadSourceLocation(Record, Idx);
    SourceLocation End = ReadSourceLocation(Record, Idx);
    switch (Kind) {
    case Node::Array:
      E->setComponent(I, Node(Start, Record[Idx++], End));
      break;
        
    case Node::Field:
      E->setComponent(I, Node(Start, ReadDeclAs<FieldDecl>(Record, Idx), End));
      break;

    case Node::Identifier:
      E->setComponent(I, Node(Start, Reader.GetIdentifier(Record[Idx++]), End));
      break;
        
    case Node::Base: {
      CXXBaseSpecifier *Base = new (*Reader.getContext()) CXXBaseSpecifier();
      *Base = Reader.ReadCXXBaseSpecifier(F, Record, Idx);
      E->setComponent(I, Node(Base));
      break;
    }
    }
  }
  
  for (unsigned I = 0, N = E->getNumExpressions(); I != N; ++I)
    E->setIndexExpr(I, Reader.ReadSubExpr());
}

void ASTStmtReader::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *E) {
  VisitExpr(E);
  E->setKind(static_cast<UnaryExprOrTypeTrait>(Record[Idx++]));
  if (Record[Idx] == 0) {
    E->setArgument(Reader.ReadSubExpr());
    ++Idx;
  } else {
    E->setArgument(GetTypeSourceInfo(Record, Idx));
  }
  E->setOperatorLoc(ReadSourceLocation(Record, Idx));
  E->setRParenLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
  VisitExpr(E);
  E->setLHS(Reader.ReadSubExpr());
  E->setRHS(Reader.ReadSubExpr());
  E->setRBracketLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitCallExpr(CallExpr *E) {
  VisitExpr(E);
  E->setNumArgs(*Reader.getContext(), Record[Idx++]);
  E->setRParenLoc(ReadSourceLocation(Record, Idx));
  E->setCallee(Reader.ReadSubExpr());
  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I)
    E->setArg(I, Reader.ReadSubExpr());
}

void ASTStmtReader::VisitCXXMemberCallExpr(CXXMemberCallExpr *E) {
  VisitCallExpr(E);
}

void ASTStmtReader::VisitMemberExpr(MemberExpr *E) {
  // Don't call VisitExpr, this is fully initialized at creation.
  assert(E->getStmtClass() == Stmt::MemberExprClass &&
         "It's a subclass, we must advance Idx!");
}

void ASTStmtReader::VisitObjCIsaExpr(ObjCIsaExpr *E) {
  VisitExpr(E);
  E->setBase(Reader.ReadSubExpr());
  E->setIsaMemberLoc(ReadSourceLocation(Record, Idx));
  E->setArrow(Record[Idx++]);
}

void ASTStmtReader::
VisitObjCIndirectCopyRestoreExpr(ObjCIndirectCopyRestoreExpr *E) {
  VisitExpr(E);
  E->Operand = Reader.ReadSubExpr();
  E->setShouldCopy(Record[Idx++]);
}

void ASTStmtReader::VisitObjCBridgedCastExpr(ObjCBridgedCastExpr *E) {
  VisitExplicitCastExpr(E);
  E->LParenLoc = ReadSourceLocation(Record, Idx);
  E->BridgeKeywordLoc = ReadSourceLocation(Record, Idx);
  E->Kind = Record[Idx++];
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
    *BaseSpec = Reader.ReadCXXBaseSpecifier(F, Record, Idx);
    *BaseI++ = BaseSpec;
  }
}

void ASTStmtReader::VisitBinaryOperator(BinaryOperator *E) {
  VisitExpr(E);
  E->setLHS(Reader.ReadSubExpr());
  E->setRHS(Reader.ReadSubExpr());
  E->setOpcode((BinaryOperator::Opcode)Record[Idx++]);
  E->setOperatorLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitCompoundAssignOperator(CompoundAssignOperator *E) {
  VisitBinaryOperator(E);
  E->setComputationLHSType(Reader.readType(F, Record, Idx));
  E->setComputationResultType(Reader.readType(F, Record, Idx));
}

void ASTStmtReader::VisitConditionalOperator(ConditionalOperator *E) {
  VisitExpr(E);
  E->SubExprs[ConditionalOperator::COND] = Reader.ReadSubExpr();
  E->SubExprs[ConditionalOperator::LHS] = Reader.ReadSubExpr();
  E->SubExprs[ConditionalOperator::RHS] = Reader.ReadSubExpr();
  E->QuestionLoc = ReadSourceLocation(Record, Idx);
  E->ColonLoc = ReadSourceLocation(Record, Idx);
}

void
ASTStmtReader::VisitBinaryConditionalOperator(BinaryConditionalOperator *E) {
  VisitExpr(E);
  E->OpaqueValue = cast<OpaqueValueExpr>(Reader.ReadSubExpr());
  E->SubExprs[BinaryConditionalOperator::COMMON] = Reader.ReadSubExpr();
  E->SubExprs[BinaryConditionalOperator::COND] = Reader.ReadSubExpr();
  E->SubExprs[BinaryConditionalOperator::LHS] = Reader.ReadSubExpr();
  E->SubExprs[BinaryConditionalOperator::RHS] = Reader.ReadSubExpr();
  E->QuestionLoc = ReadSourceLocation(Record, Idx);
  E->ColonLoc = ReadSourceLocation(Record, Idx);

  E->getOpaqueValue()->setSourceExpr(E->getCommon());
}

void ASTStmtReader::VisitImplicitCastExpr(ImplicitCastExpr *E) {
  VisitCastExpr(E);
}

void ASTStmtReader::VisitExplicitCastExpr(ExplicitCastExpr *E) {
  VisitCastExpr(E);
  E->setTypeInfoAsWritten(GetTypeSourceInfo(Record, Idx));
}

void ASTStmtReader::VisitCStyleCastExpr(CStyleCastExpr *E) {
  VisitExplicitCastExpr(E);
  E->setLParenLoc(ReadSourceLocation(Record, Idx));
  E->setRParenLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
  VisitExpr(E);
  E->setLParenLoc(ReadSourceLocation(Record, Idx));
  E->setTypeSourceInfo(GetTypeSourceInfo(Record, Idx));
  E->setInitializer(Reader.ReadSubExpr());
  E->setFileScope(Record[Idx++]);
}

void ASTStmtReader::VisitExtVectorElementExpr(ExtVectorElementExpr *E) {
  VisitExpr(E);
  E->setBase(Reader.ReadSubExpr());
  E->setAccessor(Reader.GetIdentifierInfo(Record, Idx));
  E->setAccessorLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitInitListExpr(InitListExpr *E) {
  VisitExpr(E);
  E->setSyntacticForm(cast_or_null<InitListExpr>(Reader.ReadSubStmt()));
  E->setLBraceLoc(ReadSourceLocation(Record, Idx));
  E->setRBraceLoc(ReadSourceLocation(Record, Idx));
  bool isArrayFiller = Record[Idx++];
  Expr *filler = 0;
  if (isArrayFiller) {
    filler = Reader.ReadSubExpr();
    E->ArrayFillerOrUnionFieldInit = filler;
  } else
    E->ArrayFillerOrUnionFieldInit = ReadDeclAs<FieldDecl>(Record, Idx);
  E->sawArrayRangeDesignator(Record[Idx++]);
  unsigned NumInits = Record[Idx++];
  E->reserveInits(*Reader.getContext(), NumInits);
  if (isArrayFiller) {
    for (unsigned I = 0; I != NumInits; ++I) {
      Expr *init = Reader.ReadSubExpr();
      E->updateInit(*Reader.getContext(), I, init ? init : filler);
    }
  } else {
    for (unsigned I = 0; I != NumInits; ++I)
      E->updateInit(*Reader.getContext(), I, Reader.ReadSubExpr());
  }
}

void ASTStmtReader::VisitDesignatedInitExpr(DesignatedInitExpr *E) {
  typedef DesignatedInitExpr::Designator Designator;

  VisitExpr(E);
  unsigned NumSubExprs = Record[Idx++];
  assert(NumSubExprs == E->getNumSubExprs() && "Wrong number of subexprs");
  for (unsigned I = 0; I != NumSubExprs; ++I)
    E->setSubExpr(I, Reader.ReadSubExpr());
  E->setEqualOrColonLoc(ReadSourceLocation(Record, Idx));
  E->setGNUSyntax(Record[Idx++]);

  llvm::SmallVector<Designator, 4> Designators;
  while (Idx < Record.size()) {
    switch ((DesignatorTypes)Record[Idx++]) {
    case DESIG_FIELD_DECL: {
      FieldDecl *Field = ReadDeclAs<FieldDecl>(Record, Idx);
      SourceLocation DotLoc
        = ReadSourceLocation(Record, Idx);
      SourceLocation FieldLoc
        = ReadSourceLocation(Record, Idx);
      Designators.push_back(Designator(Field->getIdentifier(), DotLoc,
                                       FieldLoc));
      Designators.back().setField(Field);
      break;
    }

    case DESIG_FIELD_NAME: {
      const IdentifierInfo *Name = Reader.GetIdentifierInfo(Record, Idx);
      SourceLocation DotLoc
        = ReadSourceLocation(Record, Idx);
      SourceLocation FieldLoc
        = ReadSourceLocation(Record, Idx);
      Designators.push_back(Designator(Name, DotLoc, FieldLoc));
      break;
    }

    case DESIG_ARRAY: {
      unsigned Index = Record[Idx++];
      SourceLocation LBracketLoc
        = ReadSourceLocation(Record, Idx);
      SourceLocation RBracketLoc
        = ReadSourceLocation(Record, Idx);
      Designators.push_back(Designator(Index, LBracketLoc, RBracketLoc));
      break;
    }

    case DESIG_ARRAY_RANGE: {
      unsigned Index = Record[Idx++];
      SourceLocation LBracketLoc
        = ReadSourceLocation(Record, Idx);
      SourceLocation EllipsisLoc
        = ReadSourceLocation(Record, Idx);
      SourceLocation RBracketLoc
        = ReadSourceLocation(Record, Idx);
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
  E->setWrittenTypeInfo(GetTypeSourceInfo(Record, Idx));
  E->setBuiltinLoc(ReadSourceLocation(Record, Idx));
  E->setRParenLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitAddrLabelExpr(AddrLabelExpr *E) {
  VisitExpr(E);
  E->setAmpAmpLoc(ReadSourceLocation(Record, Idx));
  E->setLabelLoc(ReadSourceLocation(Record, Idx));
  E->setLabel(ReadDeclAs<LabelDecl>(Record, Idx));
}

void ASTStmtReader::VisitStmtExpr(StmtExpr *E) {
  VisitExpr(E);
  E->setLParenLoc(ReadSourceLocation(Record, Idx));
  E->setRParenLoc(ReadSourceLocation(Record, Idx));
  E->setSubStmt(cast_or_null<CompoundStmt>(Reader.ReadSubStmt()));
}

void ASTStmtReader::VisitChooseExpr(ChooseExpr *E) {
  VisitExpr(E);
  E->setCond(Reader.ReadSubExpr());
  E->setLHS(Reader.ReadSubExpr());
  E->setRHS(Reader.ReadSubExpr());
  E->setBuiltinLoc(ReadSourceLocation(Record, Idx));
  E->setRParenLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitGNUNullExpr(GNUNullExpr *E) {
  VisitExpr(E);
  E->setTokenLocation(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitShuffleVectorExpr(ShuffleVectorExpr *E) {
  VisitExpr(E);
  llvm::SmallVector<Expr *, 16> Exprs;
  unsigned NumExprs = Record[Idx++];
  while (NumExprs--)
    Exprs.push_back(Reader.ReadSubExpr());
  E->setExprs(*Reader.getContext(), Exprs.data(), Exprs.size());
  E->setBuiltinLoc(ReadSourceLocation(Record, Idx));
  E->setRParenLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitBlockExpr(BlockExpr *E) {
  VisitExpr(E);
  E->setBlockDecl(ReadDeclAs<BlockDecl>(Record, Idx));
}

void ASTStmtReader::VisitBlockDeclRefExpr(BlockDeclRefExpr *E) {
  VisitExpr(E);
  E->setDecl(ReadDeclAs<VarDecl>(Record, Idx));
  E->setLocation(ReadSourceLocation(Record, Idx));
  E->setByRef(Record[Idx++]);
  E->setConstQualAdded(Record[Idx++]);
}

void ASTStmtReader::VisitGenericSelectionExpr(GenericSelectionExpr *E) {
  VisitExpr(E);
  E->NumAssocs = Record[Idx++];
  E->AssocTypes = new (*Reader.getContext()) TypeSourceInfo*[E->NumAssocs];
  E->SubExprs =
   new(*Reader.getContext()) Stmt*[GenericSelectionExpr::END_EXPR+E->NumAssocs];

  E->SubExprs[GenericSelectionExpr::CONTROLLING] = Reader.ReadSubExpr();
  for (unsigned I = 0, N = E->getNumAssocs(); I != N; ++I) {
    E->AssocTypes[I] = GetTypeSourceInfo(Record, Idx);
    E->SubExprs[GenericSelectionExpr::END_EXPR+I] = Reader.ReadSubExpr();
  }
  E->ResultIndex = Record[Idx++];

  E->GenericLoc = ReadSourceLocation(Record, Idx);
  E->DefaultLoc = ReadSourceLocation(Record, Idx);
  E->RParenLoc = ReadSourceLocation(Record, Idx);
}

//===----------------------------------------------------------------------===//
// Objective-C Expressions and Statements

void ASTStmtReader::VisitObjCStringLiteral(ObjCStringLiteral *E) {
  VisitExpr(E);
  E->setString(cast<StringLiteral>(Reader.ReadSubStmt()));
  E->setAtLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitObjCEncodeExpr(ObjCEncodeExpr *E) {
  VisitExpr(E);
  E->setEncodedTypeSourceInfo(GetTypeSourceInfo(Record, Idx));
  E->setAtLoc(ReadSourceLocation(Record, Idx));
  E->setRParenLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitObjCSelectorExpr(ObjCSelectorExpr *E) {
  VisitExpr(E);
  E->setSelector(Reader.GetSelector(Record, Idx));
  E->setAtLoc(ReadSourceLocation(Record, Idx));
  E->setRParenLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitObjCProtocolExpr(ObjCProtocolExpr *E) {
  VisitExpr(E);
  E->setProtocol(ReadDeclAs<ObjCProtocolDecl>(Record, Idx));
  E->setAtLoc(ReadSourceLocation(Record, Idx));
  E->setRParenLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitObjCIvarRefExpr(ObjCIvarRefExpr *E) {
  VisitExpr(E);
  E->setDecl(ReadDeclAs<ObjCIvarDecl>(Record, Idx));
  E->setLocation(ReadSourceLocation(Record, Idx));
  E->setBase(Reader.ReadSubExpr());
  E->setIsArrow(Record[Idx++]);
  E->setIsFreeIvar(Record[Idx++]);
}

void ASTStmtReader::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *E) {
  VisitExpr(E);
  bool Implicit = Record[Idx++] != 0;
  if (Implicit) {
    ObjCMethodDecl *Getter = ReadDeclAs<ObjCMethodDecl>(Record, Idx);
    ObjCMethodDecl *Setter = ReadDeclAs<ObjCMethodDecl>(Record, Idx);
    E->setImplicitProperty(Getter, Setter);
  } else {
    E->setExplicitProperty(ReadDeclAs<ObjCPropertyDecl>(Record, Idx));
  }
  E->setLocation(ReadSourceLocation(Record, Idx));
  E->setReceiverLocation(ReadSourceLocation(Record, Idx));
  switch (Record[Idx++]) {
  case 0:
    E->setBase(Reader.ReadSubExpr());
    break;
  case 1:
    E->setSuperReceiver(Reader.readType(F, Record, Idx));
    break;
  case 2:
    E->setClassReceiver(ReadDeclAs<ObjCInterfaceDecl>(Record, Idx));
    break;
  }
}

void ASTStmtReader::VisitObjCMessageExpr(ObjCMessageExpr *E) {
  VisitExpr(E);
  assert(Record[Idx] == E->getNumArgs());
  ++Idx;
  E->setDelegateInitCall(Record[Idx++]);
  ObjCMessageExpr::ReceiverKind Kind
    = static_cast<ObjCMessageExpr::ReceiverKind>(Record[Idx++]);
  switch (Kind) {
  case ObjCMessageExpr::Instance:
    E->setInstanceReceiver(Reader.ReadSubExpr());
    break;

  case ObjCMessageExpr::Class:
    E->setClassReceiver(GetTypeSourceInfo(Record, Idx));
    break;

  case ObjCMessageExpr::SuperClass:
  case ObjCMessageExpr::SuperInstance: {
    QualType T = Reader.readType(F, Record, Idx);
    SourceLocation SuperLoc = ReadSourceLocation(Record, Idx);
    E->setSuper(SuperLoc, T, Kind == ObjCMessageExpr::SuperInstance);
    break;
  }
  }

  assert(Kind == E->getReceiverKind());

  if (Record[Idx++])
    E->setMethodDecl(ReadDeclAs<ObjCMethodDecl>(Record, Idx));
  else
    E->setSelector(Reader.GetSelector(Record, Idx));

  E->LBracLoc = ReadSourceLocation(Record, Idx);
  E->RBracLoc = ReadSourceLocation(Record, Idx);
  E->SelectorLoc = ReadSourceLocation(Record, Idx);

  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I)
    E->setArg(I, Reader.ReadSubExpr());
}

void ASTStmtReader::VisitObjCForCollectionStmt(ObjCForCollectionStmt *S) {
  VisitStmt(S);
  S->setElement(Reader.ReadSubStmt());
  S->setCollection(Reader.ReadSubExpr());
  S->setBody(Reader.ReadSubStmt());
  S->setForLoc(ReadSourceLocation(Record, Idx));
  S->setRParenLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitObjCAtCatchStmt(ObjCAtCatchStmt *S) {
  VisitStmt(S);
  S->setCatchBody(Reader.ReadSubStmt());
  S->setCatchParamDecl(ReadDeclAs<VarDecl>(Record, Idx));
  S->setAtCatchLoc(ReadSourceLocation(Record, Idx));
  S->setRParenLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *S) {
  VisitStmt(S);
  S->setFinallyBody(Reader.ReadSubStmt());
  S->setAtFinallyLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitObjCAutoreleasePoolStmt(ObjCAutoreleasePoolStmt *S) {
  VisitStmt(S);
  S->setSubStmt(Reader.ReadSubStmt());
  S->setAtLoc(ReadSourceLocation(Record, Idx));
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
  S->setAtTryLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *S) {
  VisitStmt(S);
  S->setSynchExpr(Reader.ReadSubStmt());
  S->setSynchBody(Reader.ReadSubStmt());
  S->setAtSynchronizedLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitObjCAtThrowStmt(ObjCAtThrowStmt *S) {
  VisitStmt(S);
  S->setThrowExpr(Reader.ReadSubStmt());
  S->setThrowLoc(ReadSourceLocation(Record, Idx));
}

//===----------------------------------------------------------------------===//
// C++ Expressions and Statements
//===----------------------------------------------------------------------===//

void ASTStmtReader::VisitCXXCatchStmt(CXXCatchStmt *S) {
  VisitStmt(S);
  S->CatchLoc = ReadSourceLocation(Record, Idx);
  S->ExceptionDecl = ReadDeclAs<VarDecl>(Record, Idx);
  S->HandlerBlock = Reader.ReadSubStmt();
}

void ASTStmtReader::VisitCXXTryStmt(CXXTryStmt *S) {
  VisitStmt(S);
  assert(Record[Idx] == S->getNumHandlers() && "NumStmtFields is wrong ?");
  ++Idx;
  S->TryLoc = ReadSourceLocation(Record, Idx);
  S->getStmts()[0] = Reader.ReadSubStmt();
  for (unsigned i = 0, e = S->getNumHandlers(); i != e; ++i)
    S->getStmts()[i + 1] = Reader.ReadSubStmt();
}

void ASTStmtReader::VisitCXXForRangeStmt(CXXForRangeStmt *S) {
  VisitStmt(S);
  S->setForLoc(ReadSourceLocation(Record, Idx));
  S->setColonLoc(ReadSourceLocation(Record, Idx));
  S->setRParenLoc(ReadSourceLocation(Record, Idx));
  S->setRangeStmt(Reader.ReadSubStmt());
  S->setBeginEndStmt(Reader.ReadSubStmt());
  S->setCond(Reader.ReadSubExpr());
  S->setInc(Reader.ReadSubExpr());
  S->setLoopVarStmt(Reader.ReadSubStmt());
  S->setBody(Reader.ReadSubStmt());
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
  E->setConstructor(ReadDeclAs<CXXConstructorDecl>(Record, Idx));
  E->setLocation(ReadSourceLocation(Record, Idx));
  E->setElidable(Record[Idx++]);  
  E->setRequiresZeroInitialization(Record[Idx++]);
  E->setConstructionKind((CXXConstructExpr::ConstructionKind)Record[Idx++]);
  E->ParenRange = ReadSourceRange(Record, Idx);
}

void ASTStmtReader::VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *E) {
  VisitCXXConstructExpr(E);
  E->Type = GetTypeSourceInfo(Record, Idx);
}

void ASTStmtReader::VisitCXXNamedCastExpr(CXXNamedCastExpr *E) {
  VisitExplicitCastExpr(E);
  SourceRange R = ReadSourceRange(Record, Idx);
  E->Loc = R.getBegin();
  E->RParenLoc = R.getEnd();
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
  E->setTypeBeginLoc(ReadSourceLocation(Record, Idx));
  E->setRParenLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *E) {
  VisitExpr(E);
  E->setValue(Record[Idx++]);
  E->setLocation(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *E) {
  VisitExpr(E);
  E->setLocation(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitCXXTypeidExpr(CXXTypeidExpr *E) {
  VisitExpr(E);
  E->setSourceRange(ReadSourceRange(Record, Idx));
  if (E->isTypeOperand()) { // typeid(int)
    E->setTypeOperandSourceInfo(
        GetTypeSourceInfo(Record, Idx));
    return;
  }
  
  // typeid(42+2)
  E->setExprOperand(Reader.ReadSubExpr());
}

void ASTStmtReader::VisitCXXThisExpr(CXXThisExpr *E) {
  VisitExpr(E);
  E->setLocation(ReadSourceLocation(Record, Idx));
  E->setImplicit(Record[Idx++]);
}

void ASTStmtReader::VisitCXXThrowExpr(CXXThrowExpr *E) {
  VisitExpr(E);
  E->ThrowLoc = ReadSourceLocation(Record, Idx);
  E->Op = Reader.ReadSubExpr();
  E->IsThrownVariableInScope = Record[Idx++];
}

void ASTStmtReader::VisitCXXDefaultArgExpr(CXXDefaultArgExpr *E) {
  VisitExpr(E);

  assert(Record[Idx] == E->Param.getInt() && "We messed up at creation ?");
  ++Idx; // HasOtherExprStored and SubExpr was handled during creation.
  E->Param.setPointer(ReadDeclAs<ParmVarDecl>(Record, Idx));
  E->Loc = ReadSourceLocation(Record, Idx);
}

void ASTStmtReader::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *E) {
  VisitExpr(E);
  E->setTemporary(Reader.ReadCXXTemporary(F, Record, Idx));
  E->setSubExpr(Reader.ReadSubExpr());
}

void ASTStmtReader::VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E) {
  VisitExpr(E);
  E->TypeInfo = GetTypeSourceInfo(Record, Idx);
  E->RParenLoc = ReadSourceLocation(Record, Idx);
}

void ASTStmtReader::VisitCXXNewExpr(CXXNewExpr *E) {
  VisitExpr(E);
  E->GlobalNew = Record[Idx++];
  E->Initializer = Record[Idx++];
  E->UsualArrayDeleteWantsSize = Record[Idx++];
  bool isArray = Record[Idx++];
  unsigned NumPlacementArgs = Record[Idx++];
  unsigned NumCtorArgs = Record[Idx++];
  E->setOperatorNew(ReadDeclAs<FunctionDecl>(Record, Idx));
  E->setOperatorDelete(ReadDeclAs<FunctionDecl>(Record, Idx));
  E->setConstructor(ReadDeclAs<CXXConstructorDecl>(Record, Idx));
  E->AllocatedTypeInfo = GetTypeSourceInfo(Record, Idx);
  SourceRange TypeIdParens;
  TypeIdParens.setBegin(ReadSourceLocation(Record, Idx));
  TypeIdParens.setEnd(ReadSourceLocation(Record, Idx));
  E->TypeIdParens = TypeIdParens;
  E->StartLoc = ReadSourceLocation(Record, Idx);
  E->EndLoc = ReadSourceLocation(Record, Idx);
  E->ConstructorLParen = ReadSourceLocation(Record, Idx);
  E->ConstructorRParen = ReadSourceLocation(Record, Idx);

  E->AllocateArgsArray(*Reader.getContext(), isArray, NumPlacementArgs,
                       NumCtorArgs);

  // Install all the subexpressions.
  for (CXXNewExpr::raw_arg_iterator I = E->raw_arg_begin(),e = E->raw_arg_end();
       I != e; ++I)
    *I = Reader.ReadSubStmt();
}

void ASTStmtReader::VisitCXXDeleteExpr(CXXDeleteExpr *E) {
  VisitExpr(E);
  E->GlobalDelete = Record[Idx++];
  E->ArrayForm = Record[Idx++];
  E->ArrayFormAsWritten = Record[Idx++];
  E->UsualArrayDeleteWantsSize = Record[Idx++];
  E->OperatorDelete = ReadDeclAs<FunctionDecl>(Record, Idx);
  E->Argument = Reader.ReadSubExpr();
  E->Loc = ReadSourceLocation(Record, Idx);
}

void ASTStmtReader::VisitCXXPseudoDestructorExpr(CXXPseudoDestructorExpr *E) {
  VisitExpr(E);

  E->Base = Reader.ReadSubExpr();
  E->IsArrow = Record[Idx++];
  E->OperatorLoc = ReadSourceLocation(Record, Idx);
  E->QualifierLoc = Reader.ReadNestedNameSpecifierLoc(F, Record, Idx);
  E->ScopeType = GetTypeSourceInfo(Record, Idx);
  E->ColonColonLoc = ReadSourceLocation(Record, Idx);
  E->TildeLoc = ReadSourceLocation(Record, Idx);
  
  IdentifierInfo *II = Reader.GetIdentifierInfo(Record, Idx);
  if (II)
    E->setDestroyedType(II, ReadSourceLocation(Record, Idx));
  else
    E->setDestroyedType(GetTypeSourceInfo(Record, Idx));
}

void ASTStmtReader::VisitExprWithCleanups(ExprWithCleanups *E) {
  VisitExpr(E);
  unsigned NumTemps = Record[Idx++];
  if (NumTemps) {
    E->setNumTemporaries(*Reader.getContext(), NumTemps);
    for (unsigned i = 0; i != NumTemps; ++i)
      E->setTemporary(i, Reader.ReadCXXTemporary(F, Record, Idx));
  }
  E->setSubExpr(Reader.ReadSubExpr());
}

void
ASTStmtReader::VisitCXXDependentScopeMemberExpr(CXXDependentScopeMemberExpr *E){
  VisitExpr(E);
  
  if (Record[Idx++])
    ReadExplicitTemplateArgumentList(E->getExplicitTemplateArgs(),
                                     Record[Idx++]);

  E->Base = Reader.ReadSubExpr();
  E->BaseType = Reader.readType(F, Record, Idx);
  E->IsArrow = Record[Idx++];
  E->OperatorLoc = ReadSourceLocation(Record, Idx);
  E->QualifierLoc = Reader.ReadNestedNameSpecifierLoc(F, Record, Idx);
  E->FirstQualifierFoundInScope = ReadDeclAs<NamedDecl>(Record, Idx);
  ReadDeclarationNameInfo(E->MemberNameInfo, Record, Idx);
}

void
ASTStmtReader::VisitDependentScopeDeclRefExpr(DependentScopeDeclRefExpr *E) {
  VisitExpr(E);
  
  if (Record[Idx++])
    ReadExplicitTemplateArgumentList(E->getExplicitTemplateArgs(), 
                                     Record[Idx++]);

  E->QualifierLoc = Reader.ReadNestedNameSpecifierLoc(F, Record, Idx);
  ReadDeclarationNameInfo(E->NameInfo, Record, Idx);
}

void
ASTStmtReader::VisitCXXUnresolvedConstructExpr(CXXUnresolvedConstructExpr *E) {
  VisitExpr(E);
  assert(Record[Idx] == E->arg_size() && "Read wrong record during creation ?");
  ++Idx; // NumArgs;
  for (unsigned I = 0, N = E->arg_size(); I != N; ++I)
    E->setArg(I, Reader.ReadSubExpr());
  E->Type = GetTypeSourceInfo(Record, Idx);
  E->setLParenLoc(ReadSourceLocation(Record, Idx));
  E->setRParenLoc(ReadSourceLocation(Record, Idx));
}

void ASTStmtReader::VisitOverloadExpr(OverloadExpr *E) {
  VisitExpr(E);
  
  // Read the explicit template argument list, if available.
  if (Record[Idx++])
    ReadExplicitTemplateArgumentList(E->getExplicitTemplateArgs(),
                                     Record[Idx++]);

  unsigned NumDecls = Record[Idx++];
  UnresolvedSet<8> Decls;
  for (unsigned i = 0; i != NumDecls; ++i) {
    NamedDecl *D = ReadDeclAs<NamedDecl>(Record, Idx);
    AccessSpecifier AS = (AccessSpecifier)Record[Idx++];
    Decls.addDecl(D, AS);
  }
  E->initializeResults(*Reader.getContext(), Decls.begin(), Decls.end());

  ReadDeclarationNameInfo(E->NameInfo, Record, Idx);
  E->QualifierLoc = Reader.ReadNestedNameSpecifierLoc(F, Record, Idx);
}

void ASTStmtReader::VisitUnresolvedMemberExpr(UnresolvedMemberExpr *E) {
  VisitOverloadExpr(E);
  E->IsArrow = Record[Idx++];
  E->HasUnresolvedUsing = Record[Idx++];
  E->Base = Reader.ReadSubExpr();
  E->BaseType = Reader.readType(F, Record, Idx);
  E->OperatorLoc = ReadSourceLocation(Record, Idx);
}

void ASTStmtReader::VisitUnresolvedLookupExpr(UnresolvedLookupExpr *E) {
  VisitOverloadExpr(E);
  E->RequiresADL = Record[Idx++];
  if (E->RequiresADL)
    E->StdIsAssociatedNamespace = Record[Idx++];
  E->Overloaded = Record[Idx++];
  E->NamingClass = ReadDeclAs<CXXRecordDecl>(Record, Idx);
}

void ASTStmtReader::VisitUnaryTypeTraitExpr(UnaryTypeTraitExpr *E) {
  VisitExpr(E);
  E->UTT = (UnaryTypeTrait)Record[Idx++];
  E->Value = (bool)Record[Idx++];
  SourceRange Range = ReadSourceRange(Record, Idx);
  E->Loc = Range.getBegin();
  E->RParen = Range.getEnd();
  E->QueriedType = GetTypeSourceInfo(Record, Idx);
}

void ASTStmtReader::VisitBinaryTypeTraitExpr(BinaryTypeTraitExpr *E) {
  VisitExpr(E);
  E->BTT = (BinaryTypeTrait)Record[Idx++];
  E->Value = (bool)Record[Idx++];
  SourceRange Range = ReadSourceRange(Record, Idx);
  E->Loc = Range.getBegin();
  E->RParen = Range.getEnd();
  E->LhsType = GetTypeSourceInfo(Record, Idx);
  E->RhsType = GetTypeSourceInfo(Record, Idx);
}

void ASTStmtReader::VisitArrayTypeTraitExpr(ArrayTypeTraitExpr *E) {
  VisitExpr(E);
  E->ATT = (ArrayTypeTrait)Record[Idx++];
  E->Value = (unsigned int)Record[Idx++];
  SourceRange Range = ReadSourceRange(Record, Idx);
  E->Loc = Range.getBegin();
  E->RParen = Range.getEnd();
  E->QueriedType = GetTypeSourceInfo(Record, Idx);
}

void ASTStmtReader::VisitExpressionTraitExpr(ExpressionTraitExpr *E) {
  VisitExpr(E);
  E->ET = (ExpressionTrait)Record[Idx++];
  E->Value = (bool)Record[Idx++];
  SourceRange Range = ReadSourceRange(Record, Idx);
  E->QueriedExpression = Reader.ReadSubExpr();
  E->Loc = Range.getBegin();
  E->RParen = Range.getEnd();
}

void ASTStmtReader::VisitCXXNoexceptExpr(CXXNoexceptExpr *E) {
  VisitExpr(E);
  E->Value = (bool)Record[Idx++];
  E->Range = ReadSourceRange(Record, Idx);
  E->Operand = Reader.ReadSubExpr();
}

void ASTStmtReader::VisitPackExpansionExpr(PackExpansionExpr *E) {
  VisitExpr(E);
  E->EllipsisLoc = ReadSourceLocation(Record, Idx);
  E->NumExpansions = Record[Idx++];
  E->Pattern = Reader.ReadSubExpr();  
}

void ASTStmtReader::VisitSizeOfPackExpr(SizeOfPackExpr *E) {
  VisitExpr(E);
  E->OperatorLoc = ReadSourceLocation(Record, Idx);
  E->PackLoc = ReadSourceLocation(Record, Idx);
  E->RParenLoc = ReadSourceLocation(Record, Idx);
  E->Length = Record[Idx++];
  E->Pack = ReadDeclAs<NamedDecl>(Record, Idx);
}

void ASTStmtReader::VisitSubstNonTypeTemplateParmExpr(
                                              SubstNonTypeTemplateParmExpr *E) {
  VisitExpr(E);
  E->Param = ReadDeclAs<NonTypeTemplateParmDecl>(Record, Idx);
  E->NameLoc = ReadSourceLocation(Record, Idx);
  E->Replacement = Reader.ReadSubExpr();
}

void ASTStmtReader::VisitSubstNonTypeTemplateParmPackExpr(
                                          SubstNonTypeTemplateParmPackExpr *E) {
  VisitExpr(E);
  E->Param = ReadDeclAs<NonTypeTemplateParmDecl>(Record, Idx);
  TemplateArgument ArgPack = Reader.ReadTemplateArgument(F, Record, Idx);
  if (ArgPack.getKind() != TemplateArgument::Pack)
    return;
  
  E->Arguments = ArgPack.pack_begin();
  E->NumArguments = ArgPack.pack_size();
  E->NameLoc = ReadSourceLocation(Record, Idx);
}

void ASTStmtReader::VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *E) {
  VisitExpr(E);
  E->Temporary = Reader.ReadSubExpr();
}

void ASTStmtReader::VisitOpaqueValueExpr(OpaqueValueExpr *E) {
  VisitExpr(E);
  Idx++; // skip ID
  E->Loc = ReadSourceLocation(Record, Idx);
}

//===----------------------------------------------------------------------===//
// Microsoft Expressions and Statements
//===----------------------------------------------------------------------===//
void ASTStmtReader::VisitCXXUuidofExpr(CXXUuidofExpr *E) {
  VisitExpr(E);
  E->setSourceRange(ReadSourceRange(Record, Idx));
  if (E->isTypeOperand()) { // __uuidof(ComType)
    E->setTypeOperandSourceInfo(
        GetTypeSourceInfo(Record, Idx));
    return;
  }
  
  // __uuidof(expr)
  E->setExprOperand(Reader.ReadSubExpr());
}

void ASTStmtReader::VisitSEHExceptStmt(SEHExceptStmt *S) {
  VisitStmt(S);
  S->Loc = ReadSourceLocation(Record, Idx);
  S->Children[SEHExceptStmt::FILTER_EXPR] = Reader.ReadSubStmt();
  S->Children[SEHExceptStmt::BLOCK] = Reader.ReadSubStmt();
}

void ASTStmtReader::VisitSEHFinallyStmt(SEHFinallyStmt *S) {
  VisitStmt(S);
  S->Loc = ReadSourceLocation(Record, Idx);
  S->Block = Reader.ReadSubStmt();
}

void ASTStmtReader::VisitSEHTryStmt(SEHTryStmt *S) {
  VisitStmt(S);
  S->IsCXXTry = Record[Idx++];
  S->TryLoc = ReadSourceLocation(Record, Idx);
  S->Children[SEHTryStmt::TRY] = Reader.ReadSubStmt();
  S->Children[SEHTryStmt::HANDLER] = Reader.ReadSubStmt();
}

//===----------------------------------------------------------------------===//
// CUDA Expressions and Statements
//===----------------------------------------------------------------------===//

void ASTStmtReader::VisitCUDAKernelCallExpr(CUDAKernelCallExpr *E) {
  VisitCallExpr(E);
  E->setConfig(cast<CallExpr>(Reader.ReadSubExpr()));
}

//===----------------------------------------------------------------------===//
// OpenCL Expressions and Statements.
//===----------------------------------------------------------------------===//
void ASTStmtReader::VisitAsTypeExpr(AsTypeExpr *E) {
  VisitExpr(E);
  E->BuiltinLoc = ReadSourceLocation(Record, Idx);
  E->RParenLoc = ReadSourceLocation(Record, Idx);
  E->SrcExpr = Reader.ReadSubExpr();
}

//===----------------------------------------------------------------------===//
// ASTReader Implementation
//===----------------------------------------------------------------------===//

Stmt *ASTReader::ReadStmt(PerFileData &F) {
  switch (ReadingKind) {
  case Read_Decl:
  case Read_Type:
    return ReadStmtFromStream(F);
  case Read_Stmt:
    return ReadSubStmt();
  }

  llvm_unreachable("ReadingKind not set ?");
  return 0;
}

Expr *ASTReader::ReadExpr(PerFileData &F) {
  return cast_or_null<Expr>(ReadStmt(F));
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
Stmt *ASTReader::ReadStmtFromStream(PerFileData &F) {

  ReadingKindTracker ReadingKind(Read_Stmt, *this);
  llvm::BitstreamCursor &Cursor = F.DeclsCursor;

#ifndef NDEBUG
  unsigned PrevNumStmts = StmtStack.size();
#endif

  RecordData Record;
  unsigned Idx;
  ASTStmtReader Reader(*this, F, Cursor, Record, Idx);
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
      S = DeclRefExpr::CreateEmpty(
        *Context,
        /*HasQualifier=*/Record[ASTStmtReader::NumExprFields],
        /*HasFoundDecl=*/Record[ASTStmtReader::NumExprFields + 1],
        /*HasExplicitTemplateArgs=*/Record[ASTStmtReader::NumExprFields + 2],
        /*NumTemplateArgs=*/Record[ASTStmtReader::NumExprFields + 2] ?
          Record[ASTStmtReader::NumExprFields + 3] : 0);
      break;

    case EXPR_INTEGER_LITERAL:
      S = IntegerLiteral::Create(*Context, Empty);
      break;

    case EXPR_FLOATING_LITERAL:
      S = FloatingLiteral::Create(*Context, Empty);
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
      S = new (Context) UnaryExprOrTypeTraitExpr(Empty);
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
      NestedNameSpecifierLoc QualifierLoc;
      if (Record[Idx++]) { // HasQualifier.
        QualifierLoc = ReadNestedNameSpecifierLoc(F, Record, Idx);
      }

      TemplateArgumentListInfo ArgInfo;
      bool HasExplicitTemplateArgs = Record[Idx++];
      if (HasExplicitTemplateArgs) {
        unsigned NumTemplateArgs = Record[Idx++];
        ArgInfo.setLAngleLoc(ReadSourceLocation(F, Record, Idx));
        ArgInfo.setRAngleLoc(ReadSourceLocation(F, Record, Idx));
        for (unsigned i = 0; i != NumTemplateArgs; ++i)
          ArgInfo.addArgument(ReadTemplateArgumentLoc(F, Record, Idx));
      }
      
      NamedDecl *FoundD = ReadDeclAs<NamedDecl>(F, Record, Idx);
      AccessSpecifier AS = (AccessSpecifier)Record[Idx++];
      DeclAccessPair FoundDecl = DeclAccessPair::make(FoundD, AS);

      QualType T = readType(F, Record, Idx);
      ExprValueKind VK = static_cast<ExprValueKind>(Record[Idx++]);
      ExprObjectKind OK = static_cast<ExprObjectKind>(Record[Idx++]);
      Expr *Base = ReadSubExpr();
      ValueDecl *MemberD = ReadDeclAs<ValueDecl>(F, Record, Idx);
      SourceLocation MemberLoc = ReadSourceLocation(F, Record, Idx);
      DeclarationNameInfo MemberNameInfo(MemberD->getDeclName(), MemberLoc);
      bool IsArrow = Record[Idx++];

      S = MemberExpr::Create(*Context, Base, IsArrow, QualifierLoc,
                             MemberD, FoundDecl, MemberNameInfo,
                             HasExplicitTemplateArgs ? &ArgInfo : 0, T, VK, OK);
      ReadDeclarationNameLoc(F, cast<MemberExpr>(S)->MemberDNLoc,
                             MemberD->getDeclName(), Record, Idx);
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

    case EXPR_BINARY_CONDITIONAL_OPERATOR:
      S = new (Context) BinaryConditionalOperator(Empty);
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

    case EXPR_GENERIC_SELECTION:
      S = new (Context) GenericSelectionExpr(Empty);
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
      llvm_unreachable("mismatching AST file");
      break;
    case EXPR_OBJC_MESSAGE_EXPR:
      S = ObjCMessageExpr::CreateEmpty(*Context,
                                     Record[ASTStmtReader::NumExprFields]);
      break;
    case EXPR_OBJC_ISA:
      S = new (Context) ObjCIsaExpr(Empty);
      break;
    case EXPR_OBJC_INDIRECT_COPY_RESTORE:
      S = new (Context) ObjCIndirectCopyRestoreExpr(Empty);
      break;
    case EXPR_OBJC_BRIDGED_CAST:
      S = new (Context) ObjCBridgedCastExpr(Empty);
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
    case STMT_OBJC_AUTORELEASE_POOL:
      S = new (Context) ObjCAutoreleasePoolStmt(Empty);
      break;
    case STMT_SEH_EXCEPT:
      S = new (Context) SEHExceptStmt(Empty);
      break;
    case STMT_SEH_FINALLY:
      S = new (Context) SEHFinallyStmt(Empty);
      break;
    case STMT_SEH_TRY:
      S = new (Context) SEHTryStmt(Empty);
      break;
    case STMT_CXX_CATCH:
      S = new (Context) CXXCatchStmt(Empty);
      break;

    case STMT_CXX_TRY:
      S = CXXTryStmt::Create(*Context, Empty,
             /*NumHandlers=*/Record[ASTStmtReader::NumStmtFields]);
      break;

    case STMT_CXX_FOR_RANGE:
      S = new (Context) CXXForRangeStmt(Empty);
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
    case EXPR_CXX_UUIDOF_EXPR:
      S = new (Context) CXXUuidofExpr(Empty, true);
      break;
    case EXPR_CXX_UUIDOF_TYPE:
      S = new (Context) CXXUuidofExpr(Empty, false);
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
        
    case EXPR_EXPR_WITH_CLEANUPS:
      S = new (Context) ExprWithCleanups(Empty);
      break;
      
    case EXPR_CXX_DEPENDENT_SCOPE_MEMBER:
      S = CXXDependentScopeMemberExpr::CreateEmpty(*Context,
          /*HasExplicitTemplateArgs=*/Record[ASTStmtReader::NumExprFields],
                  /*NumTemplateArgs=*/Record[ASTStmtReader::NumExprFields]
                                   ? Record[ASTStmtReader::NumExprFields + 1] 
                                   : 0);
      break;
      
    case EXPR_CXX_DEPENDENT_SCOPE_DECL_REF:
      S = DependentScopeDeclRefExpr::CreateEmpty(*Context,
          /*HasExplicitTemplateArgs=*/Record[ASTStmtReader::NumExprFields],
                  /*NumTemplateArgs=*/Record[ASTStmtReader::NumExprFields]
                                   ? Record[ASTStmtReader::NumExprFields + 1] 
                                   : 0);
      break;
      
    case EXPR_CXX_UNRESOLVED_CONSTRUCT:
      S = CXXUnresolvedConstructExpr::CreateEmpty(*Context,
                              /*NumArgs=*/Record[ASTStmtReader::NumExprFields]);
      break;
      
    case EXPR_CXX_UNRESOLVED_MEMBER:
      S = UnresolvedMemberExpr::CreateEmpty(*Context,
          /*HasExplicitTemplateArgs=*/Record[ASTStmtReader::NumExprFields],
                  /*NumTemplateArgs=*/Record[ASTStmtReader::NumExprFields]
                                   ? Record[ASTStmtReader::NumExprFields + 1] 
                                   : 0);
      break;
      
    case EXPR_CXX_UNRESOLVED_LOOKUP:
      S = UnresolvedLookupExpr::CreateEmpty(*Context,
          /*HasExplicitTemplateArgs=*/Record[ASTStmtReader::NumExprFields],
                  /*NumTemplateArgs=*/Record[ASTStmtReader::NumExprFields]
                                   ? Record[ASTStmtReader::NumExprFields + 1] 
                                   : 0);
      break;
      
    case EXPR_CXX_UNARY_TYPE_TRAIT:
      S = new (Context) UnaryTypeTraitExpr(Empty);
      break;

    case EXPR_BINARY_TYPE_TRAIT:
      S = new (Context) BinaryTypeTraitExpr(Empty);
      break;

    case EXPR_ARRAY_TYPE_TRAIT:
      S = new (Context) ArrayTypeTraitExpr(Empty);
      break;

    case EXPR_CXX_EXPRESSION_TRAIT:
      S = new (Context) ExpressionTraitExpr(Empty);
      break;

    case EXPR_CXX_NOEXCEPT:
      S = new (Context) CXXNoexceptExpr(Empty);
      break;

    case EXPR_PACK_EXPANSION:
      S = new (Context) PackExpansionExpr(Empty);
      break;
        
    case EXPR_SIZEOF_PACK:
      S = new (Context) SizeOfPackExpr(Empty);
      break;
        
    case EXPR_SUBST_NON_TYPE_TEMPLATE_PARM:
      S = new (Context) SubstNonTypeTemplateParmExpr(Empty);
      break;
        
    case EXPR_SUBST_NON_TYPE_TEMPLATE_PARM_PACK:
      S = new (Context) SubstNonTypeTemplateParmPackExpr(Empty);
      break;
        
    case EXPR_MATERIALIZE_TEMPORARY:
      S = new (Context) MaterializeTemporaryExpr(Empty);
      break;
        
    case EXPR_OPAQUE_VALUE: {
      unsigned key = Record[ASTStmtReader::NumExprFields];
      OpaqueValueExpr *&expr = OpaqueValueExprs[key];

      // If we already have an entry for this opaque value expression,
      // don't bother reading it again.
      if (expr) {
        StmtStack.push_back(expr);
        continue;
      }

      S = expr = new (Context) OpaqueValueExpr(Empty);
      break;
    }

    case EXPR_CUDA_KERNEL_CALL:
      S = new (Context) CUDAKernelCallExpr(*Context, Empty);
      break;
        
    case EXPR_ASTYPE:
      S = new (Context) AsTypeExpr(Empty);
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
