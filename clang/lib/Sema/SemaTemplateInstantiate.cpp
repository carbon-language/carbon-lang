//===------- SemaTemplateInstantiate.cpp - C++ Template Instantiation ------===/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===/
//
//  This file implements C++ template instantiation.
//
//===----------------------------------------------------------------------===/

#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Lex/Preprocessor.h" // for the identifier table
#include "clang/Basic/LangOptions.h"
#include "llvm/Support/Compiler.h"

using namespace clang;

//===----------------------------------------------------------------------===/
// Template Instantiation Support
//===----------------------------------------------------------------------===/

Sema::InstantiatingTemplate::
InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                      ClassTemplateSpecializationDecl *Entity,
                      SourceRange InstantiationRange)
  :  SemaRef(SemaRef) {

  Invalid = CheckInstantiationDepth(PointOfInstantiation,
                                    InstantiationRange);
  if (!Invalid) {
    ActiveTemplateInstantiation Inst;
    Inst.Kind = ActiveTemplateInstantiation::TemplateInstantiation;
    Inst.PointOfInstantiation = PointOfInstantiation;
    Inst.Entity = reinterpret_cast<uintptr_t>(Entity);
    Inst.TemplateArgs = 0;
    Inst.NumTemplateArgs = 0;
    Inst.InstantiationRange = InstantiationRange;
    SemaRef.ActiveTemplateInstantiations.push_back(Inst);
    Invalid = false;
  }
}

Sema::InstantiatingTemplate::InstantiatingTemplate(Sema &SemaRef, 
                                         SourceLocation PointOfInstantiation,
                                         TemplateDecl *Template,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceRange InstantiationRange)
  : SemaRef(SemaRef) {

  Invalid = CheckInstantiationDepth(PointOfInstantiation,
                                    InstantiationRange);
  if (!Invalid) {
    ActiveTemplateInstantiation Inst;
    Inst.Kind 
      = ActiveTemplateInstantiation::DefaultTemplateArgumentInstantiation;
    Inst.PointOfInstantiation = PointOfInstantiation;
    Inst.Entity = reinterpret_cast<uintptr_t>(Template);
    Inst.TemplateArgs = TemplateArgs;
    Inst.NumTemplateArgs = NumTemplateArgs;
    Inst.InstantiationRange = InstantiationRange;
    SemaRef.ActiveTemplateInstantiations.push_back(Inst);
    Invalid = false;
  }
}

Sema::InstantiatingTemplate::~InstantiatingTemplate() {
  if (!Invalid)
    SemaRef.ActiveTemplateInstantiations.pop_back();
}

bool Sema::InstantiatingTemplate::CheckInstantiationDepth(
                                        SourceLocation PointOfInstantiation,
                                           SourceRange InstantiationRange) {
  if (SemaRef.ActiveTemplateInstantiations.size() 
       <= SemaRef.getLangOptions().InstantiationDepth)
    return false;

  SemaRef.Diag(PointOfInstantiation, 
               diag::err_template_recursion_depth_exceeded)
    << SemaRef.getLangOptions().InstantiationDepth
    << InstantiationRange;
  SemaRef.Diag(PointOfInstantiation, diag::note_template_recursion_depth)
    << SemaRef.getLangOptions().InstantiationDepth;
  return true;
}

/// \brief Post-diagnostic hook for printing the instantiation stack.
void Sema::PrintInstantiationStackHook(unsigned, void *Cookie) {
  Sema &SemaRef = *static_cast<Sema*>(Cookie);
  SemaRef.PrintInstantiationStack();
  SemaRef.LastTemplateInstantiationErrorContext 
    = SemaRef.ActiveTemplateInstantiations.back();
}

/// \brief Prints the current instantiation stack through a series of
/// notes.
void Sema::PrintInstantiationStack() {
  for (llvm::SmallVector<ActiveTemplateInstantiation, 16>::reverse_iterator
         Active = ActiveTemplateInstantiations.rbegin(),
         ActiveEnd = ActiveTemplateInstantiations.rend();
       Active != ActiveEnd;
       ++Active) {
    switch (Active->Kind) {
    case ActiveTemplateInstantiation::TemplateInstantiation: {
      ClassTemplateSpecializationDecl *Spec
        = cast<ClassTemplateSpecializationDecl>((Decl*)Active->Entity);
      Diags.Report(FullSourceLoc(Active->PointOfInstantiation, SourceMgr), 
                   diag::note_template_class_instantiation_here)
        << Context.getTypeDeclType(Spec)
        << Active->InstantiationRange;
      break;
    }

    case ActiveTemplateInstantiation::DefaultTemplateArgumentInstantiation: {
      TemplateDecl *Template = cast<TemplateDecl>((Decl *)Active->Entity);
      std::string TemplateArgsStr
        = ClassTemplateSpecializationType::PrintTemplateArgumentList(
                                                      Active->TemplateArgs, 
                                                      Active->NumTemplateArgs);
      Diags.Report(FullSourceLoc(Active->PointOfInstantiation, SourceMgr),
                   diag::note_default_arg_instantiation_here)
        << (Template->getNameAsString() + TemplateArgsStr)
        << Active->InstantiationRange;
      break;
    }
    }
  }
}

//===----------------------------------------------------------------------===/
// Template Instantiation for Types
//===----------------------------------------------------------------------===/
namespace {
  class VISIBILITY_HIDDEN TemplateTypeInstantiator {
    Sema &SemaRef;
    const TemplateArgument *TemplateArgs;
    unsigned NumTemplateArgs;
    SourceLocation Loc;
    DeclarationName Entity;

  public:
    TemplateTypeInstantiator(Sema &SemaRef, 
                             const TemplateArgument *TemplateArgs,
                             unsigned NumTemplateArgs,
                             SourceLocation Loc,
                             DeclarationName Entity) 
      : SemaRef(SemaRef), TemplateArgs(TemplateArgs), 
        NumTemplateArgs(NumTemplateArgs), Loc(Loc), Entity(Entity) { }

    QualType operator()(QualType T) const { return Instantiate(T); }
    
    QualType Instantiate(QualType T) const;

    // Declare instantiate functions for each type.
#define TYPE(Class, Base)                                       \
    QualType Instantiate##Class##Type(const Class##Type *T,     \
                                      unsigned Quals) const;
#define ABSTRACT_TYPE(Class, Base)
#include "clang/AST/TypeNodes.def"
  };
}

QualType 
TemplateTypeInstantiator::InstantiateExtQualType(const ExtQualType *T,
                                                 unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate ExtQualType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::InstantiateBuiltinType(const BuiltinType *T,
                                                 unsigned Quals) const {
  assert(false && "Builtin types are not dependent and cannot be instantiated");
  return QualType(T, Quals);
}

QualType 
TemplateTypeInstantiator::
InstantiateFixedWidthIntType(const FixedWidthIntType *T, unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate FixedWidthIntType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::InstantiateComplexType(const ComplexType *T,
                                                 unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate ComplexType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::InstantiatePointerType(const PointerType *T,
                                                 unsigned Quals) const {
  QualType PointeeType = Instantiate(T->getPointeeType());
  if (PointeeType.isNull())
    return QualType();

  return SemaRef.BuildPointerType(PointeeType, Quals, Loc, Entity);
}

QualType 
TemplateTypeInstantiator::InstantiateBlockPointerType(const BlockPointerType *T,
                                                      unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate BlockPointerType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::InstantiateReferenceType(const ReferenceType *T, 
                                                   unsigned Quals) const {
  QualType ReferentType = Instantiate(T->getPointeeType());
  if (ReferentType.isNull())
    return QualType();

  return SemaRef.BuildReferenceType(ReferentType, Quals, Loc, Entity);
}

QualType 
TemplateTypeInstantiator::
InstantiateMemberPointerType(const MemberPointerType *T,
                             unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate MemberPointerType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::
InstantiateConstantArrayType(const ConstantArrayType *T, 
                             unsigned Quals) const {
  QualType ElementType = Instantiate(T->getElementType());
  if (ElementType.isNull())
    return ElementType;
  
  // Build a temporary integer literal to specify the size for
  // BuildArrayType. Since we have already checked the size as part of
  // creating the dependent array type in the first place, we know
  // there aren't any errors.
  // FIXME: Is IntTy big enough? Maybe not, but LongLongTy causes
  // problems that I have yet to investigate.
  IntegerLiteral ArraySize(T->getSize(), SemaRef.Context.IntTy, Loc);
  return SemaRef.BuildArrayType(ElementType, T->getSizeModifier(), 
                                &ArraySize, T->getIndexTypeQualifier(), 
                                Loc, Entity);
}

QualType 
TemplateTypeInstantiator::
InstantiateIncompleteArrayType(const IncompleteArrayType *T,
                               unsigned Quals) const {
  QualType ElementType = Instantiate(T->getElementType());
  if (ElementType.isNull())
    return ElementType;
  
  return SemaRef.BuildArrayType(ElementType, T->getSizeModifier(), 
                                0, T->getIndexTypeQualifier(), 
                                Loc, Entity);
}

QualType
TemplateTypeInstantiator::
InstantiateVariableArrayType(const VariableArrayType *T,
                             unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate VariableArrayType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::
InstantiateDependentSizedArrayType(const DependentSizedArrayType *T,
                                   unsigned Quals) const {
  Expr *ArraySize = T->getSizeExpr();
  assert(ArraySize->isValueDependent() && 
         "dependent sized array types must have value dependent size expr");
  
  // Instantiate the element type if needed
  QualType ElementType = T->getElementType();
  if (ElementType->isDependentType()) {
    ElementType = Instantiate(ElementType);
    if (ElementType.isNull())
      return QualType();
  }
  
  // Instantiate the size expression
  Sema::OwningExprResult InstantiatedArraySize = 
    SemaRef.InstantiateExpr(ArraySize, TemplateArgs, NumTemplateArgs);
  if (InstantiatedArraySize.isInvalid())
    return QualType();
  
  return SemaRef.BuildArrayType(ElementType, T->getSizeModifier(),
                                (Expr *)InstantiatedArraySize.release(),
                                T->getIndexTypeQualifier(), Loc, Entity);
}

QualType 
TemplateTypeInstantiator::InstantiateVectorType(const VectorType *T,
                                             unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate VectorType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::InstantiateExtVectorType(const ExtVectorType *T,
                                                   unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate ExtVectorType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::
InstantiateFunctionProtoType(const FunctionProtoType *T,
                             unsigned Quals) const {
  QualType ResultType = Instantiate(T->getResultType());
  if (ResultType.isNull())
    return ResultType;

  llvm::SmallVector<QualType, 16> ParamTypes;
  for (FunctionProtoType::arg_type_iterator Param = T->arg_type_begin(),
                                         ParamEnd = T->arg_type_end(); 
       Param != ParamEnd; ++Param) {
    QualType P = Instantiate(*Param);
    if (P.isNull())
      return P;

    ParamTypes.push_back(P);
  }

  return SemaRef.BuildFunctionType(ResultType, &ParamTypes[0], 
                                   ParamTypes.size(),
                                   T->isVariadic(), T->getTypeQuals(),
                                   Loc, Entity);
}

QualType 
TemplateTypeInstantiator::
InstantiateFunctionNoProtoType(const FunctionNoProtoType *T,
                               unsigned Quals) const {
  assert(false && "Functions without prototypes cannot be dependent.");
  return QualType();
}

QualType 
TemplateTypeInstantiator::InstantiateTypedefType(const TypedefType *T,
                                                 unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate TypedefType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::InstantiateTypeOfExprType(const TypeOfExprType *T,
                                                    unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate TypeOfExprType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::InstantiateTypeOfType(const TypeOfType *T,
                                                unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate TypeOfType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::InstantiateRecordType(const RecordType *T,
                                                unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate RecordType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::InstantiateEnumType(const EnumType *T,
                                              unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate EnumType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::
InstantiateTemplateTypeParmType(const TemplateTypeParmType *T,
                                unsigned Quals) const {
  if (T->getDepth() == 0) {
    // Replace the template type parameter with its corresponding
    // template argument.
    assert(T->getIndex() < NumTemplateArgs && "Wrong # of template args");
    assert(TemplateArgs[T->getIndex()].getKind() == TemplateArgument::Type &&
           "Template argument kind mismatch");
    QualType Result = TemplateArgs[T->getIndex()].getAsType();
    if (Result.isNull() || !Quals) 
      return Result;

    // C++ [dcl.ref]p1:
    //   [...] Cv-qualified references are ill-formed except when
    //   the cv-qualifiers are introduced through the use of a
    //   typedef (7.1.3) or of a template type argument (14.3), in
    //   which case the cv-qualifiers are ignored.
    if (Quals && Result->isReferenceType())
      Quals = 0;

    return QualType(Result.getTypePtr(), Quals | Result.getCVRQualifiers());
  } 

  // The template type parameter comes from an inner template (e.g.,
  // the template parameter list of a member template inside the
  // template we are instantiating). Create a new template type
  // parameter with the template "level" reduced by one.
  return SemaRef.Context.getTemplateTypeParmType(T->getDepth() - 1,
                                                 T->getIndex(),
                                                 T->getName())
    .getQualifiedType(Quals);
}

QualType 
TemplateTypeInstantiator::
InstantiateClassTemplateSpecializationType(
                                  const ClassTemplateSpecializationType *T,
                                  unsigned Quals) const {
  llvm::SmallVector<TemplateArgument, 16> InstantiatedTemplateArgs;
  InstantiatedTemplateArgs.reserve(T->getNumArgs());
  for (ClassTemplateSpecializationType::iterator Arg = T->begin(), 
                                              ArgEnd = T->end();
       Arg != ArgEnd; ++Arg) {
    switch (Arg->getKind()) {
    case TemplateArgument::Type: {
      QualType T = SemaRef.InstantiateType(Arg->getAsType(), 
                                           TemplateArgs, NumTemplateArgs,
                                           Arg->getLocation(),
                                           DeclarationName());
      if (T.isNull())
        return QualType();

      InstantiatedTemplateArgs.push_back(
                                TemplateArgument(Arg->getLocation(), T));
      break;
    }

    case TemplateArgument::Declaration:
    case TemplateArgument::Integral:
      InstantiatedTemplateArgs.push_back(*Arg);
      break;

    case TemplateArgument::Expression:
      Sema::OwningExprResult E 
        = SemaRef.InstantiateExpr(Arg->getAsExpr(), TemplateArgs,
                                  NumTemplateArgs);
      if (E.isInvalid())
        return QualType();
      InstantiatedTemplateArgs.push_back((Expr *)E.release());
      break;
    }
  }

  // FIXME: We're missing the locations of the template name, '<', and
  // '>'.
  return SemaRef.CheckClassTemplateId(cast<ClassTemplateDecl>(T->getTemplate()),
                                      Loc,
                                      SourceLocation(),
                                      &InstantiatedTemplateArgs[0],
                                      InstantiatedTemplateArgs.size(),
                                      SourceLocation());
}

QualType 
TemplateTypeInstantiator::
InstantiateObjCInterfaceType(const ObjCInterfaceType *T,
                             unsigned Quals) const {
  assert(false && "Objective-C types cannot be dependent");
  return QualType();
}

QualType 
TemplateTypeInstantiator::
InstantiateObjCQualifiedInterfaceType(const ObjCQualifiedInterfaceType *T,
                                      unsigned Quals) const {
  assert(false && "Objective-C types cannot be dependent");
  return QualType();
}

QualType 
TemplateTypeInstantiator::
InstantiateObjCQualifiedIdType(const ObjCQualifiedIdType *T,
                               unsigned Quals) const {
  assert(false && "Objective-C types cannot be dependent");
  return QualType();
}

QualType 
TemplateTypeInstantiator::
InstantiateObjCQualifiedClassType(const ObjCQualifiedClassType *T,
                                  unsigned Quals) const {
  assert(false && "Objective-C types cannot be dependent");
  return QualType();
}

/// \brief The actual implementation of Sema::InstantiateType().
QualType TemplateTypeInstantiator::Instantiate(QualType T) const {
  // If T is not a dependent type, there is nothing to do.
  if (!T->isDependentType())
    return T;

  switch (T->getTypeClass()) {
#define TYPE(Class, Base)                                               \
  case Type::Class:                                                     \
    return Instantiate##Class##Type(cast<Class##Type>(T.getTypePtr()),  \
                                    T.getCVRQualifiers());
#define ABSTRACT_TYPE(Class, Base)
#include "clang/AST/TypeNodes.def"
  }
  
  assert(false && "Not all types have been decoded for instantiation");
  return QualType();
}

/// \brief Instantiate the type T with a given set of template arguments.
///
/// This routine substitutes the given template arguments into the
/// type T and produces the instantiated type.
///
/// \param T the type into which the template arguments will be
/// substituted. If this type is not dependent, it will be returned
/// immediately.
///
/// \param TemplateArgs the template arguments that will be
/// substituted for the top-level template parameters within T.
///
/// \param NumTemplateArgs the number of template arguments provided
/// by TemplateArgs.
///
/// \param Loc the location in the source code where this substitution
/// is being performed. It will typically be the location of the
/// declarator (if we're instantiating the type of some declaration)
/// or the location of the type in the source code (if, e.g., we're
/// instantiating the type of a cast expression).
///
/// \param Entity the name of the entity associated with a declaration
/// being instantiated (if any). May be empty to indicate that there
/// is no such entity (if, e.g., this is a type that occurs as part of
/// a cast expression) or that the entity has no name (e.g., an
/// unnamed function parameter).
///
/// \returns If the instantiation succeeds, the instantiated
/// type. Otherwise, produces diagnostics and returns a NULL type.
QualType Sema::InstantiateType(QualType T, 
                               const TemplateArgument *TemplateArgs,
                               unsigned NumTemplateArgs,
                               SourceLocation Loc, DeclarationName Entity) {
  assert(!ActiveTemplateInstantiations.empty() &&
         "Cannot perform an instantiation without some context on the "
         "instantiation stack");

  // If T is not a dependent type, there is nothing to do.
  if (!T->isDependentType())
    return T;

  TemplateTypeInstantiator Instantiator(*this, TemplateArgs, NumTemplateArgs,
                                        Loc, Entity);
  return Instantiator(T);
}

//===----------------------------------------------------------------------===/
// Template Instantiation for Expressions
//===----------------------------------------------------------------------===/
namespace {
  class VISIBILITY_HIDDEN TemplateExprInstantiator 
    : public StmtVisitor<TemplateExprInstantiator, Sema::OwningExprResult> {
    Sema &SemaRef;
    const TemplateArgument *TemplateArgs;
    unsigned NumTemplateArgs;

  public:
    typedef Sema::OwningExprResult OwningExprResult;

    TemplateExprInstantiator(Sema &SemaRef, 
                             const TemplateArgument *TemplateArgs,
                             unsigned NumTemplateArgs)
      : SemaRef(SemaRef), TemplateArgs(TemplateArgs), 
        NumTemplateArgs(NumTemplateArgs) { }

    // FIXME: Once we get closer to completion, replace these
    // manually-written declarations with automatically-generated ones
    // from clang/AST/StmtNodes.def.
    OwningExprResult VisitIntegerLiteral(IntegerLiteral *E);
    OwningExprResult VisitDeclRefExpr(DeclRefExpr *E);
    OwningExprResult VisitParenExpr(ParenExpr *E);
    OwningExprResult VisitUnaryOperator(UnaryOperator *E);
    OwningExprResult VisitBinaryOperator(BinaryOperator *E);
    OwningExprResult VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E);
    OwningExprResult VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E);
    OwningExprResult VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *E);

    // Base case. I'm supposed to ignore this.
    Sema::OwningExprResult VisitStmt(Stmt *S) { 
      S->dump();
      assert(false && "Cannot instantiate this kind of expression");
      return SemaRef.ExprError(); 
    }
  };
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitIntegerLiteral(IntegerLiteral *E) {
  return SemaRef.Clone(E);
}

Sema::OwningExprResult
TemplateExprInstantiator::VisitDeclRefExpr(DeclRefExpr *E) {
  Decl *D = E->getDecl();
  if (NonTypeTemplateParmDecl *NTTP = dyn_cast<NonTypeTemplateParmDecl>(D)) {
    assert(NTTP->getDepth() == 0 && "No nested templates yet");
    const TemplateArgument &Arg = TemplateArgs[NTTP->getPosition()]; 
    return SemaRef.Owned(new (SemaRef.Context) IntegerLiteral(
                                                 *Arg.getAsIntegral(),
                                                 Arg.getIntegralType(), 
                                       E->getSourceRange().getBegin()));
  } else
    assert(false && "Can't handle arbitrary declaration references");

  return SemaRef.ExprError();
}

Sema::OwningExprResult
TemplateExprInstantiator::VisitParenExpr(ParenExpr *E) {
  Sema::OwningExprResult SubExpr
    = SemaRef.InstantiateExpr(E->getSubExpr(), TemplateArgs, NumTemplateArgs);
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();

  return SemaRef.Owned(new (SemaRef.Context) ParenExpr(
                                               E->getLParen(), E->getRParen(), 
                                               (Expr *)SubExpr.release()));
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitUnaryOperator(UnaryOperator *E) {
  Sema::OwningExprResult Arg = Visit(E->getSubExpr());
  if (Arg.isInvalid())
    return SemaRef.ExprError();

  return SemaRef.CreateBuiltinUnaryOp(E->getOperatorLoc(),
                                      E->getOpcode(),
                                      move(Arg));
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitBinaryOperator(BinaryOperator *E) {
  Sema::OwningExprResult LHS = Visit(E->getLHS());
  if (LHS.isInvalid())
    return SemaRef.ExprError();

  Sema::OwningExprResult RHS = Visit(E->getRHS());
  if (RHS.isInvalid())
    return SemaRef.ExprError();

  Sema::OwningExprResult Result
    = SemaRef.CreateBuiltinBinOp(E->getOperatorLoc(), 
                                 E->getOpcode(),
                                 (Expr *)LHS.get(),
                                 (Expr *)RHS.get());
  if (Result.isInvalid())
    return SemaRef.ExprError();

  LHS.release();
  RHS.release();
  return move(Result);
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
  Sema::OwningExprResult First = Visit(E->getArg(0));
  if (First.isInvalid())
    return SemaRef.ExprError();

  Expr *Args[2] = { (Expr *)First.get(), 0 };

  Sema::OwningExprResult Second(SemaRef);
  if (E->getNumArgs() == 2) {
    Second = Visit(E->getArg(1));

    if (Second.isInvalid())
      return SemaRef.ExprError();

    Args[1] = (Expr *)Second.get();
  }

  if (!E->isTypeDependent()) { 
    // Since our original expression was not type-dependent, we do not
    // perform lookup again at instantiation time (C++ [temp.dep]p1).
    // Instead, we just build the new overloaded operator call
    // expression.
    First.release();
    Second.release();
    return SemaRef.Owned(new (SemaRef.Context) CXXOperatorCallExpr(
                                                       SemaRef.Context, 
                                                       E->getOperator(),
                                                       E->getCallee(), 
                                                       Args, E->getNumArgs(),
                                                       E->getType(), 
                                                       E->getOperatorLoc()));
  }

  bool isPostIncDec = E->getNumArgs() == 2 && 
    (E->getOperator() == OO_PlusPlus || E->getOperator() == OO_MinusMinus);
  if (E->getNumArgs() == 1 || isPostIncDec) {
    if (!Args[0]->getType()->isOverloadableType()) {
      // The argument is not of overloadable type, so try to create a
      // built-in unary operation.
      UnaryOperator::Opcode Opc 
        = UnaryOperator::getOverloadedOpcode(E->getOperator(), isPostIncDec);

      return SemaRef.CreateBuiltinUnaryOp(E->getOperatorLoc(), Opc,
                                          move(First));
    }

    // Fall through to perform overload resolution
  } else {
    assert(E->getNumArgs() == 2 && "Expected binary operation");

    Sema::OwningExprResult Result(SemaRef);
    if (!Args[0]->getType()->isOverloadableType() && 
        !Args[1]->getType()->isOverloadableType()) {
      // Neither of the arguments is an overloadable type, so try to
      // create a built-in binary operation.
      BinaryOperator::Opcode Opc = 
        BinaryOperator::getOverloadedOpcode(E->getOperator());
      Result = SemaRef.CreateBuiltinBinOp(E->getOperatorLoc(), Opc, 
                                          Args[0], Args[1]);
      if (Result.isInvalid())
        return SemaRef.ExprError();

      First.release();
      Second.release();
      return move(Result);
    }

    // Fall through to perform overload resolution.
  }

  // Compute the set of functions that were found at template
  // definition time.
  Sema::FunctionSet Functions;
  DeclRefExpr *DRE = cast<DeclRefExpr>(E->getCallee());
  OverloadedFunctionDecl *Overloads 
    = cast<OverloadedFunctionDecl>(DRE->getDecl());
  
  // FIXME: Do we have to check
  // IsAcceptableNonMemberOperatorCandidate for each of these?
  for (OverloadedFunctionDecl::function_iterator 
         F = Overloads->function_begin(),
         FEnd = Overloads->function_end();
       F != FEnd; ++F)
    Functions.insert(*F);
  
  // Add any functions found via argument-dependent lookup.
  DeclarationName OpName 
    = SemaRef.Context.DeclarationNames.getCXXOperatorName(E->getOperator());
  SemaRef.ArgumentDependentLookup(OpName, Args, E->getNumArgs(), Functions);

  // Create the overloaded operator invocation.
  if (E->getNumArgs() == 1 || isPostIncDec) {
    UnaryOperator::Opcode Opc 
      = UnaryOperator::getOverloadedOpcode(E->getOperator(), isPostIncDec);
    return SemaRef.CreateOverloadedUnaryOp(E->getOperatorLoc(), Opc,
                                           Functions, move(First));
  }

  // FIXME: This would be far less ugly if CreateOverloadedBinOp took
  // in ExprArg arguments!
  BinaryOperator::Opcode Opc = 
    BinaryOperator::getOverloadedOpcode(E->getOperator());
  OwningExprResult Result 
    = SemaRef.CreateOverloadedBinOp(E->getOperatorLoc(), Opc, 
                                    Functions, Args[0], Args[1]);

  if (Result.isInvalid())
    return SemaRef.ExprError();

  First.release();
  Second.release();
  return move(Result);  
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E) {
  bool isSizeOf = E->isSizeOf();

  if (E->isArgumentType()) {
    QualType T = E->getArgumentType();
    if (T->isDependentType()) {
      T = SemaRef.InstantiateType(T, TemplateArgs, NumTemplateArgs,
                                  /*FIXME*/E->getOperatorLoc(),
                                  &SemaRef.PP.getIdentifierTable().get("sizeof"));
      if (T.isNull())
        return SemaRef.ExprError();
    }

    return SemaRef.CreateSizeOfAlignOfExpr(T, E->getOperatorLoc(), isSizeOf,
                                           E->getSourceRange());
  } 

  Sema::OwningExprResult Arg = Visit(E->getArgumentExpr());
  if (Arg.isInvalid())
    return SemaRef.ExprError();

  Sema::OwningExprResult Result
    = SemaRef.CreateSizeOfAlignOfExpr((Expr *)Arg.get(), E->getOperatorLoc(),
                                      isSizeOf, E->getSourceRange());
  if (Result.isInvalid())
    return SemaRef.ExprError();

  Arg.release();
  return move(Result);
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCXXTemporaryObjectExpr(
                                                  CXXTemporaryObjectExpr *E) {
  QualType T = E->getType();
  if (T->isDependentType()) {
    T = SemaRef.InstantiateType(T, TemplateArgs, NumTemplateArgs,
                                E->getTypeBeginLoc(), DeclarationName());
    if (T.isNull())
      return SemaRef.ExprError();
  }

  llvm::SmallVector<Expr *, 16> Args;
  Args.reserve(E->getNumArgs());
  bool Invalid = false;
  for (CXXTemporaryObjectExpr::arg_iterator Arg = E->arg_begin(), 
                                         ArgEnd = E->arg_end();
       Arg != ArgEnd; ++Arg) {
    OwningExprResult InstantiatedArg = Visit(*Arg);
    if (InstantiatedArg.isInvalid()) {
      Invalid = true;
      break;
    }

    Args.push_back((Expr *)InstantiatedArg.release());
  }

  if (!Invalid) {
    SourceLocation CommaLoc;
    // FIXME: HACK!
    if (Args.size() > 1)
      CommaLoc 
        = SemaRef.PP.getLocForEndOfToken(Args[0]->getSourceRange().getEnd());
    Sema::OwningExprResult Result(
      SemaRef.ActOnCXXTypeConstructExpr(SourceRange(E->getTypeBeginLoc()
                                                    /*, FIXME*/),
                                        T.getAsOpaquePtr(),
                                        /*FIXME*/E->getTypeBeginLoc(),
                                        Sema::MultiExprArg(SemaRef,
                                                           (void**)&Args[0],
                                                           Args.size()),
                                        /*HACK*/&CommaLoc,
                                        E->getSourceRange().getEnd()));
    // At this point, Args no longer owns the arguments, no matter what.
    return move(Result);
  }

  // Clean up the instantiated arguments.
  // FIXME: Would rather do this with RAII.
  for (unsigned Idx = 0; Idx < Args.size(); ++Idx)
    SemaRef.DeleteExpr(Args[Idx]);

  return SemaRef.ExprError();
}

Sema::OwningExprResult 
Sema::InstantiateExpr(Expr *E, const TemplateArgument *TemplateArgs,
                      unsigned NumTemplateArgs) {
  TemplateExprInstantiator Instantiator(*this, TemplateArgs, NumTemplateArgs);
  return Instantiator.Visit(E);
}

/// \brief Instantiate the base class specifiers of the given class
/// template specialization.
///
/// Produces a diagnostic and returns true on error, returns false and
/// attaches the instantiated base classes to the class template
/// specialization if successful.
bool 
Sema::InstantiateBaseSpecifiers(
                           ClassTemplateSpecializationDecl *ClassTemplateSpec,
                           ClassTemplateDecl *ClassTemplate) {
  bool Invalid = false;
  llvm::SmallVector<CXXBaseSpecifier*, 8> InstantiatedBases;
  for (ClassTemplateSpecializationDecl::base_class_iterator
         Base = ClassTemplate->getTemplatedDecl()->bases_begin(),
         BaseEnd = ClassTemplate->getTemplatedDecl()->bases_end();
       Base != BaseEnd; ++Base) {
    if (!Base->getType()->isDependentType()) {
      // FIXME: Allocate via ASTContext
      InstantiatedBases.push_back(new CXXBaseSpecifier(*Base));
      continue;
    }

    QualType BaseType = InstantiateType(Base->getType(), 
                                        ClassTemplateSpec->getTemplateArgs(),
                                        ClassTemplateSpec->getNumTemplateArgs(),
                                        Base->getSourceRange().getBegin(),
                                        DeclarationName());
    if (BaseType.isNull()) {
      Invalid = true;
      continue;
    }

    if (CXXBaseSpecifier *InstantiatedBase
          = CheckBaseSpecifier(ClassTemplateSpec,
                               Base->getSourceRange(),
                               Base->isVirtual(),
                               Base->getAccessSpecifierAsWritten(),
                               BaseType,
                               /*FIXME: Not totally accurate */
                               Base->getSourceRange().getBegin()))
      InstantiatedBases.push_back(InstantiatedBase);
    else
      Invalid = true;
  }

  if (!Invalid &&
      AttachBaseSpecifiers(ClassTemplateSpec, &InstantiatedBases[0],
                           InstantiatedBases.size()))
    Invalid = true;

  return Invalid;
}

bool 
Sema::InstantiateClassTemplateSpecialization(
                           ClassTemplateSpecializationDecl *ClassTemplateSpec,
                           bool ExplicitInstantiation) {
  // Perform the actual instantiation on the canonical declaration.
  ClassTemplateSpec = cast<ClassTemplateSpecializationDecl>(
                               Context.getCanonicalDecl(ClassTemplateSpec));

  // We can only instantiate something that hasn't already been
  // instantiated or specialized. Fail without any diagnostics: our
  // caller will provide an error message.
  if (ClassTemplateSpec->getSpecializationKind() != TSK_Undeclared)
    return true;

  // FIXME: Push this class template instantiation onto the
  // instantiation stack, checking for recursion that exceeds a
  // certain depth.

  // FIXME: Perform class template partial specialization to select
  // the best template.
  ClassTemplateDecl *Template = ClassTemplateSpec->getSpecializedTemplate();

  if (!Template->getTemplatedDecl()->getDefinition(Context)) {
    Diag(ClassTemplateSpec->getLocation(), 
         diag::err_template_implicit_instantiate_undefined)
      << Context.getTypeDeclType(ClassTemplateSpec);
    Diag(Template->getTemplatedDecl()->getLocation(), 
         diag::note_template_decl_here);
    return true;
  }

  // Note that this is an instantiation.  
  ClassTemplateSpec->setSpecializationKind(
                        ExplicitInstantiation? TSK_ExplicitInstantiation 
                                             : TSK_ImplicitInstantiation);


  bool Invalid = false;
  
  InstantiatingTemplate Inst(*this, ClassTemplateSpec->getLocation(),
                             ClassTemplateSpec);
  if (Inst)
    return true;

  // Enter the scope of this instantiation. We don't use
  // PushDeclContext because we don't have a scope.
  DeclContext *PreviousContext = CurContext;
  CurContext = ClassTemplateSpec;

  // Start the definition of this instantiation.
  ClassTemplateSpec->startDefinition();


  // Instantiate the base class specifiers.
  if (InstantiateBaseSpecifiers(ClassTemplateSpec, Template))
    Invalid = true;

  // FIXME: Create the injected-class-name for the
  // instantiation. Should this be a typedef or something like it?

  RecordDecl *Pattern = Template->getTemplatedDecl();
  llvm::SmallVector<DeclTy *, 32> Fields;
  for (RecordDecl::decl_iterator Member = Pattern->decls_begin(),
                              MemberEnd = Pattern->decls_end();
       Member != MemberEnd; ++Member) {
    if (TypedefDecl *Typedef = dyn_cast<TypedefDecl>(*Member)) {
      // FIXME: Simplified instantiation of typedefs needs to be made
      // "real".
      QualType T = Typedef->getUnderlyingType();
      if (T->isDependentType()) {
        T = InstantiateType(T, ClassTemplateSpec->getTemplateArgs(),
                            ClassTemplateSpec->getNumTemplateArgs(),
                            Typedef->getLocation(),
                            Typedef->getDeclName());
        if (T.isNull()) {
          Invalid = true;
          T = Context.IntTy;
        }
      }
       
      // Create the new typedef
      TypedefDecl *New 
        = TypedefDecl::Create(Context, ClassTemplateSpec,
                              Typedef->getLocation(),
                              Typedef->getIdentifier(),
                              T);
      ClassTemplateSpec->addDecl(New);
    } 
    else if (FieldDecl *Field = dyn_cast<FieldDecl>(*Member)) {
      // FIXME: Simplified instantiation of fields needs to be made
      // "real".
      bool InvalidDecl = false;
      QualType T = Field->getType();
      if (T->isDependentType())  {
        T = InstantiateType(T, ClassTemplateSpec->getTemplateArgs(),
                            ClassTemplateSpec->getNumTemplateArgs(),
                            Field->getLocation(),
                            Field->getDeclName());
        if (!T.isNull() && T->isFunctionType()) {
          // C++ [temp.arg.type]p3:
          //   If a declaration acquires a function type through a type
          //   dependent on a template-parameter and this causes a
          //   declaration that does not use the syntactic form of a
          //   function declarator to have function type, the program is
          //   ill-formed.
          Diag(Field->getLocation(), diag::err_field_instantiates_to_function) 
            << T;
          T = QualType();
          InvalidDecl = true;
        }
      }

      Expr *BitWidth = Field->getBitWidth();
      if (InvalidDecl)
        BitWidth = 0;
      else if (BitWidth) {
        OwningExprResult InstantiatedBitWidth
          = InstantiateExpr(BitWidth, 
                            ClassTemplateSpec->getTemplateArgs(),
                            ClassTemplateSpec->getNumTemplateArgs());
        if (InstantiatedBitWidth.isInvalid()) {
          Invalid = InvalidDecl = true;
          BitWidth = 0;
        } else
          BitWidth = (Expr *)InstantiatedBitWidth.release();
      }

      FieldDecl *New = CheckFieldDecl(Field->getDeclName(), T,
                                      ClassTemplateSpec, 
                                      Field->getLocation(),
                                      Field->isMutable(),
                                      BitWidth,
                                      Field->getAccess(),
                                      0);
      if (New) {
        ClassTemplateSpec->addDecl(New);
        Fields.push_back(New);

        if (InvalidDecl)
          New->setInvalidDecl();

        if (New->isInvalidDecl())
          Invalid = true;
      }
    } else if (StaticAssertDecl *SA = dyn_cast<StaticAssertDecl>(*Member)) {
      Expr *AssertExpr = SA->getAssertExpr();
      
      OwningExprResult InstantiatedAssertExpr
        = InstantiateExpr(AssertExpr, 
                          ClassTemplateSpec->getTemplateArgs(),
                          ClassTemplateSpec->getNumTemplateArgs());
      if (!InstantiatedAssertExpr.isInvalid()) {
        OwningExprResult Message = Clone(SA->getMessage());

        Decl *New = 
          (Decl *)ActOnStaticAssertDeclaration(SA->getLocation(), 
                                               move(InstantiatedAssertExpr),
                                               move(Message));
        if (New->isInvalidDecl())
          Invalid = true;
          
      } else
        Invalid = true;
    }
  }

  // Finish checking fields.
  ActOnFields(0, ClassTemplateSpec->getLocation(), ClassTemplateSpec,
              &Fields[0], Fields.size(), SourceLocation(), SourceLocation(),
              0);

  // Add any implicitly-declared members that we might need.
  AddImplicitlyDeclaredMembersToClass(ClassTemplateSpec);

  // Exit the scope of this instantiation.
  CurContext = PreviousContext;

  return Invalid;
}
