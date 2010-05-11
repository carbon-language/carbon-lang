//===--- RecursiveASTVisitor.h - Recursive AST Visitor ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the RecursiveASTVisitor interface, which recursively
//  traverses the entire AST.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_AST_RECURSIVEASTVISITOR_H
#define LLVM_CLANG_AST_RECURSIVEASTVISITOR_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclFriend.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/TemplateName.h"
#include "clang/AST/Type.h"

namespace clang {
  
#define DISPATCH(NAME, CLASS, Var) \
return static_cast<Derived*>(this)->Visit ## NAME(static_cast<CLASS*>(Var))

  /**
   * \brief A visitor that recursively walks the entire Clang AST.
   *
   * Clients of this visitor should subclass the visitor (providing themselves
   * as the template argument, using the curiously recurring template pattern)
   * and override any of the Visit* methods for declaration, type, statement, 
   * expression, or other AST nodes where the visitor should customize 
   * behavior. Returning "true" from one of these overridden functions will
   * abort the traversal.
   */
  template<typename Derived>
  class RecursiveASTVisitor {
  public:
    /// \brief Return a reference to the derived class.
    Derived &getDerived() { return *static_cast<Derived*>(this); }
    
    /// \brief Recursively visit a statement or expression.
    ///
    /// \returns true if the visitation was terminated early, false otherwise.
    bool Visit(Stmt *S);
        
    /// \brief Recursively visit a type.
    ///
    /// \returns true if the visitation was terminated early, false otherwise.
    bool Visit(QualType T);    
    
    /// \brief Recursively visit a declaration.
    ///
    /// \returns true if the visitation was terminated early, false otherwise.
    bool Visit(Decl *D);
    
    /// \brief Recursively visit a C++ nested-name-specifier.
    ///
    /// \returns true if the visitation was terminated early, false otherwise.
    bool VisitNestedNameSpecifier(NestedNameSpecifier *NNS);
    
    /// \brief Recursively visit a template name.
    ///
    /// \returns true if the visitation was terminated early, false otherwise.
    bool VisitTemplateName(TemplateName Template);
    
    /// \brief Recursively visit a template argument.
    ///
    /// \returns true if the visitation was terminated early, false otherwise.
    bool VisitTemplateArgument(const TemplateArgument &Arg);
    
    /// \brief Recursively visit a set of template arguments.
    ///
    /// \returns true if the visitation was terminated early, false otherwise.
    bool VisitTemplateArguments(const TemplateArgument *Args, unsigned NumArgs);
    
    // If the implementation chooses not to implement a certain visit method, fall
    // back on VisitExpr or whatever else is the superclass.
#define STMT(CLASS, PARENT)                                   \
bool Visit ## CLASS(CLASS *S) { DISPATCH(PARENT, PARENT, S); }
#include "clang/AST/StmtNodes.inc"
    
    // If the implementation doesn't implement binary operator methods, fall back
    // on VisitBinaryOperator.
#define BINOP_FALLBACK(NAME) \
bool VisitBin ## NAME(BinaryOperator *S) {   \
DISPATCH(BinaryOperator, BinaryOperator, S); \
}
    BINOP_FALLBACK(PtrMemD)                    BINOP_FALLBACK(PtrMemI)
    BINOP_FALLBACK(Mul)   BINOP_FALLBACK(Div)  BINOP_FALLBACK(Rem)
    BINOP_FALLBACK(Add)   BINOP_FALLBACK(Sub)  BINOP_FALLBACK(Shl)
    BINOP_FALLBACK(Shr)
    
    BINOP_FALLBACK(LT)    BINOP_FALLBACK(GT)   BINOP_FALLBACK(LE)
    BINOP_FALLBACK(GE)    BINOP_FALLBACK(EQ)   BINOP_FALLBACK(NE)
    BINOP_FALLBACK(And)   BINOP_FALLBACK(Xor)  BINOP_FALLBACK(Or)
    BINOP_FALLBACK(LAnd)  BINOP_FALLBACK(LOr)
    
    BINOP_FALLBACK(Assign)
    BINOP_FALLBACK(Comma)
#undef BINOP_FALLBACK
    
    // If the implementation doesn't implement compound assignment operator
    // methods, fall back on VisitCompoundAssignOperator.
#define CAO_FALLBACK(NAME) \
bool VisitBin ## NAME(CompoundAssignOperator *S) { \
DISPATCH(CompoundAssignOperator, CompoundAssignOperator, S); \
}
    CAO_FALLBACK(MulAssign) CAO_FALLBACK(DivAssign) CAO_FALLBACK(RemAssign)
    CAO_FALLBACK(AddAssign) CAO_FALLBACK(SubAssign) CAO_FALLBACK(ShlAssign)
    CAO_FALLBACK(ShrAssign) CAO_FALLBACK(AndAssign) CAO_FALLBACK(OrAssign)
    CAO_FALLBACK(XorAssign)
#undef CAO_FALLBACK
    
    // If the implementation doesn't implement unary operator methods, fall back
    // on VisitUnaryOperator.
#define UNARYOP_FALLBACK(NAME) \
bool VisitUnary ## NAME(UnaryOperator *S) { \
DISPATCH(UnaryOperator, UnaryOperator, S);    \
}
    UNARYOP_FALLBACK(PostInc)   UNARYOP_FALLBACK(PostDec)
    UNARYOP_FALLBACK(PreInc)    UNARYOP_FALLBACK(PreDec)
    UNARYOP_FALLBACK(AddrOf)    UNARYOP_FALLBACK(Deref)
    
    UNARYOP_FALLBACK(Plus)      UNARYOP_FALLBACK(Minus)
    UNARYOP_FALLBACK(Not)       UNARYOP_FALLBACK(LNot)
    UNARYOP_FALLBACK(Real)      UNARYOP_FALLBACK(Imag)
    UNARYOP_FALLBACK(Extension) UNARYOP_FALLBACK(OffsetOf)
#undef UNARYOP_FALLBACK
    
    /// \brief Basis base for statement and expression visitation, which
    /// visits all of the substatements and subexpressions.
    bool VisitStmt(Stmt *Node);
    
    bool VisitType(Type *T);
    
#define TYPE(Class, Base) \
    bool Visit##Class##Type(Class##Type *T);
#include "clang/AST/TypeNodes.def"
    
    bool VisitDecl(Decl *D);
    
#define DECL(Class, Base)                          \
    bool Visit##Class##Decl(Class##Decl *D) {      \
      return getDerived().Visit##Base(D);          \
    }
#define ABSTRACT_DECL(Class, Base) DECL(Class, Base)
#include "clang/AST/DeclNodes.def"
  };

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::Visit(Stmt *S) {
    if (!S)
      return false;
    
    // If we have a binary expr, dispatch to the subcode of the binop.  A smart
    // optimizer (e.g. LLVM) will fold this comparison into the switch stmt
    // below.
    if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(S)) {
      switch (BinOp->getOpcode()) {
      case BinaryOperator::PtrMemD:   DISPATCH(BinPtrMemD, BinaryOperator, S);
      case BinaryOperator::PtrMemI:   DISPATCH(BinPtrMemI, BinaryOperator, S);
      case BinaryOperator::Mul:       DISPATCH(BinMul,     BinaryOperator, S);
      case BinaryOperator::Div:       DISPATCH(BinDiv,     BinaryOperator, S);
      case BinaryOperator::Rem:       DISPATCH(BinRem,     BinaryOperator, S);
      case BinaryOperator::Add:       DISPATCH(BinAdd,     BinaryOperator, S);
      case BinaryOperator::Sub:       DISPATCH(BinSub,     BinaryOperator, S);
      case BinaryOperator::Shl:       DISPATCH(BinShl,     BinaryOperator, S);
      case BinaryOperator::Shr:       DISPATCH(BinShr,     BinaryOperator, S);
        
      case BinaryOperator::LT:        DISPATCH(BinLT,      BinaryOperator, S);
      case BinaryOperator::GT:        DISPATCH(BinGT,      BinaryOperator, S);
      case BinaryOperator::LE:        DISPATCH(BinLE,      BinaryOperator, S);
      case BinaryOperator::GE:        DISPATCH(BinGE,      BinaryOperator, S);
      case BinaryOperator::EQ:        DISPATCH(BinEQ,      BinaryOperator, S);
      case BinaryOperator::NE:        DISPATCH(BinNE,      BinaryOperator, S);
        
      case BinaryOperator::And:       DISPATCH(BinAnd,     BinaryOperator, S);
      case BinaryOperator::Xor:       DISPATCH(BinXor,     BinaryOperator, S);
      case BinaryOperator::Or :       DISPATCH(BinOr,      BinaryOperator, S);
      case BinaryOperator::LAnd:      DISPATCH(BinLAnd,    BinaryOperator, S);
      case BinaryOperator::LOr :      DISPATCH(BinLOr,     BinaryOperator, S);
      case BinaryOperator::Assign:    DISPATCH(BinAssign,  BinaryOperator, S);
      case BinaryOperator::MulAssign:
        DISPATCH(BinMulAssign, CompoundAssignOperator, S);
      case BinaryOperator::DivAssign:
        DISPATCH(BinDivAssign, CompoundAssignOperator, S);
      case BinaryOperator::RemAssign:
        DISPATCH(BinRemAssign, CompoundAssignOperator, S);
      case BinaryOperator::AddAssign:
        DISPATCH(BinAddAssign, CompoundAssignOperator, S);
      case BinaryOperator::SubAssign:
        DISPATCH(BinSubAssign, CompoundAssignOperator, S);
      case BinaryOperator::ShlAssign:
        DISPATCH(BinShlAssign, CompoundAssignOperator, S);
      case BinaryOperator::ShrAssign:
        DISPATCH(BinShrAssign, CompoundAssignOperator, S);
      case BinaryOperator::AndAssign:
        DISPATCH(BinAndAssign, CompoundAssignOperator, S);
      case BinaryOperator::OrAssign:
        DISPATCH(BinOrAssign,  CompoundAssignOperator, S);
      case BinaryOperator::XorAssign:
        DISPATCH(BinXorAssign, CompoundAssignOperator, S);
      case BinaryOperator::Comma:     DISPATCH(BinComma,     BinaryOperator, S);
      }
    } else if (UnaryOperator *UnOp = dyn_cast<UnaryOperator>(S)) {
      switch (UnOp->getOpcode()) {
      case UnaryOperator::PostInc:   DISPATCH(UnaryPostInc,   UnaryOperator, S);
      case UnaryOperator::PostDec:   DISPATCH(UnaryPostDec,   UnaryOperator, S);
      case UnaryOperator::PreInc:    DISPATCH(UnaryPreInc,    UnaryOperator, S);
      case UnaryOperator::PreDec:    DISPATCH(UnaryPreDec,    UnaryOperator, S);
      case UnaryOperator::AddrOf:    DISPATCH(UnaryAddrOf,    UnaryOperator, S);
      case UnaryOperator::Deref:     DISPATCH(UnaryDeref,     UnaryOperator, S);
      case UnaryOperator::Plus:      DISPATCH(UnaryPlus,      UnaryOperator, S);
      case UnaryOperator::Minus:     DISPATCH(UnaryMinus,     UnaryOperator, S);
      case UnaryOperator::Not:       DISPATCH(UnaryNot,       UnaryOperator, S);
      case UnaryOperator::LNot:      DISPATCH(UnaryLNot,      UnaryOperator, S);
      case UnaryOperator::Real:      DISPATCH(UnaryReal,      UnaryOperator, S);
      case UnaryOperator::Imag:      DISPATCH(UnaryImag,      UnaryOperator, S);
      case UnaryOperator::Extension: DISPATCH(UnaryExtension, UnaryOperator, S);
      case UnaryOperator::OffsetOf:  DISPATCH(UnaryOffsetOf,  UnaryOperator, S);
      }
    }
    
    // Top switch stmt: dispatch to VisitFooStmt for each FooStmt.
    switch (S->getStmtClass()) {
    case Stmt::NoStmtClass: break;
#define ABSTRACT(STMT)
#define STMT(CLASS, PARENT)                              \
case Stmt::CLASS ## Class: DISPATCH(CLASS, CLASS, S);
#include "clang/AST/StmtNodes.inc"
    }
    
    return false;
  }
  
  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::Visit(QualType T) {
    if (T.isNull())
      return false;
    
    switch (T->getTypeClass()) {
#define ABSTRACT_TYPE(Class, Base)
#define TYPE(Class, Base) \
   case Type::Class: DISPATCH(Class##Type, Class##Type, T.getTypePtr());
#include "clang/AST/TypeNodes.def"
    }
    
    return false;
  }
  
  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::Visit(Decl *D) {
    if (!D)
      return false;
    
    switch (D->getKind()) {
#define ABSTRACT_DECL(Class, Base)
#define DECL(Class, Base) \
    case Decl::Class: DISPATCH(Class##Decl, Class##Decl, D);
#include "clang/AST/DeclNodes.def"
    }
    
    return false;
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitNestedNameSpecifier(
                                                    NestedNameSpecifier *NNS) {
    if (NNS->getPrefix() && 
        getDerived().VisitNestedNameSpecifier(NNS->getPrefix()))
      return true;
    
    switch (NNS->getKind()) {
    case NestedNameSpecifier::Identifier:
    case NestedNameSpecifier::Namespace:
    case NestedNameSpecifier::Global:
      return false;
        
    case NestedNameSpecifier::TypeSpec:
    case NestedNameSpecifier::TypeSpecWithTemplate:
      return Visit(QualType(NNS->getAsType(), 0));
    }
    
    return false;
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitTemplateName(TemplateName Template) {
    if (DependentTemplateName *DTN = Template.getAsDependentTemplateName())
      return DTN->getQualifier() && 
             getDerived().VisitNestedNameSpecifier(DTN->getQualifier());

    if (QualifiedTemplateName *QTN = Template.getAsQualifiedTemplateName())
      return getDerived().VisitNestedNameSpecifier(QTN->getQualifier());
    
    return false;
  }
  
  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitTemplateArgument(
                                                const TemplateArgument &Arg) {
    switch (Arg.getKind()) {       
    case TemplateArgument::Null:
    case TemplateArgument::Declaration:
    case TemplateArgument::Integral:
      return false;
        
    case TemplateArgument::Type:
      return Visit(Arg.getAsType());
        
    case TemplateArgument::Template:
      return getDerived().VisitTemplateName(Arg.getAsTemplate());
        
    case TemplateArgument::Expression:
      return getDerived().Visit(Arg.getAsExpr());
        
    case TemplateArgument::Pack:
      return getDerived().VisitTemplateArguments(Arg.pack_begin(), 
                                                 Arg.pack_size());
    }
    
    return false;
  }
  
  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitTemplateArguments(
                                                  const TemplateArgument *Args, 
                                                            unsigned NumArgs) {
    for (unsigned I = 0; I != NumArgs; ++I)
      if (getDerived().VisitTemplateArgument(Args[I]))
        return true;
    
    return false;
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitStmt(Stmt *Node) {
    for (Stmt::child_iterator C = Node->child_begin(), CEnd = Node->child_end();
         C != CEnd; ++C) {
      if (Visit(*C))
        return true;
    }
    
    return false;
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitType(Type *T) {
    return false;
  }
  
  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitBuiltinType(BuiltinType *T) {
    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitComplexType(ComplexType *T) {
    if (Visit(T->getElementType()))
      return true;
    
    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitPointerType(PointerType *T) {
    if (Visit(T->getPointeeType()))
      return true;
    
    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitBlockPointerType(
                                                         BlockPointerType *T) {
    if (Visit(T->getPointeeType()))
      return true;
    
    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitReferenceType(ReferenceType *T) {
    if (Visit(T->getPointeeType()))
      return true;
    
    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitLValueReferenceType(
                                                      LValueReferenceType *T) {
    return getDerived().VisitReferenceType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitRValueReferenceType(
                                                      RValueReferenceType *T) {
    return getDerived().VisitReferenceType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitMemberPointerType(
                                                        MemberPointerType *T) {
    if (Visit(QualType(T->getClass(), 0)) || Visit(T->getPointeeType()))
      return true;
    
    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitArrayType(ArrayType *T) {
    if (Visit(T->getElementType()))
      return true;
    
    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitConstantArrayType(
                                                        ConstantArrayType *T) {
    return getDerived().VisitArrayType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitIncompleteArrayType(
                                                      IncompleteArrayType *T) {
    return getDerived().VisitArrayType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitVariableArrayType(
                                                        VariableArrayType *T) {
    if (Visit(T->getSizeExpr()))
      return true;
    
    return getDerived().VisitArrayType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitDependentSizedArrayType(
                                                  DependentSizedArrayType *T) {
    if (T->getSizeExpr() && Visit(T->getSizeExpr()))
      return true;
    
    return getDerived().VisitArrayType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitDependentSizedExtVectorType(
                                              DependentSizedExtVectorType *T) {
    if ((T->getSizeExpr() && Visit(T->getSizeExpr())) ||
        Visit(T->getElementType()))
      return true;
    
    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitVectorType(VectorType *T) {
    if (Visit(T->getElementType()))
      return true;
    
    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitExtVectorType(ExtVectorType *T) {
    return getDerived().VisitVectorType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitFunctionType(FunctionType *T) {
    if (Visit(T->getResultType()))
      return true;
    
    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitFunctionNoProtoType(
                                                      FunctionNoProtoType *T) {
    return getDerived().VisitFunctionType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitFunctionProtoType(
                                                        FunctionProtoType *T) {
    for (FunctionProtoType::arg_type_iterator A = T->arg_type_begin(),
                                           AEnd = T->arg_type_end();
         A != AEnd; ++A) {
      if (Visit(*A))
        return true;
    }
    
    for (FunctionProtoType::exception_iterator E = T->exception_begin(),
                                            EEnd = T->exception_end();
         E != EEnd; ++E) {
      if (Visit(*E))
        return true;
    }
    
    return getDerived().VisitFunctionType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitUnresolvedUsingType(
                                                      UnresolvedUsingType *T) {
    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitTypedefType(TypedefType *T) {
    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitTypeOfExprType(TypeOfExprType *T) {
    if (Visit(T->getUnderlyingExpr()))
      return true;
    
    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitTypeOfType(TypeOfType *T) {
    if (Visit(T->getUnderlyingType()))
      return true;
    
    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitDecltypeType(DecltypeType *T) {
    if (Visit(T->getUnderlyingExpr()))
      return true;
    
    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitTagType(TagType *T) {
    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitRecordType(RecordType *T) {
    return getDerived().VisitTagType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitEnumType(EnumType *T) {
    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitTemplateTypeParmType(
                                                      TemplateTypeParmType *T) {
    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitSubstTemplateTypeParmType(
                                                SubstTemplateTypeParmType *T) {
    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitTemplateSpecializationType(
                                               TemplateSpecializationType *T) {
    if (getDerived().VisitTemplateName(T->getTemplateName()) ||
        getDerived().VisitTemplateArguments(T->getArgs(), T->getNumArgs()))
      return true;
    
    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitInjectedClassNameType(
                                                    InjectedClassNameType *T) {
    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitElaboratedType(ElaboratedType *T) {
    if (T->getQualifier() &&
        getDerived().VisitNestedNameSpecifier(T->getQualifier()))
      return true;
    if (Visit(T->getNamedType()))
      return true;

    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitDependentNameType(
                                                        DependentNameType *T) {
    if (T->getQualifier() && 
        getDerived().VisitNestedNameSpecifier(T->getQualifier()))
      return true;
    
    if (T->getTemplateId() &&
        getDerived().VisitTemplateSpecializationType(
                  const_cast<TemplateSpecializationType *>(T->getTemplateId())))
      return true;
    
    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitObjCInterfaceType(
                                                        ObjCInterfaceType *T) {
    return getDerived().VisitType(T);
  }

  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitObjCObjectPointerType(
                                                    ObjCObjectPointerType *T) {
    if (Visit(T->getPointeeType()))
      return true;
    
    return getDerived().VisitType(T);
  }
  
  template<typename Derived>
  bool RecursiveASTVisitor<Derived>::VisitDecl(Decl *D) {
    // FIXME: This is a lazy shortcut, to avoid having to provide recursive 
    // Visit*Decl defaults for every declaration kind.
    if (FunctionDecl *Function = dyn_cast<FunctionDecl>(D)) {
      if (Function->isThisDeclarationADefinition())
        return Visit(Function->getBody());
      
      return false;
    } 
    
    if (DeclContext *DC = dyn_cast<DeclContext>(D)) {
      for (DeclContext::decl_iterator Child = DC->decls_begin(),
                                   ChildEnd = DC->decls_end();
           Child != ChildEnd; ++Child)
        if (Visit(*Child))
          return true;
      
      return false;
    }
    
    return false;
  }

} // end namespace clang

#endif // LLVM_CLANG_AST_RECURSIVEASTVISITOR_H
