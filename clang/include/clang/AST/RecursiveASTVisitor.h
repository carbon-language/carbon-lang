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

// The following three macros are used for meta programming.  The code
// using them is responsible for defining macro OPERATOR().

// All unary operators.
#define UNARYOP_LIST()                          \
  OPERATOR(PostInc)   OPERATOR(PostDec)         \
  OPERATOR(PreInc)    OPERATOR(PreDec)          \
  OPERATOR(AddrOf)    OPERATOR(Deref)           \
  OPERATOR(Plus)      OPERATOR(Minus)           \
  OPERATOR(Not)       OPERATOR(LNot)            \
  OPERATOR(Real)      OPERATOR(Imag)            \
  OPERATOR(Extension) OPERATOR(OffsetOf)

// All binary operators (excluding compound assign operators).
#define BINOP_LIST() \
  OPERATOR(PtrMemD)              OPERATOR(PtrMemI)    \
  OPERATOR(Mul)   OPERATOR(Div)  OPERATOR(Rem)        \
  OPERATOR(Add)   OPERATOR(Sub)  OPERATOR(Shl)        \
  OPERATOR(Shr)                                       \
                                                      \
  OPERATOR(LT)    OPERATOR(GT)   OPERATOR(LE)         \
  OPERATOR(GE)    OPERATOR(EQ)   OPERATOR(NE)         \
  OPERATOR(And)   OPERATOR(Xor)  OPERATOR(Or)         \
  OPERATOR(LAnd)  OPERATOR(LOr)                       \
                                                      \
  OPERATOR(Assign)                                    \
  OPERATOR(Comma)

// All compound assign operators.
#define CAO_LIST()                                                      \
  OPERATOR(Mul) OPERATOR(Div) OPERATOR(Rem) OPERATOR(Add) OPERATOR(Sub) \
  OPERATOR(Shl) OPERATOR(Shr) OPERATOR(And) OPERATOR(Or)  OPERATOR(Xor)

namespace clang {

// A helper macro to implement short-circuiting when recursing.  It
// invokes CALL_EXPR, which must be a method call, on the derived
// object (s.t. a user of RecursiveASTVisitor can override the method
// in CALL_EXPR).
#define TRY_TO(CALL_EXPR) \
  do { if (!getDerived().CALL_EXPR) return false; } while (0)

/// \brief A class that does preorder depth-first traversal on the
/// entire Clang AST and visits each node.
///
/// This class performs three distinct tasks:
///   1. traverse the AST (i.e. go to each node);
///   2. at a given node, walk up the class hierarchy, starting from
///      the node's dynamic type, until the top-most class (e.g. Stmt,
///      Decl, or Type) is reached.
///   3. given a (node, class) combination, where 'class' is some base
///      class of the dynamic type of 'node', call a user-overridable
///      function to actually visit the node.
///
/// These tasks are done by three groups of methods, respectively:
///   1. TraverseDecl(Decl* x) does task #1.  It is the entry point
///      for traversing an AST rooted at x.  This method simply
///      dispatches (i.e. forwards) to TraverseFoo(Foo* x) where Foo
///      is the dynamic type of *x, which calls WalkUpFromFoo(x) and
///      then recursively visits the child nodes of x.
///      TraverseStmt(Stmt* x) and TraverseType(QualType x) work
///      similarly.
///   2. WalkUpFromFoo(Foo* x) does task #2.  It does not try to visit
///      any child node of x.  Instead, it first calls WalkUpFromBar(x)
///      where Bar is the direct parent class of Foo (unless Foo has
///      no parent), and then calls VisitFoo(x) (see the next list item).
///   3. VisitFoo(Foo* x) does task #3.
///
/// These three method groups are tiered (Traverse* > WalkUpFrom* >
/// Visit*).  A method (e.g. Traverse*) may call methods from the same
/// tier (e.g. other Traverse*) or one tier lower (e.g. WalkUpFrom*).
/// It may not call methods from a higher tier.
///
/// Note that since WalkUpFromFoo() calls WalkUpFromBar() (where Bar
/// is Foo's super class) before calling VisitFoo(), the result is
/// that the Visit*() methods for a given node are called in the
/// top-down order (e.g. for a node of type NamedDecl, the order will
/// be VisitDecl(), VisitNamedDecl(), and then VisitNamespaceDecl()).
///
/// This scheme guarantees that all Visit*() calls for the same AST
/// node are grouped together.  In other words, Visit*() methods for
/// different nodes are never interleaved.
///
/// Clients of this visitor should subclass the visitor (providing
/// themselves as the template argument, using the curiously recurring
/// template pattern) and override any of the Traverse*, WalkUpFrom*,
/// and Visit* methods for declarations, types, statements,
/// expressions, or other AST nodes where the visitor should customize
/// behavior.  Most users only need to override Visit*.  Advanced
/// users may override Traverse* and WalkUpFrom* to implement custom
/// traversal strategies.  Returning false from one of these overridden
/// functions will abort the entire traversal.
template<typename Derived>
class RecursiveASTVisitor {
public:
  /// \brief Return a reference to the derived class.
  Derived &getDerived() { return *static_cast<Derived*>(this); }

  /// \brief Recursively visit a statement or expression, by
  /// dispatching to Visit*() based on the argument's dynamic type.
  /// This is NOT meant to be overridden by a subclass.
  ///
  /// \returns false if the visitation was terminated early, true
  /// otherwise (including when the argument is NULL).
  bool TraverseStmt(Stmt *S);

  /// \brief Recursively visit a type, by dispatching to
  /// Visit*Type() based on the argument's getTypeClass() property.
  /// This is NOT meant to be overridden by a subclass.
  ///
  /// \returns false if the visitation was terminated early, true
  /// otherwise (including when the argument is a Null type).
  bool TraverseType(QualType T);

  /// \brief Recursively visit a declaration, by dispatching to
  /// Visit*Decl() based on the argument's dynamic type.  This is
  /// NOT meant to be overridden by a subclass.
  ///
  /// \returns false if the visitation was terminated early, true
  /// otherwise (including when the argument is NULL).
  bool TraverseDecl(Decl *D);

  /// \brief Recursively visit a C++ nested-name-specifier.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  bool TraverseNestedNameSpecifier(NestedNameSpecifier *NNS);

  /// \brief Recursively visit a template name and dispatch to the
  /// appropriate method.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  bool TraverseTemplateName(TemplateName Template);

  /// \brief Recursively visit a template argument and dispatch to the
  /// appropriate method for the argument type.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  // FIXME: take a TemplateArgumentLoc instead.
  bool TraverseTemplateArgument(const TemplateArgument &Arg);

  /// \brief Recursively visit a set of template arguments.
  /// This can be overridden by a subclass, but it's not expected that
  /// will be needed -- this visitor always dispatches to another.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  // FIXME: take a TemplateArgumentLoc* (or TemplateArgumentListInfo) instead.
  bool TraverseTemplateArguments(const TemplateArgument *Args,
                                 unsigned NumArgs);

  // ---- Methods on Stmts ----

  // Declare Traverse*() for all concrete Stmt classes.
#define ABSTRACT_STMT(STMT)
#define STMT(CLASS, PARENT)                                     \
  bool Traverse##CLASS(CLASS *S);
#include "clang/AST/StmtNodes.inc"
  // The above header #undefs ABSTRACT_STMT and STMT upon exit.

  // Define WalkUpFrom*() and empty Visit*() for all Stmt classes.
  bool WalkUpFromStmt(Stmt *S) { return getDerived().VisitStmt(S); }
  bool VisitStmt(Stmt *S) { return true; }
#define STMT(CLASS, PARENT)                                     \
  bool WalkUpFrom##CLASS(CLASS *S) {                            \
    TRY_TO(WalkUpFrom##PARENT(S));                              \
    TRY_TO(Visit##CLASS(S));                                    \
    return true;                                                \
  }                                                             \
  bool Visit##CLASS(CLASS *S) { return true; }
#include "clang/AST/StmtNodes.inc"

  // Define Traverse*(), WalkUpFrom*(), and Visit*() for unary
  // operator methods.  Unary operators are not classes in themselves
  // (they're all opcodes in UnaryOperator) but do have visitors.
#define OPERATOR(NAME)                                           \
  bool TraverseUnary##NAME(UnaryOperator *S) {                  \
    TRY_TO(WalkUpFromUnary##NAME(S));                           \
    TRY_TO(TraverseStmt(S->getSubExpr()));                      \
    return true;                                                \
  }                                                             \
  bool WalkUpFromUnary##NAME(UnaryOperator *S) {                \
    TRY_TO(WalkUpFromUnaryOperator(S));                         \
    TRY_TO(VisitUnary##NAME(S));                                \
    return true;                                                \
  }                                                             \
  bool VisitUnary##NAME(UnaryOperator *S) { return true; }

  UNARYOP_LIST()
#undef OPERATOR

  // Define Traverse*(), WalkUpFrom*(), and Visit*() for binary
  // operator methods.  Binary operators are not classes in themselves
  // (they're all opcodes in BinaryOperator) but do have visitors.
#define GENERAL_BINOP_FALLBACK(NAME, BINOP_TYPE)                \
  bool TraverseBin##NAME(BINOP_TYPE *S) {                       \
    TRY_TO(WalkUpFromBin##NAME(S));                             \
    TRY_TO(TraverseStmt(S->getLHS()));                          \
    TRY_TO(TraverseStmt(S->getRHS()));                          \
    return true;                                                \
  }                                                             \
  bool WalkUpFromBin##NAME(BINOP_TYPE *S) {                     \
    TRY_TO(WalkUpFrom##BINOP_TYPE(S));                          \
    TRY_TO(VisitBin##NAME(S));                                  \
    return true;                                                \
  }                                                             \
  bool VisitBin##NAME(BINOP_TYPE *S) { return true; }

#define OPERATOR(NAME) GENERAL_BINOP_FALLBACK(NAME, BinaryOperator)
  BINOP_LIST()
#undef OPERATOR

  // Define Traverse*(), WalkUpFrom*(), and Visit*() for compound
  // assignment methods.  Compound assignment operators are not
  // classes in themselves (they're all opcodes in
  // CompoundAssignOperator) but do have visitors.
#define OPERATOR(NAME) \
  GENERAL_BINOP_FALLBACK(NAME##Assign, CompoundAssignOperator)

  CAO_LIST()
#undef OPERATOR
#undef GENERAL_BINOP_FALLBACK

  // ---- Methods on Types ----
  // FIXME: revamp to take TypeLoc's rather than Types.

  // Declare Traverse*() for all concrete Type classes.
#define ABSTRACT_TYPE(CLASS, BASE)
#define TYPE(CLASS, BASE) \
  bool Traverse##CLASS##Type(CLASS##Type *T);
#include "clang/AST/TypeNodes.def"
  // The above header #undefs ABSTRACT_TYPE and TYPE upon exit.

  // Define WalkUpFrom*() and empty Visit*() for all Type classes.
  bool WalkUpFromType(Type *T) { return getDerived().VisitType(T); }
  bool VisitType(Type *T) { return true; }
#define TYPE(CLASS, BASE)                                       \
  bool WalkUpFrom##CLASS##Type(CLASS##Type *T) {                \
    TRY_TO(WalkUpFrom##BASE(T));                                \
    TRY_TO(Visit##CLASS##Type(T));                              \
    return true;                                                \
  }                                                             \
  bool Visit##CLASS##Type(CLASS##Type *T) { return true; }
#include "clang/AST/TypeNodes.def"

  // ---- Methods on Decls ----

  // Declare Traverse*() for all concrete Decl classes.
#define ABSTRACT_DECL(DECL)
#define DECL(CLASS, BASE) \
  bool Traverse##CLASS##Decl(CLASS##Decl *D);
#include "clang/AST/DeclNodes.inc"
  // The above header #undefs ABSTRACT_DECL and DECL upon exit.

  // Define WalkUpFrom*() and empty Visit*() for all Decl classes.
  bool WalkUpFromDecl(Decl *D) { return getDerived().VisitDecl(D); }
  bool VisitDecl(Decl *D) { return true; }
#define DECL(CLASS, BASE)                                       \
  bool WalkUpFrom##CLASS##Decl(CLASS##Decl *D) {                \
    TRY_TO(WalkUpFrom##BASE(D));                                \
    TRY_TO(Visit##CLASS##Decl(D));                              \
    return true;                                                \
  }                                                             \
  bool Visit##CLASS##Decl(CLASS##Decl *D) { return true; }
#include "clang/AST/DeclNodes.inc"

private:
  // These are helper methods used by more than one Traverse* method.
  bool TraverseTemplateParameterListHelper(TemplateParameterList *TPL);
  bool TraverseTemplateArgumentLocsHelper(const TemplateArgumentLoc *TAL,
                                          unsigned Count);
  bool TraverseRecordHelper(RecordDecl *D);
  bool TraverseCXXRecordHelper(CXXRecordDecl *D);
  bool TraverseDeclaratorHelper(DeclaratorDecl *D);
  bool TraverseFunctionHelper(FunctionDecl *D);
  bool TraverseVarHelper(VarDecl *D);
};

#define DISPATCH(NAME, CLASS, VAR) \
  return getDerived().Traverse##NAME(static_cast<CLASS*>(VAR))

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseStmt(Stmt *S) {
  if (!S)
    return true;

  // If we have a binary expr, dispatch to the subcode of the binop.  A smart
  // optimizer (e.g. LLVM) will fold this comparison into the switch stmt
  // below.
  if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(S)) {
    switch (BinOp->getOpcode()) {
#define OPERATOR(NAME) \
    case BinaryOperator::NAME: DISPATCH(Bin##PtrMemD, BinaryOperator, S);

    BINOP_LIST()
#undef OPERATOR
#undef BINOP_LIST

#define OPERATOR(NAME)                                          \
    case BinaryOperator::NAME##Assign:                          \
      DISPATCH(Bin##NAME##Assign, CompoundAssignOperator, S);

    CAO_LIST()
#undef OPERATOR
#undef CAO_LIST
    }
  } else if (UnaryOperator *UnOp = dyn_cast<UnaryOperator>(S)) {
    switch (UnOp->getOpcode()) {
#define OPERATOR(NAME)                                                  \
    case UnaryOperator::NAME: DISPATCH(Unary##NAME, UnaryOperator, S);

    UNARYOP_LIST()
#undef OPERATOR
#undef UNARYOP_LIST
    }
  }

  // Top switch stmt: dispatch to TraverseFooStmt for each concrete FooStmt.
  switch (S->getStmtClass()) {
  case Stmt::NoStmtClass: break;
#define ABSTRACT_STMT(STMT)
#define STMT(CLASS, PARENT) \
  case Stmt::CLASS##Class: DISPATCH(CLASS, CLASS, S);
#include "clang/AST/StmtNodes.inc"
  }

  return true;
}

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseType(QualType T) {
  if (T.isNull())
    return true;

  switch (T->getTypeClass()) {
#define ABSTRACT_TYPE(CLASS, BASE)
#define TYPE(CLASS, BASE) \
  case Type::CLASS: DISPATCH(CLASS##Type, CLASS##Type, T.getTypePtr());
#include "clang/AST/TypeNodes.def"
  }

  return true;
}

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseDecl(Decl *D) {
  if (!D)
    return true;

  switch (D->getKind()) {
#define ABSTRACT_DECL(DECL)
#define DECL(CLASS, BASE) \
  case Decl::CLASS: DISPATCH(CLASS##Decl, CLASS##Decl, D);
#include "clang/AST/DeclNodes.inc"
 }

  return true;
}

#undef DISPATCH

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseNestedNameSpecifier(
                                                    NestedNameSpecifier *NNS) {
  if (!NNS)
    return true;

  if (NNS->getPrefix())
    TRY_TO(TraverseNestedNameSpecifier(NNS->getPrefix()));

  switch (NNS->getKind()) {
  case NestedNameSpecifier::Identifier:
  case NestedNameSpecifier::Namespace:
  case NestedNameSpecifier::Global:
    return true;

  case NestedNameSpecifier::TypeSpec:
  case NestedNameSpecifier::TypeSpecWithTemplate:
    TRY_TO(TraverseType(QualType(NNS->getAsType(), 0)));
  }

  return true;
}

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseTemplateName(TemplateName Template) {
  if (DependentTemplateName *DTN = Template.getAsDependentTemplateName())
    TRY_TO(TraverseNestedNameSpecifier(DTN->getQualifier()));
  else if (QualifiedTemplateName *QTN = Template.getAsQualifiedTemplateName())
    TRY_TO(TraverseNestedNameSpecifier(QTN->getQualifier()));

  return true;
}

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseTemplateArgument(
                                                const TemplateArgument &Arg) {
  switch (Arg.getKind()) {
  case TemplateArgument::Null:
  case TemplateArgument::Declaration:
  case TemplateArgument::Integral:
    return true;

  case TemplateArgument::Type:
    return getDerived().TraverseType(Arg.getAsType());

  case TemplateArgument::Template:
    return getDerived().TraverseTemplateName(Arg.getAsTemplate());

  case TemplateArgument::Expression:
    return getDerived().TraverseStmt(Arg.getAsExpr());

  case TemplateArgument::Pack:
    return getDerived().TraverseTemplateArguments(Arg.pack_begin(),
                                                  Arg.pack_size());
  }

  return true;
}

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseTemplateArguments(
                                                  const TemplateArgument *Args,
                                                            unsigned NumArgs) {
  for (unsigned I = 0; I != NumArgs; ++I) {
    TRY_TO(TraverseTemplateArgument(Args[I]));
  }

  return true;
}

// ----------------- Type traversal -----------------

// This macro makes available a variable T, the passed-in type.
#define DEF_TRAVERSE_TYPE(TYPE, CODE)                     \
  template<typename Derived>                                           \
  bool RecursiveASTVisitor<Derived>::Traverse##TYPE (TYPE *T) {        \
    TRY_TO(WalkUpFrom##TYPE (T));                                      \
    { CODE; }                                                          \
    return true;                                                       \
  }

DEF_TRAVERSE_TYPE(BuiltinType, { })

DEF_TRAVERSE_TYPE(ComplexType, {
    TRY_TO(TraverseType(T->getElementType()));
  })

DEF_TRAVERSE_TYPE(PointerType, {
    TRY_TO(TraverseType(T->getPointeeType()));
  })

DEF_TRAVERSE_TYPE(BlockPointerType, {
    TRY_TO(TraverseType(T->getPointeeType()));
  })

DEF_TRAVERSE_TYPE(LValueReferenceType, {
    TRY_TO(TraverseType(T->getPointeeType()));
  })

DEF_TRAVERSE_TYPE(RValueReferenceType, {
    TRY_TO(TraverseType(T->getPointeeType()));
  })

DEF_TRAVERSE_TYPE(MemberPointerType, {
    TRY_TO(TraverseType(QualType(T->getClass(), 0)));
    TRY_TO(TraverseType(T->getPointeeType()));
  })

DEF_TRAVERSE_TYPE(ConstantArrayType, {
    TRY_TO(TraverseType(T->getElementType()));
  })

DEF_TRAVERSE_TYPE(IncompleteArrayType, {
    TRY_TO(TraverseType(T->getElementType()));
  })

DEF_TRAVERSE_TYPE(VariableArrayType, {
    TRY_TO(TraverseType(T->getElementType()));
    TRY_TO(TraverseStmt(T->getSizeExpr()));
  })

DEF_TRAVERSE_TYPE(DependentSizedArrayType, {
    TRY_TO(TraverseType(T->getElementType()));
    if (T->getSizeExpr())
      TRY_TO(TraverseStmt(T->getSizeExpr()));
  })

DEF_TRAVERSE_TYPE(DependentSizedExtVectorType, {
    if (T->getSizeExpr())
      TRY_TO(TraverseStmt(T->getSizeExpr()));
    TRY_TO(TraverseType(T->getElementType()));
  })

DEF_TRAVERSE_TYPE(VectorType, {
    TRY_TO(TraverseType(T->getElementType()));
  })

DEF_TRAVERSE_TYPE(ExtVectorType, { })

DEF_TRAVERSE_TYPE(FunctionNoProtoType, {
    TRY_TO(TraverseType(T->getResultType()));
  })

DEF_TRAVERSE_TYPE(FunctionProtoType, {
    TRY_TO(TraverseType(T->getResultType()));

    for (FunctionProtoType::arg_type_iterator A = T->arg_type_begin(),
                                           AEnd = T->arg_type_end();
         A != AEnd; ++A) {
      TRY_TO(TraverseType(*A));
    }

    for (FunctionProtoType::exception_iterator E = T->exception_begin(),
                                            EEnd = T->exception_end();
         E != EEnd; ++E) {
      TRY_TO(TraverseType(*E));
    }
  })

DEF_TRAVERSE_TYPE(UnresolvedUsingType, { })
DEF_TRAVERSE_TYPE(TypedefType, { })

DEF_TRAVERSE_TYPE(TypeOfExprType, {
    TRY_TO(TraverseStmt(T->getUnderlyingExpr()));
  })

DEF_TRAVERSE_TYPE(TypeOfType, {
    TRY_TO(TraverseType(T->getUnderlyingType()));
  })

DEF_TRAVERSE_TYPE(DecltypeType, {
    TRY_TO(TraverseStmt(T->getUnderlyingExpr()));
  })

DEF_TRAVERSE_TYPE(RecordType, { })
DEF_TRAVERSE_TYPE(EnumType, { })
DEF_TRAVERSE_TYPE(TemplateTypeParmType, { })
DEF_TRAVERSE_TYPE(SubstTemplateTypeParmType, { })

DEF_TRAVERSE_TYPE(TemplateSpecializationType, {
    TRY_TO(TraverseTemplateName(T->getTemplateName()));
    TRY_TO(TraverseTemplateArguments(T->getArgs(), T->getNumArgs()));
  })

DEF_TRAVERSE_TYPE(InjectedClassNameType, { })

DEF_TRAVERSE_TYPE(ElaboratedType, {
    if (T->getQualifier()) {
      TRY_TO(TraverseNestedNameSpecifier(T->getQualifier()));
    }
    TRY_TO(TraverseType(T->getNamedType()));
  })

DEF_TRAVERSE_TYPE(DependentNameType, {
    TRY_TO(TraverseNestedNameSpecifier(T->getQualifier()));
  })

DEF_TRAVERSE_TYPE(DependentTemplateSpecializationType, {
    TRY_TO(TraverseNestedNameSpecifier(T->getQualifier()));
    TRY_TO(TraverseTemplateArguments(T->getArgs(), T->getNumArgs()));
  })

DEF_TRAVERSE_TYPE(ObjCInterfaceType, { })

DEF_TRAVERSE_TYPE(ObjCObjectType, {
    // We have to watch out here because an ObjCInterfaceType's base
    // type is itself.
    if (T->getBaseType().getTypePtr() != T)
      TRY_TO(TraverseType(T->getBaseType()));
  })

DEF_TRAVERSE_TYPE(ObjCObjectPointerType, {
    TRY_TO(TraverseType(T->getPointeeType()));
  })

#undef DEF_TRAVERSE_TYPE

// ----------------- Decl traversal -----------------
//
// For a Decl, we automate (in the DEF_TRAVERSE_DECL macro) traversing
// the children that come from the DeclContext associated with it.
// Therefore each Traverse* only needs to worry about children other
// than those.

// This macro makes available a variable D, the passed-in decl.
#define DEF_TRAVERSE_DECL(DECL, CODE)                           \
template<typename Derived>                                      \
bool RecursiveASTVisitor<Derived>::Traverse##DECL (DECL *D) {   \
  TRY_TO(WalkUpFrom##DECL (D));                                 \
  { CODE; }                                                     \
  if (DeclContext *DC = dyn_cast<DeclContext>(D)) {             \
    for (DeclContext::decl_iterator Child = DC->decls_begin(),  \
                                 ChildEnd = DC->decls_end();    \
         Child != ChildEnd; ++Child)                            \
      TRY_TO(TraverseDecl(*Child));                             \
  }                                                             \
  return true;                                                  \
}

DEF_TRAVERSE_DECL(AccessSpecDecl, { })

DEF_TRAVERSE_DECL(BlockDecl, {
    // We don't traverse nodes in param_begin()/param_end(), as they
    // appear in decls_begin()/decls_end() and thus are handled by the
    // DEF_TRAVERSE_DECL macro already.
    TRY_TO(TraverseStmt(D->getBody()));
  })

DEF_TRAVERSE_DECL(FileScopeAsmDecl, {
    TRY_TO(TraverseStmt(D->getAsmString()));
  })

DEF_TRAVERSE_DECL(FriendDecl, {
    TRY_TO(TraverseDecl(D->getFriendDecl()));
  })

DEF_TRAVERSE_DECL(FriendTemplateDecl, {
    TRY_TO(TraverseDecl(D->getFriendDecl()));
    for (unsigned I = 0, E = D->getNumTemplateParameters(); I < E; ++I) {
      TemplateParameterList *TPL = D->getTemplateParameterList(I);
      for (TemplateParameterList::iterator ITPL = TPL->begin(),
                                           ETPL = TPL->end();
           ITPL != ETPL; ++ITPL) {
        TRY_TO(TraverseDecl(*ITPL));
      }
    }
  })

DEF_TRAVERSE_DECL(LinkageSpecDecl, { })

DEF_TRAVERSE_DECL(ObjCClassDecl, {
    // FIXME: implement this
  })

DEF_TRAVERSE_DECL(ObjCForwardProtocolDecl, {
    // FIXME: implement this
  })

DEF_TRAVERSE_DECL(ObjCPropertyImplDecl, {
    // FIXME: implement this
  })

DEF_TRAVERSE_DECL(StaticAssertDecl, {
    TRY_TO(TraverseStmt(D->getAssertExpr()));
    TRY_TO(TraverseStmt(D->getMessage()));
  })

DEF_TRAVERSE_DECL(TranslationUnitDecl, {
    // Code in an unnamed namespace shows up automatically in
    // decls_begin()/decls_end().  Thus we don't need to recurse on
    // D->getAnonymousNamespace().
  })

DEF_TRAVERSE_DECL(NamespaceAliasDecl, {
    // We shouldn't traverse an aliased namespace, since it will be
    // defined (and, therefore, traversed) somewhere else.
    //
    // This return statement makes sure the traversal of nodes in
    // decls_begin()/decls_end() (done in the DEF_TRAVERSE_DECL macro)
    // is skipped - don't remove it.
    return true;
  })

DEF_TRAVERSE_DECL(NamespaceDecl, {
    // Code in an unnamed namespace shows up automatically in
    // decls_begin()/decls_end().  Thus we don't need to recurse on
    // D->getAnonymousNamespace().
  })

DEF_TRAVERSE_DECL(ObjCCompatibleAliasDecl, {
    // FIXME: implement
  })

DEF_TRAVERSE_DECL(ObjCCategoryDecl, {
    // FIXME: implement
  })

DEF_TRAVERSE_DECL(ObjCCategoryImplDecl, {
    // FIXME: implement
  })

DEF_TRAVERSE_DECL(ObjCImplementationDecl, {
    // FIXME: implement
  })

DEF_TRAVERSE_DECL(ObjCInterfaceDecl, {
    // FIXME: implement
  })

DEF_TRAVERSE_DECL(ObjCProtocolDecl, {
    // FIXME: implement
  })

DEF_TRAVERSE_DECL(ObjCMethodDecl, {
    // FIXME: implement
  })

DEF_TRAVERSE_DECL(ObjCPropertyDecl, {
    // FIXME: implement
  })

DEF_TRAVERSE_DECL(UsingDecl, {
    TRY_TO(TraverseNestedNameSpecifier(D->getTargetNestedNameDecl()));
    for (UsingDecl::shadow_iterator I = D->shadow_begin(), E = D->shadow_end();
         I != E; ++I) {
      TRY_TO(TraverseDecl(*I));
    }
  })

DEF_TRAVERSE_DECL(UsingDirectiveDecl, {
    TRY_TO(TraverseNestedNameSpecifier(D->getQualifier()));
  })

DEF_TRAVERSE_DECL(UsingShadowDecl, { })

// A helper method for TemplateDecl's children.
template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseTemplateParameterListHelper(
    TemplateParameterList *TPL) {
  if (TPL) {
    for (TemplateParameterList::iterator I = TPL->begin(), E = TPL->end();
         I != E; ++I) {
      TRY_TO(TraverseDecl(*I));
    }
  }
  return true;
}

DEF_TRAVERSE_DECL(ClassTemplateDecl, {
    TRY_TO(TraverseDecl(D->getTemplatedDecl()));
    TRY_TO(TraverseTemplateParameterListHelper(D->getTemplateParameters()));
    // We should not traverse the specializations/partial
    // specializations.  Those will show up in other contexts.
    // getInstantiatedFromMemberTemplate() is just a link from a
    // template instantiation back to the template from which it was
    // instantiated, and thus should not be traversed either.
  })

DEF_TRAVERSE_DECL(FunctionTemplateDecl, {
    TRY_TO(TraverseDecl(D->getTemplatedDecl()));
    TRY_TO(TraverseTemplateParameterListHelper(D->getTemplateParameters()));
  })

DEF_TRAVERSE_DECL(TemplateTemplateParmDecl, {
    // D is the "T" in something like
    //   template <template <typename> class T> class container { };
    TRY_TO(TraverseDecl(D->getTemplatedDecl()));
    // FIXME: pass along the full TemplateArgumentLoc
    if (D->hasDefaultArgument()) {
      TRY_TO(TraverseTemplateArgument(D->getDefaultArgument().getArgument()));
    }
    TRY_TO(TraverseTemplateParameterListHelper(D->getTemplateParameters()));
  })

DEF_TRAVERSE_DECL(TemplateTypeParmDecl, {
    // D is the "T" in something like "template<typename T> class vector;"
    if (D->hasDefaultArgument())
      TRY_TO(TraverseType(D->getDefaultArgument()));
    if (D->getTypeForDecl())
      TRY_TO(TraverseType(QualType(D->getTypeForDecl(), 0)));
  })

DEF_TRAVERSE_DECL(TypedefDecl, {
    // FIXME: I *think* this returns the right type (D's immediate typedef)
    TRY_TO(TraverseType(D->getUnderlyingType()));
    // We shouldn't traverse D->getTypeForDecl(); it's a result of
    // declaring the typedef, not something that was written in the
    // source.
  })

DEF_TRAVERSE_DECL(UnresolvedUsingTypenameDecl, {
    // A dependent using declaration which was marked with 'typename'.
    //   template<class T> class A : public B<T> { using typename B<T>::foo; };
    TRY_TO(TraverseNestedNameSpecifier(D->getTargetNestedNameSpecifier()));
    // We shouldn't traverse D->getTypeForDecl(); it's a result of
    // declaring the type, not something that was written in the
    // source.
  })

DEF_TRAVERSE_DECL(EnumDecl, {
    if (D->getTypeForDecl())
      TRY_TO(TraverseType(QualType(D->getTypeForDecl(), 0)));

    TRY_TO(TraverseNestedNameSpecifier(D->getQualifier()));
    // The enumerators are already traversed by
    // decls_begin()/decls_end().
  })


// Helper methods for RecordDecl and its children.
template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseRecordHelper(
    RecordDecl *D) {
  // We shouldn't traverse D->getTypeForDecl(); it's a result of
  // declaring the type, not something that was written in the source.
  //
  // The anonymous struct or union object is the variable or field
  // whose type is the anonymous struct or union.  We shouldn't
  // traverse D->getAnonymousStructOrUnionObject(), as it's not
  // something that is explicitly written in the source.
  TRY_TO(TraverseNestedNameSpecifier(D->getQualifier()));
  return true;
}

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseCXXRecordHelper(
    CXXRecordDecl *D) {
  if (!TraverseRecordHelper(D))
    return false;
  if (D->hasDefinition()) {
    for (CXXRecordDecl::base_class_iterator I = D->bases_begin(),
                                            E = D->bases_end();
         I != E; ++I) {
      TRY_TO(TraverseType(I->getType()));
    }
    // We don't traverse the friends or the conversions, as they are
    // already in decls_begin()/decls_end().
  }
  return true;
}

DEF_TRAVERSE_DECL(RecordDecl, {
    TRY_TO(TraverseRecordHelper(D));
  })

DEF_TRAVERSE_DECL(CXXRecordDecl, {
    TRY_TO(TraverseCXXRecordHelper(D));
  })

DEF_TRAVERSE_DECL(ClassTemplateSpecializationDecl, {
    // FIXME: we probably want to traverse the TemplateArgumentLoc for
    // explicitly-written specializations and instantiations, rather
    // than the computed template arguments.
    const TemplateArgumentList &TAL = D->getTemplateArgs();
    for (unsigned I = 0; I < TAL.size(); ++I) {
      TRY_TO(TraverseTemplateArgument(TAL.get(I)));
    }
    TRY_TO(TraverseCXXRecordHelper(D));
  })

template <typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseTemplateArgumentLocsHelper(
    const TemplateArgumentLoc *TAL, unsigned Count) {
  for (unsigned I = 0; I < Count; ++I) {
    // FIXME: pass along the loc information to this callback as well?
    TRY_TO(TraverseTemplateArgument(TAL[I].getArgument()));
  }
  return true;
}

DEF_TRAVERSE_DECL(ClassTemplatePartialSpecializationDecl, {
    // The partial specialization.
    TemplateParameterList *TPL = D->getTemplateParameters();
    if (TPL) {
      for (TemplateParameterList::iterator I = TPL->begin(), E = TPL->end();
           I != E; ++I) {
        TRY_TO(TraverseDecl(*I));
      }
    }
    // The args that remains unspecialized.
    TRY_TO(TraverseTemplateArgumentLocsHelper(
        D->getTemplateArgsAsWritten(), D->getNumTemplateArgsAsWritten()));

    // Don't need the ClassTemplatePartialSpecializationHelper, even
    // though that's our parent class -- we already visit all the
    // template args here.
    TRY_TO(TraverseCXXRecordHelper(D));
  })

DEF_TRAVERSE_DECL(EnumConstantDecl, {
    TRY_TO(TraverseStmt(D->getInitExpr()));
  })

DEF_TRAVERSE_DECL(UnresolvedUsingValueDecl, {
    // Like UnresolvedUsingTypenameDecl, but without the 'typename':
    //    template <class T> Class A : public Base<T> { using Base<T>::foo; };
    TRY_TO(TraverseNestedNameSpecifier(D->getTargetNestedNameSpecifier()));
  })

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseDeclaratorHelper(DeclaratorDecl *D) {
  TRY_TO(TraverseNestedNameSpecifier(D->getQualifier()));
  if (D->getTypeSourceInfo())
    TRY_TO(TraverseType(D->getTypeSourceInfo()->getType()));
  return true;
}

DEF_TRAVERSE_DECL(FieldDecl, {
    TRY_TO(TraverseDeclaratorHelper(D));
    if (D->isBitField())
      TRY_TO(TraverseStmt(D->getBitWidth()));
  })

DEF_TRAVERSE_DECL(ObjCAtDefsFieldDecl, {
    TRY_TO(TraverseDeclaratorHelper(D));
    if (D->isBitField())
      TRY_TO(TraverseStmt(D->getBitWidth()));
    // FIXME: implement the rest.
  })

DEF_TRAVERSE_DECL(ObjCIvarDecl, {
    TRY_TO(TraverseDeclaratorHelper(D));
    if (D->isBitField())
      TRY_TO(TraverseStmt(D->getBitWidth()));
    // FIXME: implement the rest.
  })

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseFunctionHelper(FunctionDecl *D) {
  // The function parameters will be traversed by
  // decls_begin/decls_end.  We don't traverse D->getResultType(), as
  // it's part of the TypeSourceInfo for the function declaration.
  TRY_TO(TraverseDeclaratorHelper(D));
  if (D->isThisDeclarationADefinition()) {
    TRY_TO(TraverseStmt(D->getBody()));
  }
  return true;
}

DEF_TRAVERSE_DECL(FunctionDecl, {
    TRY_TO(TraverseFunctionHelper(D));
  })

DEF_TRAVERSE_DECL(CXXMethodDecl, {
    // FIXME: verify we don't need anything more than the superclass does.
    // If we do, change 3 subclasses to call our helper method, instead.
    TRY_TO(TraverseFunctionHelper(D));
  })

DEF_TRAVERSE_DECL(CXXConstructorDecl, {
    TRY_TO(TraverseFunctionHelper(D));
    for (CXXConstructorDecl::init_iterator I = D->init_begin(),
                                           E = D->init_end();
         I != E; ++I) {
      // FIXME: figure out how to recurse on the initializers
    }
  })

DEF_TRAVERSE_DECL(CXXConversionDecl, {
    TRY_TO(TraverseFunctionHelper(D));
    // FIXME: is this redundant with GetResultType in VisitFunctionDecl?
    TRY_TO(TraverseType(D->getConversionType()));
  })

DEF_TRAVERSE_DECL(CXXDestructorDecl, {
    TRY_TO(TraverseFunctionHelper(D));
  })

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseVarHelper(VarDecl *D) {
  TRY_TO(TraverseDeclaratorHelper(D));
  // FIXME: This often double-counts -- for instance, for all local
  // vars, though not for global vars -- because the initializer is
  // also captured when the var-decl is in a DeclStmt.
  TRY_TO(TraverseStmt(D->getInit()));
  return true;
}

DEF_TRAVERSE_DECL(VarDecl, {
    TRY_TO(TraverseVarHelper(D));
  })

DEF_TRAVERSE_DECL(ImplicitParamDecl, {
    TRY_TO(TraverseVarHelper(D));
  })

DEF_TRAVERSE_DECL(NonTypeTemplateParmDecl, {
    // A non-type template parameter, e.g. "S" in template<int S> class Foo ...
    TRY_TO(TraverseStmt(D->getDefaultArgument()));
    TRY_TO(TraverseVarHelper(D));
  })

DEF_TRAVERSE_DECL(ParmVarDecl, {
    if (D->hasDefaultArg() &&
        D->hasUninstantiatedDefaultArg() &&
        !D->hasUnparsedDefaultArg())
      TRY_TO(TraverseStmt(D->getUninstantiatedDefaultArg()));

    if (D->hasDefaultArg() &&
        !D->hasUninstantiatedDefaultArg() &&
        !D->hasUnparsedDefaultArg())
      TRY_TO(TraverseStmt(D->getDefaultArg()));

    // FIXME: we should traverse the TypeSourceInfo, which is far
    // better that traversing getOriginalType(), since the
    // TypeSourceInfo describes the original type as written.
    TRY_TO(TraverseVarHelper(D));
  })

#undef DEF_TRAVERSE_DECL

// ----------------- Stmt traversal -----------------
//
// For stmts, we automate (in the DEF_TRAVERSE_STMT macro) iterating
// over the children defined in child_begin/child_end (every stmt
// defines these, though sometimes the range is empty).  Each
// individual Traverse* method only needs to worry about children
// other than those.  To see what child_begin()/end() does for a given
// class, see, e.g.,
// http://clang.llvm.org/doxygen/Stmt_8cpp_source.html

// This macro makes available a variable S, the passed-in stmt.
#define DEF_TRAVERSE_STMT(STMT, CODE)                                   \
template<typename Derived>                                              \
bool RecursiveASTVisitor<Derived>::Traverse##STMT (STMT *S) {           \
  TRY_TO(WalkUpFrom##STMT(S));                                          \
  { CODE; }                                                             \
  for (Stmt::child_iterator C = S->child_begin(), CEnd = S->child_end(); \
       C != CEnd; ++C) {                                                \
    TRY_TO(TraverseStmt(*C));                                           \
  }                                                                     \
  return true;                                                          \
}

DEF_TRAVERSE_STMT(AsmStmt, {
    TRY_TO(TraverseStmt(S->getAsmString()));
    for (unsigned I = 0, E = S->getNumInputs(); I < E; ++I) {
      TRY_TO(TraverseStmt(S->getInputConstraintLiteral(I)));
    }
    for (unsigned I = 0, E = S->getNumOutputs(); I < E; ++I) {
      TRY_TO(TraverseStmt(S->getOutputConstraintLiteral(I)));
    }
    for (unsigned I = 0, E = S->getNumClobbers(); I < E; ++I) {
      TRY_TO(TraverseStmt(S->getClobber(I)));
    }
    // child_begin()/end() iterates over inputExpr and outputExpr.
  })

DEF_TRAVERSE_STMT(CXXCatchStmt, {
    // We don't traverse S->getCaughtType(), as we are already
    // traversing the exception object, which has this type.
    // child_begin()/end() iterates over the handler block.
  })

DEF_TRAVERSE_STMT(ForStmt, {
    TRY_TO(TraverseDecl(S->getConditionVariable()));
    // child_begin()/end() iterates over init, cond, inc, and body stmts.
  })

DEF_TRAVERSE_STMT(IfStmt, {
    TRY_TO(TraverseDecl(S->getConditionVariable()));
    // child_begin()/end() iterates over cond, then, and else stmts.
  })

DEF_TRAVERSE_STMT(WhileStmt, {
    TRY_TO(TraverseDecl(S->getConditionVariable()));
    // child_begin()/end() iterates over cond, then, and else stmts.
  })

// These non-expr stmts (most of them), do not need any action except
// iterating over the children.
DEF_TRAVERSE_STMT(BreakStmt, { })
DEF_TRAVERSE_STMT(CompoundStmt, { })
DEF_TRAVERSE_STMT(ContinueStmt, { })
DEF_TRAVERSE_STMT(CXXTryStmt, { })
DEF_TRAVERSE_STMT(DeclStmt, { })
DEF_TRAVERSE_STMT(DoStmt, { })
DEF_TRAVERSE_STMT(GotoStmt, { })
DEF_TRAVERSE_STMT(IndirectGotoStmt, { })
DEF_TRAVERSE_STMT(LabelStmt, { })
DEF_TRAVERSE_STMT(NullStmt, { })
DEF_TRAVERSE_STMT(ObjCAtCatchStmt, { })
DEF_TRAVERSE_STMT(ObjCAtFinallyStmt, { })
DEF_TRAVERSE_STMT(ObjCAtSynchronizedStmt, { })
DEF_TRAVERSE_STMT(ObjCAtThrowStmt, { })
DEF_TRAVERSE_STMT(ObjCAtTryStmt, { })
DEF_TRAVERSE_STMT(ObjCForCollectionStmt, { })
DEF_TRAVERSE_STMT(ReturnStmt, { })
DEF_TRAVERSE_STMT(SwitchStmt, { })
DEF_TRAVERSE_STMT(SwitchCase, { })
DEF_TRAVERSE_STMT(CaseStmt, { })
DEF_TRAVERSE_STMT(DefaultStmt, { })

DEF_TRAVERSE_STMT(CXXDependentScopeMemberExpr, {
    if (S->hasExplicitTemplateArgs()) {
      TRY_TO(TraverseTemplateArgumentLocsHelper(
          S->getTemplateArgs(), S->getNumTemplateArgs()));
    }
    TRY_TO(TraverseNestedNameSpecifier(S->getQualifier()));
  })

DEF_TRAVERSE_STMT(DeclRefExpr, {
    TRY_TO(TraverseTemplateArgumentLocsHelper(
        S->getTemplateArgs(), S->getNumTemplateArgs()));
    // FIXME: Should we be recursing on the qualifier?
    TRY_TO(TraverseNestedNameSpecifier(S->getQualifier()));
  })

DEF_TRAVERSE_STMT(DependentScopeDeclRefExpr, {
    // FIXME: Should we be recursing on these two things?
    if (S->hasExplicitTemplateArgs()) {
      TRY_TO(TraverseTemplateArgumentLocsHelper(
          S->getExplicitTemplateArgs().getTemplateArgs(),
          S->getNumTemplateArgs()));
    }
    TRY_TO(TraverseNestedNameSpecifier(S->getQualifier()));
  })

DEF_TRAVERSE_STMT(MemberExpr, {
    TRY_TO(TraverseTemplateArgumentLocsHelper(
        S->getTemplateArgs(), S->getNumTemplateArgs()));
    // FIXME: Should we be recursing on the qualifier?
    TRY_TO(TraverseNestedNameSpecifier(S->getQualifier()));
  })

DEF_TRAVERSE_STMT(ImplicitCastExpr, {
    // We don't traverse the cast type, as it's not written in the
    // source code.
  })

DEF_TRAVERSE_STMT(CStyleCastExpr, {
    TRY_TO(TraverseType(S->getTypeAsWritten()));
  })

DEF_TRAVERSE_STMT(CXXFunctionalCastExpr, {
    TRY_TO(TraverseType(S->getTypeAsWritten()));
  })

DEF_TRAVERSE_STMT(CXXConstCastExpr, {
    TRY_TO(TraverseType(S->getTypeAsWritten()));
  })

DEF_TRAVERSE_STMT(CXXDynamicCastExpr, {
    TRY_TO(TraverseType(S->getTypeAsWritten()));
  })

DEF_TRAVERSE_STMT(CXXReinterpretCastExpr, {
    TRY_TO(TraverseType(S->getTypeAsWritten()));
  })

DEF_TRAVERSE_STMT(CXXStaticCastExpr, {
    TRY_TO(TraverseType(S->getTypeAsWritten()));
  })

// These exprs (most of them), do not need any action except iterating
// over the children.
DEF_TRAVERSE_STMT(AddrLabelExpr, { })
DEF_TRAVERSE_STMT(ArraySubscriptExpr, { })
DEF_TRAVERSE_STMT(BlockDeclRefExpr, { })
DEF_TRAVERSE_STMT(BlockExpr, { })
DEF_TRAVERSE_STMT(ChooseExpr, { })
DEF_TRAVERSE_STMT(CompoundLiteralExpr, { })
DEF_TRAVERSE_STMT(CXXBindReferenceExpr, { })
DEF_TRAVERSE_STMT(CXXBindTemporaryExpr, { })
DEF_TRAVERSE_STMT(CXXBoolLiteralExpr, { })
DEF_TRAVERSE_STMT(CXXDefaultArgExpr, { })
DEF_TRAVERSE_STMT(CXXDeleteExpr, { })
DEF_TRAVERSE_STMT(CXXExprWithTemporaries, { })
DEF_TRAVERSE_STMT(CXXNewExpr, { })
DEF_TRAVERSE_STMT(CXXNullPtrLiteralExpr, { })
DEF_TRAVERSE_STMT(CXXPseudoDestructorExpr, { })
DEF_TRAVERSE_STMT(CXXThisExpr, { })
DEF_TRAVERSE_STMT(CXXThrowExpr, { })
DEF_TRAVERSE_STMT(CXXTypeidExpr, { })
DEF_TRAVERSE_STMT(CXXUnresolvedConstructExpr, { })
DEF_TRAVERSE_STMT(CXXZeroInitValueExpr, { })
DEF_TRAVERSE_STMT(DesignatedInitExpr, { })
DEF_TRAVERSE_STMT(ExtVectorElementExpr, { })
DEF_TRAVERSE_STMT(GNUNullExpr, { })
DEF_TRAVERSE_STMT(ImplicitValueInitExpr, { })
DEF_TRAVERSE_STMT(InitListExpr, { })
DEF_TRAVERSE_STMT(ObjCEncodeExpr, { })
DEF_TRAVERSE_STMT(ObjCImplicitSetterGetterRefExpr, { })
DEF_TRAVERSE_STMT(ObjCIsaExpr, { })
DEF_TRAVERSE_STMT(ObjCIvarRefExpr, { })
DEF_TRAVERSE_STMT(ObjCMessageExpr, { })
DEF_TRAVERSE_STMT(ObjCPropertyRefExpr, { })
DEF_TRAVERSE_STMT(ObjCProtocolExpr, { })
DEF_TRAVERSE_STMT(ObjCSelectorExpr, { })
DEF_TRAVERSE_STMT(ObjCSuperExpr, { })
DEF_TRAVERSE_STMT(OffsetOfExpr, { })
DEF_TRAVERSE_STMT(ParenExpr, { })
DEF_TRAVERSE_STMT(ParenListExpr, { })
DEF_TRAVERSE_STMT(PredefinedExpr, { })
DEF_TRAVERSE_STMT(ShuffleVectorExpr, { })
DEF_TRAVERSE_STMT(SizeOfAlignOfExpr, { })
DEF_TRAVERSE_STMT(StmtExpr, { })
DEF_TRAVERSE_STMT(TypesCompatibleExpr, { })
DEF_TRAVERSE_STMT(UnaryTypeTraitExpr, { })
DEF_TRAVERSE_STMT(UnresolvedLookupExpr, { })
DEF_TRAVERSE_STMT(UnresolvedMemberExpr, { })
DEF_TRAVERSE_STMT(VAArgExpr, { })
DEF_TRAVERSE_STMT(CXXConstructExpr, { })
DEF_TRAVERSE_STMT(CXXTemporaryObjectExpr, { })
DEF_TRAVERSE_STMT(CallExpr, { })
DEF_TRAVERSE_STMT(CXXMemberCallExpr, { })
DEF_TRAVERSE_STMT(CXXOperatorCallExpr, { })

// These operators (all of them) do not need any action except
// iterating over the children.
DEF_TRAVERSE_STMT(ConditionalOperator, { })
DEF_TRAVERSE_STMT(UnaryOperator, { })
DEF_TRAVERSE_STMT(BinaryOperator, { })
DEF_TRAVERSE_STMT(CompoundAssignOperator, { })

// These literals (all of them) do not need any action.
DEF_TRAVERSE_STMT(IntegerLiteral, { })
DEF_TRAVERSE_STMT(CharacterLiteral, { })
DEF_TRAVERSE_STMT(FloatingLiteral, { })
DEF_TRAVERSE_STMT(ImaginaryLiteral, { })
DEF_TRAVERSE_STMT(StringLiteral, { })
DEF_TRAVERSE_STMT(ObjCStringLiteral, { })

// FIXME: look at the following tricky-seeming exprs to see if we
// need to recurse on anything.  These are ones that have methods
// returning decls or qualtypes or nestednamespecifier -- though I'm
// not sure if they own them -- or just seemed very complicated, or
// had lots of sub-types to explore.
//
// VisitOverloadExpr and its children: recurse on template args? etc?

// FIXME: go through all the stmts and exprs again, and see which of them
// create new types, and recurse on the types (TypeLocs?) of those.
// Candidates:
//    http://clang.llvm.org/doxygen/classclang_1_1CXXTypeidExpr.html
//    http://clang.llvm.org/doxygen/classclang_1_1SizeOfAlignOfExpr.html
//    http://clang.llvm.org/doxygen/classclang_1_1TypesCompatibleExpr.html
//    http://clang.llvm.org/doxygen/classclang_1_1CXXUnresolvedConstructExpr.html
//    Every class that has getQualifier.

#undef DEF_TRAVERSE_STMT

#undef TRY_TO

} // end namespace clang

#endif // LLVM_CLANG_AST_RECURSIVEASTVISITOR_H
