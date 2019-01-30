//===--- ASTDumper.cpp - Dumping implementation for ASTs ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AST dump methods, which dump out the
// AST in a form that exposes type details and other fields.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDumperUtils.h"
#include "clang/AST/Attr.h"
#include "clang/AST/AttrVisitor.h"
#include "clang/AST/CommentVisitor.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclLookups.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/LocInfoType.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/TemplateArgumentVisitor.h"
#include "clang/AST/TextNodeDumper.h"
#include "clang/AST/TypeVisitor.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;
using namespace clang::comments;

//===----------------------------------------------------------------------===//
// ASTDumper Visitor
//===----------------------------------------------------------------------===//

namespace  {

  class ASTDumper
      : public ConstDeclVisitor<ASTDumper>,
        public ConstStmtVisitor<ASTDumper>,
        public ConstCommentVisitor<ASTDumper, void, const FullComment *>,
        public TypeVisitor<ASTDumper>,
        public ConstAttrVisitor<ASTDumper>,
        public ConstTemplateArgumentVisitor<ASTDumper> {

    TextNodeDumper NodeDumper;

    raw_ostream &OS;

    /// Indicates whether we should trigger deserialization of nodes that had
    /// not already been loaded.
    bool Deserialize = false;

    const bool ShowColors;

    /// Dump a child of the current node.
    template<typename Fn> void dumpChild(Fn DoDumpChild) {
      NodeDumper.AddChild(DoDumpChild);
    }
    template <typename Fn> void dumpChild(StringRef Label, Fn DoDumpChild) {
      NodeDumper.AddChild(Label, DoDumpChild);
    }

  public:
    ASTDumper(raw_ostream &OS, const CommandTraits *Traits,
              const SourceManager *SM)
        : ASTDumper(OS, Traits, SM,
                    SM && SM->getDiagnostics().getShowColors()) {}

    ASTDumper(raw_ostream &OS, const CommandTraits *Traits,
              const SourceManager *SM, bool ShowColors)
        : ASTDumper(OS, Traits, SM, ShowColors, LangOptions()) {}
    ASTDumper(raw_ostream &OS, const CommandTraits *Traits,
              const SourceManager *SM, bool ShowColors,
              const PrintingPolicy &PrintPolicy)
        : NodeDumper(OS, ShowColors, SM, PrintPolicy, Traits), OS(OS),
          ShowColors(ShowColors) {}

    void setDeserialize(bool D) { Deserialize = D; }

    void dumpDecl(const Decl *D);
    void dumpStmt(const Stmt *S, StringRef Label = {});

    // Utilities
    void dumpTypeAsChild(QualType T);
    void dumpTypeAsChild(const Type *T);
    void dumpDeclContext(const DeclContext *DC);
    void dumpLookups(const DeclContext *DC, bool DumpDecls);
    void dumpAttr(const Attr *A);

    // C++ Utilities
    void dumpCXXCtorInitializer(const CXXCtorInitializer *Init);
    void dumpTemplateParameters(const TemplateParameterList *TPL);
    void dumpTemplateArgumentListInfo(const TemplateArgumentListInfo &TALI);
    void dumpTemplateArgumentLoc(const TemplateArgumentLoc &A,
                                 const Decl *From = nullptr,
                                 const char *Label = nullptr);
    void dumpTemplateArgumentList(const TemplateArgumentList &TAL);
    void dumpTemplateArgument(const TemplateArgument &A,
                              SourceRange R = SourceRange(),
                              const Decl *From = nullptr,
                              const char *Label = nullptr);
    template <typename SpecializationDecl>
    void dumpTemplateDeclSpecialization(const SpecializationDecl *D,
                                        bool DumpExplicitInst,
                                        bool DumpRefOnly);
    template <typename TemplateDecl>
    void dumpTemplateDecl(const TemplateDecl *D, bool DumpExplicitInst);

    // Objective-C utilities.
    void dumpObjCTypeParamList(const ObjCTypeParamList *typeParams);

    // Types
    void VisitComplexType(const ComplexType *T) {
      dumpTypeAsChild(T->getElementType());
    }
    void VisitLocInfoType(const LocInfoType *T) {
      dumpTypeAsChild(T->getTypeSourceInfo()->getType());
    }
    void VisitPointerType(const PointerType *T) {
      dumpTypeAsChild(T->getPointeeType());
    }
    void VisitBlockPointerType(const BlockPointerType *T) {
      dumpTypeAsChild(T->getPointeeType());
    }
    void VisitReferenceType(const ReferenceType *T) {
      dumpTypeAsChild(T->getPointeeType());
    }
    void VisitMemberPointerType(const MemberPointerType *T) {
      dumpTypeAsChild(T->getClass());
      dumpTypeAsChild(T->getPointeeType());
    }
    void VisitArrayType(const ArrayType *T) {
      dumpTypeAsChild(T->getElementType());
    }
    void VisitVariableArrayType(const VariableArrayType *T) {
      VisitArrayType(T);
      dumpStmt(T->getSizeExpr());
    }
    void VisitDependentSizedArrayType(const DependentSizedArrayType *T) {
      dumpTypeAsChild(T->getElementType());
      dumpStmt(T->getSizeExpr());
    }
    void VisitDependentSizedExtVectorType(
        const DependentSizedExtVectorType *T) {
      dumpTypeAsChild(T->getElementType());
      dumpStmt(T->getSizeExpr());
    }
    void VisitVectorType(const VectorType *T) {
      dumpTypeAsChild(T->getElementType());
    }
    void VisitFunctionType(const FunctionType *T) {
      dumpTypeAsChild(T->getReturnType());
    }
    void VisitFunctionProtoType(const FunctionProtoType *T) {
      VisitFunctionType(T);
      for (const QualType &PT : T->getParamTypes())
        dumpTypeAsChild(PT);
    }
    void VisitTypeOfExprType(const TypeOfExprType *T) {
      dumpStmt(T->getUnderlyingExpr());
    }
    void VisitDecltypeType(const DecltypeType *T) {
      dumpStmt(T->getUnderlyingExpr());
    }
    void VisitUnaryTransformType(const UnaryTransformType *T) {
      dumpTypeAsChild(T->getBaseType());
    }
    void VisitAttributedType(const AttributedType *T) {
      // FIXME: AttrKind
      dumpTypeAsChild(T->getModifiedType());
    }
    void VisitSubstTemplateTypeParmType(const SubstTemplateTypeParmType *T) {
      dumpTypeAsChild(T->getReplacedParameter());
    }
    void VisitSubstTemplateTypeParmPackType(
        const SubstTemplateTypeParmPackType *T) {
      dumpTypeAsChild(T->getReplacedParameter());
      dumpTemplateArgument(T->getArgumentPack());
    }
    void VisitTemplateSpecializationType(const TemplateSpecializationType *T) {
      for (const auto &Arg : *T)
        dumpTemplateArgument(Arg);
      if (T->isTypeAlias())
        dumpTypeAsChild(T->getAliasedType());
    }
    void VisitObjCObjectPointerType(const ObjCObjectPointerType *T) {
      dumpTypeAsChild(T->getPointeeType());
    }
    void VisitAtomicType(const AtomicType *T) {
      dumpTypeAsChild(T->getValueType());
    }
    void VisitPipeType(const PipeType *T) {
      dumpTypeAsChild(T->getElementType());
    }
    void VisitAdjustedType(const AdjustedType *T) {
      dumpTypeAsChild(T->getOriginalType());
    }
    void VisitPackExpansionType(const PackExpansionType *T) {
      if (!T->isSugared())
        dumpTypeAsChild(T->getPattern());
    }
    // FIXME: ElaboratedType, DependentNameType,
    // DependentTemplateSpecializationType, ObjCObjectType

    // Decls
    void VisitTypedefDecl(const TypedefDecl *D);
    void VisitEnumConstantDecl(const EnumConstantDecl *D);
    void VisitFunctionDecl(const FunctionDecl *D);
    void VisitFieldDecl(const FieldDecl *D);
    void VisitVarDecl(const VarDecl *D);
    void VisitDecompositionDecl(const DecompositionDecl *D);
    void VisitBindingDecl(const BindingDecl *D);
    void VisitFileScopeAsmDecl(const FileScopeAsmDecl *D);
    void VisitCapturedDecl(const CapturedDecl *D);

    // OpenMP decls
    void VisitOMPThreadPrivateDecl(const OMPThreadPrivateDecl *D);
    void VisitOMPDeclareReductionDecl(const OMPDeclareReductionDecl *D);
    void VisitOMPCapturedExprDecl(const OMPCapturedExprDecl *D);

    // C++ Decls
    void VisitTypeAliasDecl(const TypeAliasDecl *D);
    void VisitTypeAliasTemplateDecl(const TypeAliasTemplateDecl *D);
    void VisitStaticAssertDecl(const StaticAssertDecl *D);
    void VisitFunctionTemplateDecl(const FunctionTemplateDecl *D);
    void VisitClassTemplateDecl(const ClassTemplateDecl *D);
    void VisitClassTemplateSpecializationDecl(
        const ClassTemplateSpecializationDecl *D);
    void VisitClassTemplatePartialSpecializationDecl(
        const ClassTemplatePartialSpecializationDecl *D);
    void VisitClassScopeFunctionSpecializationDecl(
        const ClassScopeFunctionSpecializationDecl *D);
    void VisitBuiltinTemplateDecl(const BuiltinTemplateDecl *D);
    void VisitVarTemplateDecl(const VarTemplateDecl *D);
    void VisitVarTemplateSpecializationDecl(
        const VarTemplateSpecializationDecl *D);
    void VisitVarTemplatePartialSpecializationDecl(
        const VarTemplatePartialSpecializationDecl *D);
    void VisitTemplateTypeParmDecl(const TemplateTypeParmDecl *D);
    void VisitNonTypeTemplateParmDecl(const NonTypeTemplateParmDecl *D);
    void VisitTemplateTemplateParmDecl(const TemplateTemplateParmDecl *D);
    void VisitUsingShadowDecl(const UsingShadowDecl *D);
    void VisitFriendDecl(const FriendDecl *D);

    // ObjC Decls
    void VisitObjCMethodDecl(const ObjCMethodDecl *D);
    void VisitObjCCategoryDecl(const ObjCCategoryDecl *D);
    void VisitObjCInterfaceDecl(const ObjCInterfaceDecl *D);
    void VisitObjCImplementationDecl(const ObjCImplementationDecl *D);
    void Visit(const BlockDecl::Capture &C);
    void VisitBlockDecl(const BlockDecl *D);

    // Stmts.
    void VisitDeclStmt(const DeclStmt *Node);
    void VisitAttributedStmt(const AttributedStmt *Node);
    void VisitCXXCatchStmt(const CXXCatchStmt *Node);
    void VisitCapturedStmt(const CapturedStmt *Node);

    // OpenMP
    void Visit(const OMPClause *C);
    void VisitOMPExecutableDirective(const OMPExecutableDirective *Node);

    // Exprs
    void VisitInitListExpr(const InitListExpr *ILE);
    void VisitBlockExpr(const BlockExpr *Node);
    void VisitOpaqueValueExpr(const OpaqueValueExpr *Node);
    void Visit(const GenericSelectionExpr::ConstAssociation &A);
    void VisitGenericSelectionExpr(const GenericSelectionExpr *E);

    // C++
    void VisitLambdaExpr(const LambdaExpr *Node) {
      dumpDecl(Node->getLambdaClass());
    }
    void VisitSizeOfPackExpr(const SizeOfPackExpr *Node);

    // ObjC
    void VisitObjCAtCatchStmt(const ObjCAtCatchStmt *Node);

    // Comments.
    void dumpComment(const Comment *C, const FullComment *FC);

    void VisitExpressionTemplateArgument(const TemplateArgument &TA) {
      dumpStmt(TA.getAsExpr());
    }
    void VisitPackTemplateArgument(const TemplateArgument &TA) {
      for (const auto &TArg : TA.pack_elements())
        dumpTemplateArgument(TArg);
    }

// Implements Visit methods for Attrs.
#include "clang/AST/AttrNodeTraverse.inc"
  };
}

//===----------------------------------------------------------------------===//
//  Utilities
//===----------------------------------------------------------------------===//

void ASTDumper::dumpTypeAsChild(QualType T) {
  SplitQualType SQT = T.split();
  if (!SQT.Quals.hasQualifiers())
    return dumpTypeAsChild(SQT.Ty);

  dumpChild([=] {
    NodeDumper.Visit(T);
    dumpTypeAsChild(T.split().Ty);
  });
}

void ASTDumper::dumpTypeAsChild(const Type *T) {
  dumpChild([=] {
    NodeDumper.Visit(T);
    if (!T)
      return;
    TypeVisitor<ASTDumper>::Visit(T);

    QualType SingleStepDesugar =
        T->getLocallyUnqualifiedSingleStepDesugaredType();
    if (SingleStepDesugar != QualType(T, 0))
      dumpTypeAsChild(SingleStepDesugar);
  });
}

void ASTDumper::dumpDeclContext(const DeclContext *DC) {
  if (!DC)
    return;

  for (const auto *D : (Deserialize ? DC->decls() : DC->noload_decls()))
    dumpDecl(D);
}

void ASTDumper::dumpLookups(const DeclContext *DC, bool DumpDecls) {
  dumpChild([=] {
    OS << "StoredDeclsMap ";
    NodeDumper.dumpBareDeclRef(cast<Decl>(DC));

    const DeclContext *Primary = DC->getPrimaryContext();
    if (Primary != DC) {
      OS << " primary";
      NodeDumper.dumpPointer(cast<Decl>(Primary));
    }

    bool HasUndeserializedLookups = Primary->hasExternalVisibleStorage();

    auto Range = Deserialize
                     ? Primary->lookups()
                     : Primary->noload_lookups(/*PreserveInternalState=*/true);
    for (auto I = Range.begin(), E = Range.end(); I != E; ++I) {
      DeclarationName Name = I.getLookupName();
      DeclContextLookupResult R = *I;

      dumpChild([=] {
        OS << "DeclarationName ";
        {
          ColorScope Color(OS, ShowColors, DeclNameColor);
          OS << '\'' << Name << '\'';
        }

        for (DeclContextLookupResult::iterator RI = R.begin(), RE = R.end();
             RI != RE; ++RI) {
          dumpChild([=] {
            NodeDumper.dumpBareDeclRef(*RI);

            if ((*RI)->isHidden())
              OS << " hidden";

            // If requested, dump the redecl chain for this lookup.
            if (DumpDecls) {
              // Dump earliest decl first.
              std::function<void(Decl *)> DumpWithPrev = [&](Decl *D) {
                if (Decl *Prev = D->getPreviousDecl())
                  DumpWithPrev(Prev);
                dumpDecl(D);
              };
              DumpWithPrev(*RI);
            }
          });
        }
      });
    }

    if (HasUndeserializedLookups) {
      dumpChild([=] {
        ColorScope Color(OS, ShowColors, UndeserializedColor);
        OS << "<undeserialized lookups>";
      });
    }
  });
}

void ASTDumper::dumpAttr(const Attr *A) {
  dumpChild([=] {
    NodeDumper.Visit(A);
    ConstAttrVisitor<ASTDumper>::Visit(A);
  });
}

//===----------------------------------------------------------------------===//
//  C++ Utilities
//===----------------------------------------------------------------------===//

void ASTDumper::dumpCXXCtorInitializer(const CXXCtorInitializer *Init) {
  dumpChild([=] {
    NodeDumper.Visit(Init);
    dumpStmt(Init->getInit());
  });
}

void ASTDumper::dumpTemplateParameters(const TemplateParameterList *TPL) {
  if (!TPL)
    return;

  for (const auto &TP : *TPL)
    dumpDecl(TP);
}

void ASTDumper::dumpTemplateArgumentListInfo(
    const TemplateArgumentListInfo &TALI) {
  for (const auto &TA : TALI.arguments())
    dumpTemplateArgumentLoc(TA);
}

void ASTDumper::dumpTemplateArgumentLoc(const TemplateArgumentLoc &A,
                                        const Decl *From, const char *Label) {
  dumpTemplateArgument(A.getArgument(), A.getSourceRange(), From, Label);
}

void ASTDumper::dumpTemplateArgumentList(const TemplateArgumentList &TAL) {
  for (unsigned i = 0, e = TAL.size(); i < e; ++i)
    dumpTemplateArgument(TAL[i]);
}

void ASTDumper::dumpTemplateArgument(const TemplateArgument &A, SourceRange R,
                                     const Decl *From, const char *Label) {
  dumpChild([=] {
    NodeDumper.Visit(A, R, From, Label);
    ConstTemplateArgumentVisitor<ASTDumper>::Visit(A);
  });
}

//===----------------------------------------------------------------------===//
//  Objective-C Utilities
//===----------------------------------------------------------------------===//
void ASTDumper::dumpObjCTypeParamList(const ObjCTypeParamList *typeParams) {
  if (!typeParams)
    return;

  for (const auto &typeParam : *typeParams) {
    dumpDecl(typeParam);
  }
}

//===----------------------------------------------------------------------===//
//  Decl dumping methods.
//===----------------------------------------------------------------------===//

void ASTDumper::dumpDecl(const Decl *D) {
  dumpChild([=] {
    NodeDumper.Visit(D);
    if (!D)
      return;

    ConstDeclVisitor<ASTDumper>::Visit(D);

    for (const auto &A : D->attrs())
      dumpAttr(A);

    if (const FullComment *Comment =
            D->getASTContext().getLocalCommentForDeclUncached(D))
      dumpComment(Comment, Comment);

    // Decls within functions are visited by the body.
    if (!isa<FunctionDecl>(*D) && !isa<ObjCMethodDecl>(*D)) {
      if (const auto *DC = dyn_cast<DeclContext>(D))
        dumpDeclContext(DC);
    }
  });
}

void ASTDumper::VisitTypedefDecl(const TypedefDecl *D) {
  dumpTypeAsChild(D->getUnderlyingType());
}

void ASTDumper::VisitEnumConstantDecl(const EnumConstantDecl *D) {
  if (const Expr *Init = D->getInitExpr())
    dumpStmt(Init);
}

void ASTDumper::VisitFunctionDecl(const FunctionDecl *D) {
  if (const auto *FTSI = D->getTemplateSpecializationInfo())
    dumpTemplateArgumentList(*FTSI->TemplateArguments);

  if (D->param_begin())
    for (const auto *Parameter : D->parameters())
      dumpDecl(Parameter);

  if (const auto *C = dyn_cast<CXXConstructorDecl>(D))
    for (const auto *I : C->inits())
      dumpCXXCtorInitializer(I);

  if (D->doesThisDeclarationHaveABody())
    dumpStmt(D->getBody());
}

void ASTDumper::VisitFieldDecl(const FieldDecl *D) {
  if (D->isBitField())
    dumpStmt(D->getBitWidth());
  if (Expr *Init = D->getInClassInitializer())
    dumpStmt(Init);
}

void ASTDumper::VisitVarDecl(const VarDecl *D) {
  if (D->hasInit())
    dumpStmt(D->getInit());
}

void ASTDumper::VisitDecompositionDecl(const DecompositionDecl *D) {
  VisitVarDecl(D);
  for (const auto *B : D->bindings())
    dumpDecl(B);
}

void ASTDumper::VisitBindingDecl(const BindingDecl *D) {
  if (const auto *E = D->getBinding())
    dumpStmt(E);
}

void ASTDumper::VisitFileScopeAsmDecl(const FileScopeAsmDecl *D) {
  dumpStmt(D->getAsmString());
}

void ASTDumper::VisitCapturedDecl(const CapturedDecl *D) {
  dumpStmt(D->getBody());
}

//===----------------------------------------------------------------------===//
// OpenMP Declarations
//===----------------------------------------------------------------------===//

void ASTDumper::VisitOMPThreadPrivateDecl(const OMPThreadPrivateDecl *D) {
  for (const auto *E : D->varlists())
    dumpStmt(E);
}

void ASTDumper::VisitOMPDeclareReductionDecl(const OMPDeclareReductionDecl *D) {
  dumpStmt(D->getCombiner());
  if (const auto *Initializer = D->getInitializer())
    dumpStmt(Initializer);
}

void ASTDumper::VisitOMPCapturedExprDecl(const OMPCapturedExprDecl *D) {
  dumpStmt(D->getInit());
}

//===----------------------------------------------------------------------===//
// C++ Declarations
//===----------------------------------------------------------------------===//

void ASTDumper::VisitTypeAliasDecl(const TypeAliasDecl *D) {
  dumpTypeAsChild(D->getUnderlyingType());
}

void ASTDumper::VisitTypeAliasTemplateDecl(const TypeAliasTemplateDecl *D) {
  dumpTemplateParameters(D->getTemplateParameters());
  dumpDecl(D->getTemplatedDecl());
}

void ASTDumper::VisitStaticAssertDecl(const StaticAssertDecl *D) {
  dumpStmt(D->getAssertExpr());
  dumpStmt(D->getMessage());
}

template <typename SpecializationDecl>
void ASTDumper::dumpTemplateDeclSpecialization(const SpecializationDecl *D,
                                               bool DumpExplicitInst,
                                               bool DumpRefOnly) {
  bool DumpedAny = false;
  for (const auto *RedeclWithBadType : D->redecls()) {
    // FIXME: The redecls() range sometimes has elements of a less-specific
    // type. (In particular, ClassTemplateSpecializationDecl::redecls() gives
    // us TagDecls, and should give CXXRecordDecls).
    auto *Redecl = dyn_cast<SpecializationDecl>(RedeclWithBadType);
    if (!Redecl) {
      // Found the injected-class-name for a class template. This will be dumped
      // as part of its surrounding class so we don't need to dump it here.
      assert(isa<CXXRecordDecl>(RedeclWithBadType) &&
             "expected an injected-class-name");
      continue;
    }

    switch (Redecl->getTemplateSpecializationKind()) {
    case TSK_ExplicitInstantiationDeclaration:
    case TSK_ExplicitInstantiationDefinition:
      if (!DumpExplicitInst)
        break;
      LLVM_FALLTHROUGH;
    case TSK_Undeclared:
    case TSK_ImplicitInstantiation:
      if (DumpRefOnly)
        NodeDumper.dumpDeclRef(Redecl);
      else
        dumpDecl(Redecl);
      DumpedAny = true;
      break;
    case TSK_ExplicitSpecialization:
      break;
    }
  }

  // Ensure we dump at least one decl for each specialization.
  if (!DumpedAny)
    NodeDumper.dumpDeclRef(D);
}

template <typename TemplateDecl>
void ASTDumper::dumpTemplateDecl(const TemplateDecl *D, bool DumpExplicitInst) {
  dumpTemplateParameters(D->getTemplateParameters());

  dumpDecl(D->getTemplatedDecl());

  for (const auto *Child : D->specializations())
    dumpTemplateDeclSpecialization(Child, DumpExplicitInst,
                                   !D->isCanonicalDecl());
}

void ASTDumper::VisitFunctionTemplateDecl(const FunctionTemplateDecl *D) {
  // FIXME: We don't add a declaration of a function template specialization
  // to its context when it's explicitly instantiated, so dump explicit
  // instantiations when we dump the template itself.
  dumpTemplateDecl(D, true);
}

void ASTDumper::VisitClassTemplateDecl(const ClassTemplateDecl *D) {
  dumpTemplateDecl(D, false);
}

void ASTDumper::VisitClassTemplateSpecializationDecl(
    const ClassTemplateSpecializationDecl *D) {
  dumpTemplateArgumentList(D->getTemplateArgs());
}

void ASTDumper::VisitClassTemplatePartialSpecializationDecl(
    const ClassTemplatePartialSpecializationDecl *D) {
  VisitClassTemplateSpecializationDecl(D);
  dumpTemplateParameters(D->getTemplateParameters());
}

void ASTDumper::VisitClassScopeFunctionSpecializationDecl(
    const ClassScopeFunctionSpecializationDecl *D) {
  dumpDecl(D->getSpecialization());
  if (D->hasExplicitTemplateArgs())
    dumpTemplateArgumentListInfo(D->templateArgs());
}

void ASTDumper::VisitVarTemplateDecl(const VarTemplateDecl *D) {
  dumpTemplateDecl(D, false);
}

void ASTDumper::VisitBuiltinTemplateDecl(const BuiltinTemplateDecl *D) {
  dumpTemplateParameters(D->getTemplateParameters());
}

void ASTDumper::VisitVarTemplateSpecializationDecl(
    const VarTemplateSpecializationDecl *D) {
  dumpTemplateArgumentList(D->getTemplateArgs());
  VisitVarDecl(D);
}

void ASTDumper::VisitVarTemplatePartialSpecializationDecl(
    const VarTemplatePartialSpecializationDecl *D) {
  dumpTemplateParameters(D->getTemplateParameters());
  VisitVarTemplateSpecializationDecl(D);
}

void ASTDumper::VisitTemplateTypeParmDecl(const TemplateTypeParmDecl *D) {
  if (D->hasDefaultArgument())
    dumpTemplateArgument(D->getDefaultArgument(), SourceRange(),
                         D->getDefaultArgStorage().getInheritedFrom(),
                         D->defaultArgumentWasInherited() ? "inherited from"
                                                          : "previous");
}

void ASTDumper::VisitNonTypeTemplateParmDecl(const NonTypeTemplateParmDecl *D) {
  if (D->hasDefaultArgument())
    dumpTemplateArgument(D->getDefaultArgument(), SourceRange(),
                         D->getDefaultArgStorage().getInheritedFrom(),
                         D->defaultArgumentWasInherited() ? "inherited from"
                                                          : "previous");
}

void ASTDumper::VisitTemplateTemplateParmDecl(
    const TemplateTemplateParmDecl *D) {
  dumpTemplateParameters(D->getTemplateParameters());
  if (D->hasDefaultArgument())
    dumpTemplateArgumentLoc(
        D->getDefaultArgument(), D->getDefaultArgStorage().getInheritedFrom(),
        D->defaultArgumentWasInherited() ? "inherited from" : "previous");
}

void ASTDumper::VisitUsingShadowDecl(const UsingShadowDecl *D) {
  if (auto *TD = dyn_cast<TypeDecl>(D->getUnderlyingDecl()))
    dumpTypeAsChild(TD->getTypeForDecl());
}

void ASTDumper::VisitFriendDecl(const FriendDecl *D) {
  if (!D->getFriendType())
    dumpDecl(D->getFriendDecl());
}

//===----------------------------------------------------------------------===//
// Obj-C Declarations
//===----------------------------------------------------------------------===//

void ASTDumper::VisitObjCMethodDecl(const ObjCMethodDecl *D) {
  if (D->isThisDeclarationADefinition())
    dumpDeclContext(D);
  else
    for (const ParmVarDecl *Parameter : D->parameters())
      dumpDecl(Parameter);

  if (D->hasBody())
    dumpStmt(D->getBody());
}

void ASTDumper::VisitObjCCategoryDecl(const ObjCCategoryDecl *D) {
  dumpObjCTypeParamList(D->getTypeParamList());
}

void ASTDumper::VisitObjCInterfaceDecl(const ObjCInterfaceDecl *D) {
  dumpObjCTypeParamList(D->getTypeParamListAsWritten());
}

void ASTDumper::VisitObjCImplementationDecl(const ObjCImplementationDecl *D) {
  for (const auto &I : D->inits())
    dumpCXXCtorInitializer(I);
}

void ASTDumper::Visit(const BlockDecl::Capture &C) {
  dumpChild([=] {
    NodeDumper.Visit(C);
    if (C.hasCopyExpr())
      dumpStmt(C.getCopyExpr());
  });
}

void ASTDumper::VisitBlockDecl(const BlockDecl *D) {
  for (const auto &I : D->parameters())
    dumpDecl(I);

  for (const auto &I : D->captures())
    Visit(I);
  dumpStmt(D->getBody());
}

//===----------------------------------------------------------------------===//
//  Stmt dumping methods.
//===----------------------------------------------------------------------===//

void ASTDumper::dumpStmt(const Stmt *S, StringRef Label) {
  dumpChild(Label, [=] {
    NodeDumper.Visit(S);

    if (!S) {
      return;
    }

    ConstStmtVisitor<ASTDumper>::Visit(S);

    // Some statements have custom mechanisms for dumping their children.
    if (isa<DeclStmt>(S) || isa<GenericSelectionExpr>(S)) {
      return;
    }

    for (const Stmt *SubStmt : S->children())
      dumpStmt(SubStmt);
  });
}

void ASTDumper::VisitDeclStmt(const DeclStmt *Node) {
  for (const auto &D : Node->decls())
    dumpDecl(D);
}

void ASTDumper::VisitAttributedStmt(const AttributedStmt *Node) {
  for (const auto *A : Node->getAttrs())
    dumpAttr(A);
}

void ASTDumper::VisitCXXCatchStmt(const CXXCatchStmt *Node) {
  dumpDecl(Node->getExceptionDecl());
}

void ASTDumper::VisitCapturedStmt(const CapturedStmt *Node) {
  dumpDecl(Node->getCapturedDecl());
}

//===----------------------------------------------------------------------===//
//  OpenMP dumping methods.
//===----------------------------------------------------------------------===//

void ASTDumper::Visit(const OMPClause *C) {
  dumpChild([=] {
    NodeDumper.Visit(C);
    for (const auto *S : C->children())
      dumpStmt(S);
  });
}

void ASTDumper::VisitOMPExecutableDirective(
    const OMPExecutableDirective *Node) {
  for (const auto *C : Node->clauses())
    Visit(C);
}

//===----------------------------------------------------------------------===//
//  Expr dumping methods.
//===----------------------------------------------------------------------===//


void ASTDumper::VisitInitListExpr(const InitListExpr *ILE) {
  if (auto *Filler = ILE->getArrayFiller()) {
    dumpStmt(Filler, "array_filler");
  }
}

void ASTDumper::VisitBlockExpr(const BlockExpr *Node) {
  dumpDecl(Node->getBlockDecl());
}

void ASTDumper::VisitOpaqueValueExpr(const OpaqueValueExpr *Node) {
  if (Expr *Source = Node->getSourceExpr())
    dumpStmt(Source);
}

void ASTDumper::Visit(const GenericSelectionExpr::ConstAssociation &A) {
  dumpChild([=] {
    NodeDumper.Visit(A);
    if (const TypeSourceInfo *TSI = A.getTypeSourceInfo())
      dumpTypeAsChild(TSI->getType());
    dumpStmt(A.getAssociationExpr());
  });
}

void ASTDumper::VisitGenericSelectionExpr(const GenericSelectionExpr *E) {
  dumpStmt(E->getControllingExpr());
  dumpTypeAsChild(E->getControllingExpr()->getType()); // FIXME: remove

  for (const auto &Assoc : E->associations()) {
    Visit(Assoc);
  }
}

//===----------------------------------------------------------------------===//
// C++ Expressions
//===----------------------------------------------------------------------===//

void ASTDumper::VisitSizeOfPackExpr(const SizeOfPackExpr *Node) {
  if (Node->isPartiallySubstituted())
    for (const auto &A : Node->getPartialArguments())
      dumpTemplateArgument(A);
}

//===----------------------------------------------------------------------===//
// Obj-C Expressions
//===----------------------------------------------------------------------===//

void ASTDumper::VisitObjCAtCatchStmt(const ObjCAtCatchStmt *Node) {
  if (const VarDecl *CatchParam = Node->getCatchParamDecl())
    dumpDecl(CatchParam);
}

//===----------------------------------------------------------------------===//
// Comments
//===----------------------------------------------------------------------===//

void ASTDumper::dumpComment(const Comment *C, const FullComment *FC) {
  dumpChild([=] {
    NodeDumper.Visit(C, FC);
    if (!C) {
      return;
    }
    ConstCommentVisitor<ASTDumper, void, const FullComment *>::visit(C, FC);
    for (Comment::child_iterator I = C->child_begin(), E = C->child_end();
         I != E; ++I)
      dumpComment(*I, FC);
  });
}

//===----------------------------------------------------------------------===//
// Type method implementations
//===----------------------------------------------------------------------===//

void QualType::dump(const char *msg) const {
  if (msg)
    llvm::errs() << msg << ": ";
  dump();
}

LLVM_DUMP_METHOD void QualType::dump() const { dump(llvm::errs()); }

LLVM_DUMP_METHOD void QualType::dump(llvm::raw_ostream &OS) const {
  ASTDumper Dumper(OS, nullptr, nullptr);
  Dumper.dumpTypeAsChild(*this);
}

LLVM_DUMP_METHOD void Type::dump() const { dump(llvm::errs()); }

LLVM_DUMP_METHOD void Type::dump(llvm::raw_ostream &OS) const {
  QualType(this, 0).dump(OS);
}

//===----------------------------------------------------------------------===//
// Decl method implementations
//===----------------------------------------------------------------------===//

LLVM_DUMP_METHOD void Decl::dump() const { dump(llvm::errs()); }

LLVM_DUMP_METHOD void Decl::dump(raw_ostream &OS, bool Deserialize) const {
  const ASTContext &Ctx = getASTContext();
  const SourceManager &SM = Ctx.getSourceManager();
  ASTDumper P(OS, &Ctx.getCommentCommandTraits(), &SM,
              SM.getDiagnostics().getShowColors(), Ctx.getPrintingPolicy());
  P.setDeserialize(Deserialize);
  P.dumpDecl(this);
}

LLVM_DUMP_METHOD void Decl::dumpColor() const {
  const ASTContext &Ctx = getASTContext();
  ASTDumper P(llvm::errs(), &Ctx.getCommentCommandTraits(),
              &Ctx.getSourceManager(), /*ShowColors*/ true,
              Ctx.getPrintingPolicy());
  P.dumpDecl(this);
}

LLVM_DUMP_METHOD void DeclContext::dumpLookups() const {
  dumpLookups(llvm::errs());
}

LLVM_DUMP_METHOD void DeclContext::dumpLookups(raw_ostream &OS,
                                               bool DumpDecls,
                                               bool Deserialize) const {
  const DeclContext *DC = this;
  while (!DC->isTranslationUnit())
    DC = DC->getParent();
  ASTContext &Ctx = cast<TranslationUnitDecl>(DC)->getASTContext();
  const SourceManager &SM = Ctx.getSourceManager();
  ASTDumper P(OS, &Ctx.getCommentCommandTraits(), &Ctx.getSourceManager(),
              SM.getDiagnostics().getShowColors(), Ctx.getPrintingPolicy());
  P.setDeserialize(Deserialize);
  P.dumpLookups(this, DumpDecls);
}

//===----------------------------------------------------------------------===//
// Stmt method implementations
//===----------------------------------------------------------------------===//

LLVM_DUMP_METHOD void Stmt::dump(SourceManager &SM) const {
  dump(llvm::errs(), SM);
}

LLVM_DUMP_METHOD void Stmt::dump(raw_ostream &OS, SourceManager &SM) const {
  ASTDumper P(OS, nullptr, &SM);
  P.dumpStmt(this);
}

LLVM_DUMP_METHOD void Stmt::dump(raw_ostream &OS) const {
  ASTDumper P(OS, nullptr, nullptr);
  P.dumpStmt(this);
}

LLVM_DUMP_METHOD void Stmt::dump() const {
  ASTDumper P(llvm::errs(), nullptr, nullptr);
  P.dumpStmt(this);
}

LLVM_DUMP_METHOD void Stmt::dumpColor() const {
  ASTDumper P(llvm::errs(), nullptr, nullptr, /*ShowColors*/true);
  P.dumpStmt(this);
}

//===----------------------------------------------------------------------===//
// Comment method implementations
//===----------------------------------------------------------------------===//

LLVM_DUMP_METHOD void Comment::dump() const {
  dump(llvm::errs(), nullptr, nullptr);
}

LLVM_DUMP_METHOD void Comment::dump(const ASTContext &Context) const {
  dump(llvm::errs(), &Context.getCommentCommandTraits(),
       &Context.getSourceManager());
}

void Comment::dump(raw_ostream &OS, const CommandTraits *Traits,
                   const SourceManager *SM) const {
  const FullComment *FC = dyn_cast<FullComment>(this);
  if (!FC)
    return;
  ASTDumper D(OS, Traits, SM);
  D.dumpComment(FC, FC);
}

LLVM_DUMP_METHOD void Comment::dumpColor() const {
  const FullComment *FC = dyn_cast<FullComment>(this);
  if (!FC)
    return;
  ASTDumper D(llvm::errs(), nullptr, nullptr, /*ShowColors*/true);
  D.dumpComment(FC, FC);
}
