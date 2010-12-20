//===------- SemaTemplateVariadic.cpp - C++ Variadic Templates ------------===/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===/
//
//  This file implements semantic analysis for C++0x variadic templates.
//===----------------------------------------------------------------------===/

#include "clang/Sema/Sema.h"
#include "clang/Sema/ParsedTemplate.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/TypeLoc.h"

using namespace clang;

//----------------------------------------------------------------------------
// Visitor that collects unexpanded parameter packs
//----------------------------------------------------------------------------

// FIXME: No way to easily map from TemplateTypeParmTypes to
// TemplateTypeParmDecls, so we have this horrible PointerUnion.
typedef std::pair<llvm::PointerUnion<const TemplateTypeParmType*, NamedDecl*>,
                  SourceLocation> UnexpandedParameterPack;

namespace {
  /// \brief A class that collects unexpanded parameter packs.
  class CollectUnexpandedParameterPacksVisitor :
    public RecursiveASTVisitor<CollectUnexpandedParameterPacksVisitor> 
  {
    typedef RecursiveASTVisitor<CollectUnexpandedParameterPacksVisitor>
      inherited;

    llvm::SmallVectorImpl<UnexpandedParameterPack> &Unexpanded;

  public:
    explicit CollectUnexpandedParameterPacksVisitor(
                  llvm::SmallVectorImpl<UnexpandedParameterPack> &Unexpanded)
      : Unexpanded(Unexpanded) { }

    //------------------------------------------------------------------------
    // Recording occurrences of (unexpanded) parameter packs.
    //------------------------------------------------------------------------

    /// \brief Record occurrences of template type parameter packs.
    bool VisitTemplateTypeParmTypeLoc(TemplateTypeParmTypeLoc TL) {
      if (TL.getTypePtr()->isParameterPack())
        Unexpanded.push_back(std::make_pair(TL.getTypePtr(), TL.getNameLoc()));
      return true;
    }

    /// \brief Record occurrences of template type parameter packs
    /// when we don't have proper source-location information for
    /// them.
    ///
    /// Ideally, this routine would never be used.
    bool VisitTemplateTypeParmType(TemplateTypeParmType *T) {
      if (T->isParameterPack())
        Unexpanded.push_back(std::make_pair(T, SourceLocation()));

      return true;
    }

    // FIXME: Record occurrences of non-type and template template
    // parameter packs.

    // FIXME: Once we have pack expansions in the AST, block their
    // traversal.

    //------------------------------------------------------------------------
    // Pruning the search for unexpanded parameter packs.
    //------------------------------------------------------------------------

    /// \brief Suppress traversal into statements and expressions that
    /// do not contain unexpanded parameter packs.
    bool TraverseStmt(Stmt *S) { 
      if (Expr *E = dyn_cast_or_null<Expr>(S))
        if (E->containsUnexpandedParameterPack())
          return inherited::TraverseStmt(E);

      return true; 
    }

    /// \brief Suppress traversal into types that do not contain
    /// unexpanded parameter packs.
    bool TraverseType(QualType T) {
      if (!T.isNull() && T->containsUnexpandedParameterPack())
        return inherited::TraverseType(T);

      return true;
    }

    /// \brief Suppress traversel into types with location information
    /// that do not contain unexpanded parameter packs.
    bool TraverseTypeLoc(TypeLoc TL) {
      if (!TL.getType().isNull() && TL.
          getType()->containsUnexpandedParameterPack())
        return inherited::TraverseTypeLoc(TL);

      return true;
    }

    /// \brief Suppress traversal of non-parameter declarations, since
    /// they cannot contain unexpanded parameter packs.
    bool TraverseDecl(Decl *D) { 
      if (D && isa<ParmVarDecl>(D))
        return inherited::TraverseDecl(D);

      return true; 
    }
  };
}

/// \brief Diagnose all of the unexpanded parameter packs in the given
/// vector.
static void 
DiagnoseUnexpandedParameterPacks(Sema &S, SourceLocation Loc,
                                 Sema::UnexpandedParameterPackContext UPPC,
             const llvm::SmallVectorImpl<UnexpandedParameterPack> &Unexpanded) {
  llvm::SmallVector<SourceLocation, 4> Locations;
  llvm::SmallVector<IdentifierInfo *, 4> Names;
  llvm::SmallPtrSet<IdentifierInfo *, 4> NamesKnown;

  for (unsigned I = 0, N = Unexpanded.size(); I != N; ++I) {
    IdentifierInfo *Name = 0;
    if (const TemplateTypeParmType *TTP
          = Unexpanded[I].first.dyn_cast<const TemplateTypeParmType *>())
      Name = TTP->getName();
    else
      Name = Unexpanded[I].first.get<NamedDecl *>()->getIdentifier();

    if (Name && NamesKnown.insert(Name))
      Names.push_back(Name);

    if (Unexpanded[I].second.isValid())
      Locations.push_back(Unexpanded[I].second);
  }

  DiagnosticBuilder DB
    = Names.size() == 0? S.Diag(Loc, diag::err_unexpanded_parameter_pack_0)
                           << (int)UPPC
    : Names.size() == 1? S.Diag(Loc, diag::err_unexpanded_parameter_pack_1)
                           << (int)UPPC << Names[0]
    : Names.size() == 2? S.Diag(Loc, diag::err_unexpanded_parameter_pack_2)
                           << (int)UPPC << Names[0] << Names[1]
    : S.Diag(Loc, diag::err_unexpanded_parameter_pack_3_or_more)
        << (int)UPPC << Names[0] << Names[1];

  for (unsigned I = 0, N = Locations.size(); I != N; ++I)
    DB << SourceRange(Locations[I]);
}

bool Sema::DiagnoseUnexpandedParameterPack(SourceLocation Loc, 
                                           TypeSourceInfo *T,
                                         UnexpandedParameterPackContext UPPC) {
  // C++0x [temp.variadic]p5:
  //   An appearance of a name of a parameter pack that is not expanded is 
  //   ill-formed.
  if (!T->getType()->containsUnexpandedParameterPack())
    return false;

  llvm::SmallVector<UnexpandedParameterPack, 2> Unexpanded;
  CollectUnexpandedParameterPacksVisitor(Unexpanded).TraverseTypeLoc(
                                                              T->getTypeLoc());
  assert(!Unexpanded.empty() && "Unable to find unexpanded parameter packs");
  DiagnoseUnexpandedParameterPacks(*this, Loc, UPPC, Unexpanded);
  return true;
}

bool Sema::DiagnoseUnexpandedParameterPack(Expr *E,
                                        UnexpandedParameterPackContext UPPC) {
  // C++0x [temp.variadic]p5:
  //   An appearance of a name of a parameter pack that is not expanded is 
  //   ill-formed.
  if (!E->containsUnexpandedParameterPack())
    return false;

  llvm::SmallVector<UnexpandedParameterPack, 2> Unexpanded;
  CollectUnexpandedParameterPacksVisitor(Unexpanded).TraverseStmt(E);
  assert(!Unexpanded.empty() && "Unable to find unexpanded parameter packs");
  DiagnoseUnexpandedParameterPacks(*this, E->getLocStart(), UPPC, Unexpanded);
  return true;
}

bool Sema::DiagnoseUnexpandedParameterPack(const CXXScopeSpec &SS,
                                        UnexpandedParameterPackContext UPPC) {
  // C++0x [temp.variadic]p5:
  //   An appearance of a name of a parameter pack that is not expanded is 
  //   ill-formed.
  if (!SS.getScopeRep() || 
      !SS.getScopeRep()->containsUnexpandedParameterPack())
    return false;

  llvm::SmallVector<UnexpandedParameterPack, 2> Unexpanded;
  CollectUnexpandedParameterPacksVisitor(Unexpanded)
    .TraverseNestedNameSpecifier(SS.getScopeRep());
  assert(!Unexpanded.empty() && "Unable to find unexpanded parameter packs");
  DiagnoseUnexpandedParameterPacks(*this, SS.getRange().getBegin(), 
                                   UPPC, Unexpanded);
  return true;
}

bool Sema::DiagnoseUnexpandedParameterPack(const DeclarationNameInfo &NameInfo,
                                         UnexpandedParameterPackContext UPPC) {
  // C++0x [temp.variadic]p5:
  //   An appearance of a name of a parameter pack that is not expanded is 
  //   ill-formed.
  switch (NameInfo.getName().getNameKind()) {
  case DeclarationName::Identifier:
  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
  case DeclarationName::CXXOperatorName:
  case DeclarationName::CXXLiteralOperatorName:
  case DeclarationName::CXXUsingDirective:
    return false;

  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
  case DeclarationName::CXXConversionFunctionName:
    // FIXME: We shouldn't need this null check!
    if (TypeSourceInfo *TSInfo = NameInfo.getNamedTypeInfo())
      return DiagnoseUnexpandedParameterPack(NameInfo.getLoc(), TSInfo, UPPC);

    if (!NameInfo.getName().getCXXNameType()->containsUnexpandedParameterPack())
      return false;

    break;
  }

  llvm::SmallVector<UnexpandedParameterPack, 2> Unexpanded;
  CollectUnexpandedParameterPacksVisitor(Unexpanded)
    .TraverseType(NameInfo.getName().getCXXNameType());
  assert(!Unexpanded.empty() && "Unable to find unexpanded parameter packs");
  DiagnoseUnexpandedParameterPacks(*this, NameInfo.getLoc(), UPPC, Unexpanded);
  return true;
}

bool Sema::DiagnoseUnexpandedParameterPack(SourceLocation Loc,
                                           TemplateName Template,
                                       UnexpandedParameterPackContext UPPC) {
  
  if (Template.isNull() || !Template.containsUnexpandedParameterPack())
    return false;

  llvm::SmallVector<UnexpandedParameterPack, 2> Unexpanded;
  CollectUnexpandedParameterPacksVisitor(Unexpanded)
    .TraverseTemplateName(Template);
  assert(!Unexpanded.empty() && "Unable to find unexpanded parameter packs");
  DiagnoseUnexpandedParameterPacks(*this, Loc, UPPC, Unexpanded);
  return true;
}

ParsedTemplateArgument 
Sema::ActOnPackExpansion(const ParsedTemplateArgument &Arg,
                         SourceLocation EllipsisLoc) {
  if (Arg.isInvalid())
    return Arg;

  switch (Arg.getKind()) {
  case ParsedTemplateArgument::Type: {
    TypeResult Result = ActOnPackExpansion(Arg.getAsType(), EllipsisLoc);
    if (Result.isInvalid())
      return ParsedTemplateArgument();

    return ParsedTemplateArgument(Arg.getKind(), Result.get().getAsOpaquePtr(), 
                                  Arg.getLocation());
  }

  case ParsedTemplateArgument::NonType:
    Diag(EllipsisLoc, diag::err_pack_expansion_unsupported)
      << 0;
    return ParsedTemplateArgument();

  case ParsedTemplateArgument::Template:
    Diag(EllipsisLoc, diag::err_pack_expansion_unsupported)
      << 1;
    return ParsedTemplateArgument();
  }
  llvm_unreachable("Unhandled template argument kind?");
  return ParsedTemplateArgument();
}

TypeResult Sema::ActOnPackExpansion(ParsedType Type, 
                                    SourceLocation EllipsisLoc) {
  TypeSourceInfo *TSInfo;
  GetTypeFromParser(Type, &TSInfo);
  if (!TSInfo)
    return true;

  // C++0x [temp.variadic]p5:
  //   The pattern of a pack expansion shall name one or more
  //   parameter packs that are not expanded by a nested pack
  //   expansion.
  if (!TSInfo->getType()->containsUnexpandedParameterPack()) {
    Diag(EllipsisLoc, diag::err_pack_expansion_without_parameter_packs)
      << TSInfo->getTypeLoc().getSourceRange();
    return true;
  }

  // Create the pack expansion type and source-location information.
  QualType Result = Context.getPackExpansionType(TSInfo->getType());
  TypeSourceInfo *TSResult = Context.CreateTypeSourceInfo(Result);
  PackExpansionTypeLoc TL = cast<PackExpansionTypeLoc>(TSResult->getTypeLoc());
  TL.setEllipsisLoc(EllipsisLoc);

  // Copy over the source-location information from the type.
  memcpy(TL.getNextTypeLoc().getOpaqueData(),
         TSInfo->getTypeLoc().getOpaqueData(),
         TSInfo->getTypeLoc().getFullDataSize());
  return CreateParsedType(Result, TSResult);
}
