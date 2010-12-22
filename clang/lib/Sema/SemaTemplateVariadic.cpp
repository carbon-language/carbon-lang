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
#include "clang/Sema/Template.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/TypeLoc.h"

using namespace clang;

//----------------------------------------------------------------------------
// Visitor that collects unexpanded parameter packs
//----------------------------------------------------------------------------

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

    bool shouldWalkTypesOfTypeLocs() const { return false; }
    
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

void Sema::collectUnexpandedParameterPacks(TemplateArgument Arg,
                   llvm::SmallVectorImpl<UnexpandedParameterPack> &Unexpanded) {
  CollectUnexpandedParameterPacksVisitor(Unexpanded)
    .TraverseTemplateArgument(Arg);
}

void Sema::collectUnexpandedParameterPacks(TemplateArgumentLoc Arg,
                   llvm::SmallVectorImpl<UnexpandedParameterPack> &Unexpanded) {
  CollectUnexpandedParameterPacksVisitor(Unexpanded)
    .TraverseTemplateArgumentLoc(Arg);
}

void Sema::collectUnexpandedParameterPacks(QualType T,
                   llvm::SmallVectorImpl<UnexpandedParameterPack> &Unexpanded) {
  CollectUnexpandedParameterPacksVisitor(Unexpanded).TraverseType(T);  
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

  TypeSourceInfo *TSResult = CheckPackExpansion(TSInfo, EllipsisLoc);
  if (!TSResult)
    return true;
  
  return CreateParsedType(TSResult->getType(), TSResult);
}

TypeSourceInfo *Sema::CheckPackExpansion(TypeSourceInfo *Pattern,
                                         SourceLocation EllipsisLoc) {
  // C++0x [temp.variadic]p5:
  //   The pattern of a pack expansion shall name one or more
  //   parameter packs that are not expanded by a nested pack
  //   expansion.
  if (!Pattern->getType()->containsUnexpandedParameterPack()) {
    Diag(EllipsisLoc, diag::err_pack_expansion_without_parameter_packs)
      << Pattern->getTypeLoc().getSourceRange();
    return 0;
  }
  
  // Create the pack expansion type and source-location information.
  QualType Result = Context.getPackExpansionType(Pattern->getType());
  TypeSourceInfo *TSResult = Context.CreateTypeSourceInfo(Result);
  PackExpansionTypeLoc TL = cast<PackExpansionTypeLoc>(TSResult->getTypeLoc());
  TL.setEllipsisLoc(EllipsisLoc);
  
  // Copy over the source-location information from the type.
  memcpy(TL.getNextTypeLoc().getOpaqueData(),
         Pattern->getTypeLoc().getOpaqueData(),
         Pattern->getTypeLoc().getFullDataSize());
  return TSResult;
}


bool Sema::CheckParameterPacksForExpansion(SourceLocation EllipsisLoc,
                                           SourceRange PatternRange,
                                     const UnexpandedParameterPack *Unexpanded,
                                           unsigned NumUnexpanded,
                             const MultiLevelTemplateArgumentList &TemplateArgs,
                                           bool &ShouldExpand,
                                           unsigned &NumExpansions) {                                        
  ShouldExpand = true;
  std::pair<IdentifierInfo *, SourceLocation> FirstPack;
  bool HaveFirstPack = false;
  
  for (unsigned I = 0; I != NumUnexpanded; ++I) {
    // Compute the depth and index for this parameter pack.
    unsigned Depth;
    unsigned Index;
    IdentifierInfo *Name;
    
    if (const TemplateTypeParmType *TTP
        = Unexpanded[I].first.dyn_cast<const TemplateTypeParmType *>()) {
      Depth = TTP->getDepth();
      Index = TTP->getIndex();
      Name = TTP->getName();
    } else {
      NamedDecl *ND = Unexpanded[I].first.get<NamedDecl *>();
      if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(ND)) {
        Depth = TTP->getDepth();
        Index = TTP->getIndex();
      } else if (NonTypeTemplateParmDecl *NTTP
                 = dyn_cast<NonTypeTemplateParmDecl>(ND)) {        
        Depth = NTTP->getDepth();
        Index = NTTP->getIndex();
      } else {
        TemplateTemplateParmDecl *TTP = cast<TemplateTemplateParmDecl>(ND);
        Depth = TTP->getDepth();
        Index = TTP->getIndex();
      }
      // FIXME: Variadic templates function parameter packs?
      Name = ND->getIdentifier();
    }
    
    // If we don't have a template argument at this depth/index, then we 
    // cannot expand the pack expansion. Make a note of this, but we still 
    // want to check that any parameter packs we *do* have arguments for.
    if (!TemplateArgs.hasTemplateArgument(Depth, Index)) {
      ShouldExpand = false;
      continue;
    }
    
    // Determine the size of the argument pack.
    unsigned NewPackSize = TemplateArgs(Depth, Index).pack_size();
    if (!HaveFirstPack) {
      // The is the first pack we've seen for which we have an argument. 
      // Record it.
      NumExpansions = NewPackSize;
      FirstPack.first = Name;
      FirstPack.second = Unexpanded[I].second;
      HaveFirstPack = true;
      continue;
    }
    
    if (NewPackSize != NumExpansions) {
      // C++0x [temp.variadic]p5:
      //   All of the parameter packs expanded by a pack expansion shall have 
      //   the same number of arguments specified.
      Diag(EllipsisLoc, diag::err_pack_expansion_length_conflict)
        << FirstPack.first << Name << NumExpansions << NewPackSize
        << SourceRange(FirstPack.second) << SourceRange(Unexpanded[I].second);
      return true;
    }
  }
  
  return false;
}
