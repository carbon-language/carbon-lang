// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TODO: Remove this once https://github.com/llvm/llvm-project/pull/204458
// is merged.

#include "third_party/llvm/multiplex_external_sema_source.h"

#include "clang/Sema/Lookup.h"

namespace Carbon {

// NOLINTBEGIN(modernize-loop-convert)
// NOLINTBEGIN(modernize-use-trailing-return-type)
// NOLINTBEGIN(readability-identifier-naming)
// NOLINTBEGIN(readability-redundant-nested-if)

char MultiplexExternalSemaSource::ID;

/// Constructs an empty multiplexing external sema source.
MultiplexExternalSemaSource::MultiplexExternalSemaSource() = default;

/// Constructs a new multiplexing external sema source and appends the
/// given element to it.
///
MultiplexExternalSemaSource::MultiplexExternalSemaSource(
    llvm::IntrusiveRefCntPtr<ExternalSemaSource> S1,
    llvm::IntrusiveRefCntPtr<ExternalSemaSource> S2) {
  Sources.push_back(std::move(S1));
  Sources.push_back(std::move(S2));
}

/// Appends new source to the source list.
///
///\param[in] source - An ExternalSemaSource.
///
void MultiplexExternalSemaSource::AddSource(
    llvm::IntrusiveRefCntPtr<ExternalSemaSource> Source) {
  Sources.push_back(std::move(Source));
}

//===----------------------------------------------------------------------===//
// ExternalASTSource.
//===----------------------------------------------------------------------===//

clang::Decl* MultiplexExternalSemaSource::GetExternalDecl(
    clang::GlobalDeclID ID) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    if (clang::Decl* Result = Sources[i]->GetExternalDecl(ID)) {
      return Result;
    }
  }
  return nullptr;
}

void MultiplexExternalSemaSource::CompleteRedeclChain(const clang::Decl* D) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->CompleteRedeclChain(D);
  }
}

clang::Selector MultiplexExternalSemaSource::GetExternalSelector(uint32_t ID) {
  clang::Selector Sel;
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sel = Sources[i]->GetExternalSelector(ID);
    if (!Sel.isNull()) {
      return Sel;
    }
  }
  return Sel;
}

uint32_t MultiplexExternalSemaSource::GetNumExternalSelectors() {
  uint32_t total = 0;
  for (size_t i = 0; i < Sources.size(); ++i) {
    total += Sources[i]->GetNumExternalSelectors();
  }
  return total;
}

clang::Stmt* MultiplexExternalSemaSource::GetExternalDeclStmt(uint64_t Offset) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    if (clang::Stmt* Result = Sources[i]->GetExternalDeclStmt(Offset)) {
      return Result;
    }
  }
  return nullptr;
}

clang::CXXBaseSpecifier*
MultiplexExternalSemaSource::GetExternalCXXBaseSpecifiers(uint64_t Offset) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    if (clang::CXXBaseSpecifier* R =
            Sources[i]->GetExternalCXXBaseSpecifiers(Offset)) {
      return R;
    }
  }
  return nullptr;
}

clang::CXXCtorInitializer**
MultiplexExternalSemaSource::GetExternalCXXCtorInitializers(uint64_t Offset) {
  for (auto& S : Sources) {
    if (auto* R = S->GetExternalCXXCtorInitializers(Offset)) {
      return R;
    }
  }
  return nullptr;
}

clang::ExternalASTSource::ExtKind
MultiplexExternalSemaSource::hasExternalDefinitions(const clang::Decl* D) {
  for (const auto& S : Sources) {
    if (auto EK = S->hasExternalDefinitions(D)) {
      if (EK != EK_ReplyHazy) {
        return EK;
      }
    }
  }
  return EK_ReplyHazy;
}

bool MultiplexExternalSemaSource::wasThisDeclarationADefinition(
    const clang::FunctionDecl* FD) {
  for (const auto& S : Sources) {
    if (S->wasThisDeclarationADefinition(FD)) {
      return true;
    }
  }
  return false;
}

bool MultiplexExternalSemaSource::FindExternalVisibleDeclsByName(
    const clang::DeclContext* DC, clang::DeclarationName Name,
    const clang::DeclContext* OriginalDC) {
  bool AnyDeclsFound = false;
  for (size_t i = 0; i < Sources.size(); ++i) {
    AnyDeclsFound |=
        Sources[i]->FindExternalVisibleDeclsByName(DC, Name, OriginalDC);
  }
  return AnyDeclsFound;
}

bool MultiplexExternalSemaSource::LoadExternalSpecializations(
    const clang::Decl* D, bool OnlyPartial) {
  bool Loaded = false;
  for (size_t i = 0; i < Sources.size(); ++i) {
    Loaded |= Sources[i]->LoadExternalSpecializations(D, OnlyPartial);
  }
  return Loaded;
}

bool MultiplexExternalSemaSource::LoadExternalSpecializations(
    const clang::Decl* D,
    llvm::ArrayRef<clang::TemplateArgument> TemplateArgs) {
  bool AnyNewSpecsLoaded = false;
  for (size_t i = 0; i < Sources.size(); ++i) {
    AnyNewSpecsLoaded |=
        Sources[i]->LoadExternalSpecializations(D, TemplateArgs);
  }
  return AnyNewSpecsLoaded;
}

void MultiplexExternalSemaSource::completeVisibleDeclsMap(
    const clang::DeclContext* DC) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->completeVisibleDeclsMap(DC);
  }
}

void MultiplexExternalSemaSource::FindExternalLexicalDecls(
    const clang::DeclContext* DC,
    llvm::function_ref<bool(clang::Decl::Kind)> IsKindWeWant,
    llvm::SmallVectorImpl<clang::Decl*>& Result) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->FindExternalLexicalDecls(DC, IsKindWeWant, Result);
  }
}

void MultiplexExternalSemaSource::FindFileRegionDecls(
    clang::FileID File, unsigned Offset, unsigned Length,
    llvm::SmallVectorImpl<clang::Decl*>& Decls) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->FindFileRegionDecls(File, Offset, Length, Decls);
  }
}

void MultiplexExternalSemaSource::CompleteType(clang::TagDecl* Tag) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->CompleteType(Tag);
  }
}

void MultiplexExternalSemaSource::CompleteType(
    clang::ObjCInterfaceDecl* Class) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->CompleteType(Class);
  }
}

void MultiplexExternalSemaSource::ReadComments() {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->ReadComments();
  }
}

void MultiplexExternalSemaSource::StartedDeserializing() {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->StartedDeserializing();
  }
}

void MultiplexExternalSemaSource::FinishedDeserializing() {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->FinishedDeserializing();
  }
}

void MultiplexExternalSemaSource::StartTranslationUnit(
    clang::ASTConsumer* Consumer) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->StartTranslationUnit(Consumer);
  }
}

void MultiplexExternalSemaSource::PrintStats() {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->PrintStats();
  }
}

clang::Module* MultiplexExternalSemaSource::getModule(unsigned ID) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    if (auto* M = Sources[i]->getModule(ID)) {
      return M;
    }
  }
  return nullptr;
}

bool MultiplexExternalSemaSource::layoutRecordType(
    const clang::RecordDecl* Record, uint64_t& Size, uint64_t& Alignment,
    llvm::DenseMap<const clang::FieldDecl*, uint64_t>& FieldOffsets,
    llvm::DenseMap<const clang::CXXRecordDecl*, clang::CharUnits>& BaseOffsets,
    llvm::DenseMap<const clang::CXXRecordDecl*, clang::CharUnits>&
        VirtualBaseOffsets) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    if (Sources[i]->layoutRecordType(Record, Size, Alignment, FieldOffsets,
                                     BaseOffsets, VirtualBaseOffsets)) {
      return true;
    }
  }
  return false;
}

void MultiplexExternalSemaSource::getMemoryBufferSizes(
    MemoryBufferSizes& sizes) const {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->getMemoryBufferSizes(sizes);
  }
}

//===----------------------------------------------------------------------===//
// ExternalSemaSource.
//===----------------------------------------------------------------------===//

void MultiplexExternalSemaSource::InitializeSema(clang::Sema& S) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->InitializeSema(S);
  }
}

void MultiplexExternalSemaSource::ForgetSema() {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->ForgetSema();
  }
}

void MultiplexExternalSemaSource::ReadMethodPool(clang::Selector Sel) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->ReadMethodPool(Sel);
  }
}

void MultiplexExternalSemaSource::updateOutOfDateSelector(clang::Selector Sel) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->updateOutOfDateSelector(Sel);
  }
}

void MultiplexExternalSemaSource::ReadKnownNamespaces(
    llvm::SmallVectorImpl<clang::NamespaceDecl*>& Namespaces) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->ReadKnownNamespaces(Namespaces);
  }
}

void MultiplexExternalSemaSource::ReadUndefinedButUsed(
    llvm::MapVector<clang::NamedDecl*, clang::SourceLocation>& Undefined) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->ReadUndefinedButUsed(Undefined);
  }
}

void MultiplexExternalSemaSource::ReadMismatchingDeleteExpressions(
    llvm::MapVector<
        clang::FieldDecl*,
        llvm::SmallVector<std::pair<clang::SourceLocation, bool>, 4>>& Exprs) {
  for (auto& Source : Sources) {
    Source->ReadMismatchingDeleteExpressions(Exprs);
  }
}

bool MultiplexExternalSemaSource::LookupUnqualified(clang::LookupResult& R,
                                                    clang::Scope* S) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->LookupUnqualified(R, S);
  }

  return !R.empty();
}

void MultiplexExternalSemaSource::ReadTentativeDefinitions(
    llvm::SmallVectorImpl<clang::VarDecl*>& TentativeDefs) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->ReadTentativeDefinitions(TentativeDefs);
  }
}

void MultiplexExternalSemaSource::ReadUnusedFileScopedDecls(
    llvm::SmallVectorImpl<const clang::DeclaratorDecl*>& Decls) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->ReadUnusedFileScopedDecls(Decls);
  }
}

void MultiplexExternalSemaSource::ReadDelegatingConstructors(
    llvm::SmallVectorImpl<clang::CXXConstructorDecl*>& Decls) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->ReadDelegatingConstructors(Decls);
  }
}

void MultiplexExternalSemaSource::ReadExtVectorDecls(
    llvm::SmallVectorImpl<clang::TypedefNameDecl*>& Decls) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->ReadExtVectorDecls(Decls);
  }
}

void MultiplexExternalSemaSource::ReadDeclsToCheckForDeferredDiags(
    llvm::SmallSetVector<clang::Decl*, 4>& Decls) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->ReadDeclsToCheckForDeferredDiags(Decls);
  }
}

void MultiplexExternalSemaSource::ReadUnusedLocalTypedefNameCandidates(
    llvm::SmallSetVector<const clang::TypedefNameDecl*, 4>& Decls) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->ReadUnusedLocalTypedefNameCandidates(Decls);
  }
}

void MultiplexExternalSemaSource::ReadReferencedSelectors(
    llvm::SmallVectorImpl<std::pair<clang::Selector, clang::SourceLocation>>&
        Sels) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->ReadReferencedSelectors(Sels);
  }
}

void MultiplexExternalSemaSource::ReadWeakUndeclaredIdentifiers(
    llvm::SmallVectorImpl<std::pair<clang::IdentifierInfo*, clang::WeakInfo>>&
        WI) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->ReadWeakUndeclaredIdentifiers(WI);
  }
}

void MultiplexExternalSemaSource::ReadExtnameUndeclaredIdentifiers(
    llvm::SmallVectorImpl<
        std::pair<clang::IdentifierInfo*, clang::AsmLabelAttr*>>& EI) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->ReadExtnameUndeclaredIdentifiers(EI);
  }
}

void MultiplexExternalSemaSource::ReadUsedVTables(
    llvm::SmallVectorImpl<clang::ExternalVTableUse>& VTables) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->ReadUsedVTables(VTables);
  }
}

void MultiplexExternalSemaSource::ReadPendingInstantiations(
    llvm::SmallVectorImpl<std::pair<clang::ValueDecl*, clang::SourceLocation>>&
        Pending) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->ReadPendingInstantiations(Pending);
  }
}

void MultiplexExternalSemaSource::ReadLateParsedTemplates(
    llvm::MapVector<const clang::FunctionDecl*,
                    std::unique_ptr<clang::LateParsedTemplate>>& LPTMap) {
  for (size_t i = 0; i < Sources.size(); ++i) {
    Sources[i]->ReadLateParsedTemplates(LPTMap);
  }
}

clang::TypoCorrection MultiplexExternalSemaSource::CorrectTypo(
    const clang::DeclarationNameInfo& Typo, int LookupKind, clang::Scope* S,
    clang::CXXScopeSpec* SS, clang::CorrectionCandidateCallback& CCC,
    clang::DeclContext* MemberContext, bool EnteringContext,
    const clang::ObjCObjectPointerType* OPT) {
  for (size_t I = 0, E = Sources.size(); I < E; ++I) {
    if (clang::TypoCorrection C =
            Sources[I]->CorrectTypo(Typo, LookupKind, S, SS, CCC, MemberContext,
                                    EnteringContext, OPT)) {
      return C;
    }
  }
  return clang::TypoCorrection();
}

bool MultiplexExternalSemaSource::MaybeDiagnoseMissingCompleteType(
    clang::SourceLocation Loc, clang::QualType T) {
  for (size_t I = 0, E = Sources.size(); I < E; ++I) {
    if (Sources[I]->MaybeDiagnoseMissingCompleteType(Loc, T)) {
      return true;
    }
  }
  return false;
}

void MultiplexExternalSemaSource::AssignedLambdaNumbering(
    clang::CXXRecordDecl* Lambda) {
  for (auto& Source : Sources) {
    Source->AssignedLambdaNumbering(Lambda);
  }
}

// NOLINTEND(readability-redundant-nested-if)
// NOLINTEND(readability-identifier-naming)
// NOLINTEND(modernize-use-trailing-return-type)
// NOLINTEND(modernize-loop-convert)

}  // namespace Carbon
