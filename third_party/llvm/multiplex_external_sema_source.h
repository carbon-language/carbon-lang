// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TODO: Remove this once https://github.com/llvm/llvm-project/pull/204458
// is merged.

#ifndef CARBON_THIRD_PARTY_LLVM_MULTIPLEX_EXTERNAL_SEMA_SOURCE_H_
#define CARBON_THIRD_PARTY_LLVM_MULTIPLEX_EXTERNAL_SEMA_SOURCE_H_

#include <utility>

#include "clang/Sema/ExternalSemaSource.h"
#include "clang/Sema/Weak.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace Carbon {

class CXXConstructorDecl;
class CXXRecordDecl;
class DeclaratorDecl;
struct ExternalVTableUse;
class LookupResult;
class NamespaceDecl;
class Scope;
class Sema;
class TypedefNameDecl;
class ValueDecl;
class VarDecl;

// NOLINTBEGIN(modernize-use-trailing-return-type)
// NOLINTBEGIN(readability-identifier-naming)

/// An abstract interface that should be implemented by
/// external AST sources that also provide information for semantic
/// analysis.
class MultiplexExternalSemaSource : public clang::ExternalSemaSource {
  /// LLVM-style RTTI.
  static char ID;

 private:
  llvm::SmallVector<llvm::IntrusiveRefCntPtr<ExternalSemaSource>, 2> Sources;

 public:
  /// Constructs an empty multiplexing external sema source.
  MultiplexExternalSemaSource();

  /// Constructs a new multiplexing external sema source and appends the
  /// given element to it.
  ///
  ///\param[in] S1 - A non-null (old) ExternalSemaSource.
  ///\param[in] S2 - A non-null (new) ExternalSemaSource.
  ///
  MultiplexExternalSemaSource(llvm::IntrusiveRefCntPtr<ExternalSemaSource> S1,
                              llvm::IntrusiveRefCntPtr<ExternalSemaSource> S2);

  /// Appends new source to the source list.
  ///
  ///\param[in] Source - An ExternalSemaSource.
  ///
  void AddSource(llvm::IntrusiveRefCntPtr<ExternalSemaSource> Source);

  /// Remove all sources for which the predicate returns true.
  ///
  /// \param P - A predicate that takes an
  /// IntrusiveRefCntPtr<ExternalSemaSource> param and returns true if
  /// the source should be removed, false otherwise.
  template <typename UnaryPredicate>
  void EraseIf(UnaryPredicate P) {
    llvm::erase_if(Sources, P);
  }

  //===--------------------------------------------------------------------===//
  // ExternalASTSource.
  //===--------------------------------------------------------------------===//

  /// Resolve a declaration ID into a declaration, potentially
  /// building a new declaration.
  clang::Decl* GetExternalDecl(clang::GlobalDeclID ID) override;

  /// Complete the redeclaration chain if it's been extended since the
  /// previous generation of the AST source.
  void CompleteRedeclChain(const clang::Decl* D) override;

  /// Resolve a selector ID into a selector.
  clang::Selector GetExternalSelector(uint32_t ID) override;

  /// Returns the number of selectors known to the external AST
  /// source.
  uint32_t GetNumExternalSelectors() override;

  /// Resolve the offset of a statement in the decl stream into
  /// a statement.
  clang::Stmt* GetExternalDeclStmt(uint64_t Offset) override;

  /// Resolve the offset of a set of C++ base specifiers in the decl
  /// stream into an array of specifiers.
  clang::CXXBaseSpecifier* GetExternalCXXBaseSpecifiers(
      uint64_t Offset) override;

  /// Resolve a handle to a list of ctor initializers into the list of
  /// initializers themselves.
  clang::CXXCtorInitializer** GetExternalCXXCtorInitializers(
      uint64_t Offset) override;

  ExtKind hasExternalDefinitions(const clang::Decl* D) override;

  bool wasThisDeclarationADefinition(const clang::FunctionDecl* FD) override;

  /// Find all declarations with the given name in the
  /// given context.
  bool FindExternalVisibleDeclsByName(
      const clang::DeclContext* DC, clang::DeclarationName Name,
      const clang::DeclContext* OriginalDC) override;

  bool LoadExternalSpecializations(const clang::Decl* D,
                                   bool OnlyPartial) override;

  bool LoadExternalSpecializations(
      const clang::Decl* D,
      llvm::ArrayRef<clang::TemplateArgument> TemplateArgs) override;

  /// Ensures that the table of all visible declarations inside this
  /// context is up to date.
  void completeVisibleDeclsMap(const clang::DeclContext* DC) override;

  /// Finds all declarations lexically contained within the given
  /// DeclContext, after applying an optional filter predicate.
  ///
  /// \param IsKindWeWant a predicate function that returns true if the passed
  /// declaration kind is one we are looking for.
  void FindExternalLexicalDecls(
      const clang::DeclContext* DC,
      llvm::function_ref<bool(clang::Decl::Kind)> IsKindWeWant,
      llvm::SmallVectorImpl<clang::Decl*>& Result) override;

  /// Get the decls that are contained in a file in the Offset/Length
  /// range. \p Length can be 0 to indicate a point at \p Offset instead of
  /// a range.
  void FindFileRegionDecls(clang::FileID File, unsigned Offset, unsigned Length,
                           llvm::SmallVectorImpl<clang::Decl*>& Decls) override;

  /// Gives the external AST source an opportunity to complete
  /// an incomplete type.
  void CompleteType(clang::TagDecl* Tag) override;

  /// Gives the external AST source an opportunity to complete an
  /// incomplete Objective-C class.
  ///
  /// This routine will only be invoked if the "externally completed" bit is
  /// set on the ObjCInterfaceDecl via the function
  /// \c ObjCInterfaceDecl::setExternallyCompleted().
  void CompleteType(clang::ObjCInterfaceDecl* Class) override;

  /// Loads comment ranges.
  void ReadComments() override;

  /// Notify ExternalASTSource that we started deserialization of
  /// a decl or type so until FinishedDeserializing is called there may be
  /// decls that are initializing. Must be paired with FinishedDeserializing.
  void StartedDeserializing() override;

  /// Notify ExternalASTSource that we finished the deserialization of
  /// a decl or type. Must be paired with StartedDeserializing.
  void FinishedDeserializing() override;

  /// Function that will be invoked when we begin parsing a new
  /// translation unit involving this external AST source.
  void StartTranslationUnit(clang::ASTConsumer* Consumer) override;

  /// Print any statistics that have been gathered regarding
  /// the external AST source.
  void PrintStats() override;

  /// Retrieve the module that corresponds to the given module ID.
  clang::Module* getModule(unsigned ID) override;

  /// Perform layout on the given record.
  ///
  /// This routine allows the external AST source to provide an specific
  /// layout for a record, overriding the layout that would normally be
  /// constructed. It is intended for clients who receive specific layout
  /// details rather than source code (such as LLDB). The client is expected
  /// to fill in the field offsets, base offsets, virtual base offsets, and
  /// complete object size.
  ///
  /// \param Record The record whose layout is being requested.
  ///
  /// \param Size The final size of the record, in bits.
  ///
  /// \param Alignment The final alignment of the record, in bits.
  ///
  /// \param FieldOffsets The offset of each of the fields within the record,
  /// expressed in bits. All of the fields must be provided with offsets.
  ///
  /// \param BaseOffsets The offset of each of the direct, non-virtual base
  /// classes. If any bases are not given offsets, the bases will be laid
  /// out according to the ABI.
  ///
  /// \param VirtualBaseOffsets The offset of each of the virtual base classes
  /// (either direct or not). If any bases are not given offsets, the bases will
  /// be laid out according to the ABI.
  ///
  /// \returns true if the record layout was provided, false otherwise.
  bool layoutRecordType(
      const clang::RecordDecl* Record, uint64_t& Size, uint64_t& Alignment,
      llvm::DenseMap<const clang::FieldDecl*, uint64_t>& FieldOffsets,
      llvm::DenseMap<const clang::CXXRecordDecl*, clang::CharUnits>&
          BaseOffsets,
      llvm::DenseMap<const clang::CXXRecordDecl*, clang::CharUnits>&
          VirtualBaseOffsets) override;

  /// Return the amount of memory used by memory buffers, breaking down
  /// by heap-backed versus mmap'ed memory.
  void getMemoryBufferSizes(MemoryBufferSizes& sizes) const override;

  //===--------------------------------------------------------------------===//
  // ExternalSemaSource.
  //===--------------------------------------------------------------------===//

  /// Initialize the semantic source with the Sema instance
  /// being used to perform semantic analysis on the abstract syntax
  /// tree.
  void InitializeSema(clang::Sema& S) override;

  /// Inform the semantic consumer that Sema is no longer available.
  void ForgetSema() override;

  /// Load the contents of the global method pool for a given
  /// selector.
  void ReadMethodPool(clang::Selector Sel) override;

  /// Load the contents of the global method pool for a given
  /// selector if necessary.
  void updateOutOfDateSelector(clang::Selector Sel) override;

  /// Load the set of namespaces that are known to the external source,
  /// which will be used during typo correction.
  void ReadKnownNamespaces(
      llvm::SmallVectorImpl<clang::NamespaceDecl*>& Namespaces) override;

  /// Load the set of used but not defined functions or variables with
  /// internal linkage, or used but not defined inline functions.
  void ReadUndefinedButUsed(
      llvm::MapVector<clang::NamedDecl*, clang::SourceLocation>& Undefined)
      override;

  void ReadMismatchingDeleteExpressions(
      llvm::MapVector<clang::FieldDecl*,
                      llvm::SmallVector<std::pair<clang::SourceLocation, bool>,
                                        4>>& Exprs) override;

  /// Do last resort, unqualified lookup on a LookupResult that
  /// Sema cannot find.
  ///
  /// \param R a LookupResult that is being recovered.
  ///
  /// \param S the Scope of the identifier occurrence.
  ///
  /// \return true to tell Sema to recover using the LookupResult.
  bool LookupUnqualified(clang::LookupResult& R, clang::Scope* S) override;

  /// Read the set of tentative definitions known to the external Sema
  /// source.
  ///
  /// The external source should append its own tentative definitions to the
  /// given vector of tentative definitions. Note that this routine may be
  /// invoked multiple times; the external source should take care not to
  /// introduce the same declarations repeatedly.
  void ReadTentativeDefinitions(
      llvm::SmallVectorImpl<clang::VarDecl*>& Defs) override;

  /// Read the set of unused file-scope declarations known to the
  /// external Sema source.
  ///
  /// The external source should append its own unused, filed-scope to the
  /// given vector of declarations. Note that this routine may be
  /// invoked multiple times; the external source should take care not to
  /// introduce the same declarations repeatedly.
  void ReadUnusedFileScopedDecls(
      llvm::SmallVectorImpl<const clang::DeclaratorDecl*>& Decls) override;

  /// Read the set of delegating constructors known to the
  /// external Sema source.
  ///
  /// The external source should append its own delegating constructors to the
  /// given vector of declarations. Note that this routine may be
  /// invoked multiple times; the external source should take care not to
  /// introduce the same declarations repeatedly.
  void ReadDelegatingConstructors(
      llvm::SmallVectorImpl<clang::CXXConstructorDecl*>& Decls) override;

  /// Read the set of ext_vector type declarations known to the
  /// external Sema source.
  ///
  /// The external source should append its own ext_vector type declarations to
  /// the given vector of declarations. Note that this routine may be
  /// invoked multiple times; the external source should take care not to
  /// introduce the same declarations repeatedly.
  void ReadExtVectorDecls(
      llvm::SmallVectorImpl<clang::TypedefNameDecl*>& Decls) override;

  /// Read the set of potentially unused typedefs known to the source.
  ///
  /// The external source should append its own potentially unused local
  /// typedefs to the given vector of declarations. Note that this routine may
  /// be invoked multiple times; the external source should take care not to
  /// introduce the same declarations repeatedly.
  void ReadUnusedLocalTypedefNameCandidates(
      llvm::SmallSetVector<const clang::TypedefNameDecl*, 4>& Decls) override;

  /// Read the set of referenced selectors known to the
  /// external Sema source.
  ///
  /// The external source should append its own referenced selectors to the
  /// given vector of selectors. Note that this routine
  /// may be invoked multiple times; the external source should take care not
  /// to introduce the same selectors repeatedly.
  void ReadReferencedSelectors(
      llvm::SmallVectorImpl<std::pair<clang::Selector, clang::SourceLocation>>&
          Sels) override;

  /// Read the set of weak, undeclared identifiers known to the
  /// external Sema source.
  ///
  /// The external source should append its own weak, undeclared identifiers to
  /// the given vector. Note that this routine may be invoked multiple times;
  /// the external source should take care not to introduce the same identifiers
  /// repeatedly.
  void ReadWeakUndeclaredIdentifiers(
      llvm::SmallVectorImpl<std::pair<clang::IdentifierInfo*, clang::WeakInfo>>&
          WI) override;

  /// Read the set of #pragma redefine_extname'd, undeclared identifiers known
  /// to the external Sema source.
  ///
  /// The external source should append its own #pragma redefine_extname'd,
  /// undeclared identifiers to the given vector. Note that this routine may be
  /// invoked multiple times; the external source should take care not to
  /// introduce the same identifiers repeatedly.
  void ReadExtnameUndeclaredIdentifiers(
      llvm::SmallVectorImpl<std::pair<clang::IdentifierInfo*,
                                      clang::AsmLabelAttr*>>& EI) override;

  /// Read the set of used vtables known to the external Sema source.
  ///
  /// The external source should append its own used vtables to the given
  /// vector. Note that this routine may be invoked multiple times; the external
  /// source should take care not to introduce the same vtables repeatedly.
  void ReadUsedVTables(
      llvm::SmallVectorImpl<clang::ExternalVTableUse>& VTables) override;

  /// Read the set of pending instantiations known to the external
  /// Sema source.
  ///
  /// The external source should append its own pending instantiations to the
  /// given vector. Note that this routine may be invoked multiple times; the
  /// external source should take care not to introduce the same instantiations
  /// repeatedly.
  void ReadPendingInstantiations(
      llvm::SmallVectorImpl<
          std::pair<clang::ValueDecl*, clang::SourceLocation>>& Pending)
      override;

  /// Read the set of late parsed template functions for this source.
  ///
  /// The external source should insert its own late parsed template functions
  /// into the map. Note that this routine may be invoked multiple times; the
  /// external source should take care not to introduce the same map entries
  /// repeatedly.
  void ReadLateParsedTemplates(
      llvm::MapVector<const clang::FunctionDecl*,
                      std::unique_ptr<clang::LateParsedTemplate>>& LPTMap)
      override;

  /// Read the set of decls to be checked for deferred diags.
  ///
  /// The external source should append its own potentially emitted function
  /// and variable decls which may cause deferred diags. Note that this routine
  /// may be invoked multiple times; the external source should take care not to
  /// introduce the same declarations repeatedly.
  void ReadDeclsToCheckForDeferredDiags(
      llvm::SmallSetVector<clang::Decl*, 4>& Decls) override;

  /// \copydoc ExternalSemaSource::CorrectTypo
  /// \note Returns the first nonempty correction.
  clang::TypoCorrection CorrectTypo(
      const clang::DeclarationNameInfo& Typo, int LookupKind, clang::Scope* S,
      clang::CXXScopeSpec* SS, clang::CorrectionCandidateCallback& CCC,
      clang::DeclContext* MemberContext, bool EnteringContext,
      const clang::ObjCObjectPointerType* OPT) override;

  /// Produces a diagnostic note if one of the attached sources
  /// contains a complete definition for \p T. Queries the sources in list
  /// order until the first one claims that a diagnostic was produced.
  ///
  /// \param Loc the location at which a complete type was required but not
  /// provided
  ///
  /// \param T the \c QualType that should have been complete at \p Loc
  ///
  /// \return true if a diagnostic was produced, false otherwise.
  bool MaybeDiagnoseMissingCompleteType(clang::SourceLocation Loc,
                                        clang::QualType T) override;

  // Inform all attached sources that a mangling number was assigned.
  void AssignedLambdaNumbering(clang::CXXRecordDecl* Lambda) override;

  /// LLVM-style RTTI.
  /// \{
  bool isA(const void* ClassID) const override {
    return ClassID == &ID || ExternalSemaSource::isA(ClassID);
  }
  static bool classof(const clang::ExternalASTSource* S) { return S->isA(&ID); }
  /// \}
};

// NOLINTEND(readability-identifier-naming)
// NOLINTEND(modernize-use-trailing-return-type)

}  // namespace Carbon

#endif  // CARBON_THIRD_PARTY_LLVM_MULTIPLEX_EXTERNAL_SEMA_SOURCE_H_
