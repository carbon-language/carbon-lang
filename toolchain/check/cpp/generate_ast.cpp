// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/generate_ast.h"

#include <memory>
#include <string>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/DiagnosticParse.h"
#include "clang/Basic/FileManager.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/TextDiagnostic.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/Sema/MultiplexExternalSemaSource.h"
#include "clang/Sema/Sema.h"
#include "common/check.h"
#include "common/map.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/cpp/access.h"
#include "toolchain/check/cpp/diagnostic_consumer.h"
#include "toolchain/check/cpp/diagnostic_listener.h"
#include "toolchain/check/cpp/domain.h"
#include "toolchain/check/cpp/export.h"
#include "toolchain/check/cpp/import.h"
#include "toolchain/check/cpp/location.h"
#include "toolchain/check/cpp/type_mapping.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/diagnostics/emitter.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/cpp_file.h"
#include "toolchain/sem_ir/read_only_ast_source.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Add a line marker directive pointing at the location of the `import Cpp`
// declaration in the Carbon source file. This will cause Clang's diagnostics
// machinery to track and report the location in Carbon code where the import
// was written.
static auto GenerateLineMarker(Context& context, llvm::raw_ostream& out,
                               int line) {
  out << "# " << line << " \""
      << FormatEscaped(context.tokens().source().filename()) << "\"\n";
}

// Appends a line marker and the specified `code` to `out`, adjusting the
// `line` number if the `code_token` represents a block string literal.
static auto AppendInlineCode(Context& context, llvm::raw_ostream& out,
                             Lex::TokenIndex code_token, llvm::StringRef code)
    -> void {
  // Compute the line number on which the C++ code starts. Usually the code
  // is specified as a block string literal and starts on the line after the
  // start of the string token.
  // TODO: Determine if this is a block string literal without calling
  // `GetTokenText`, which re-lexes the string.
  int line = context.tokens().GetLineNumber(code_token);
  if (context.tokens().GetTokenText(code_token).contains('\n')) {
    ++line;
  }

  GenerateLineMarker(context, out, line);
  out << code << "\n";
}

// Generates C++ file contents to #include all requested imports.
static auto GenerateCppIncludesHeaderCode(
    Context& context, llvm::ArrayRef<Parse::Tree::PackagingNames> imports)
    -> std::string {
  RawStringOstream code_stream;
  for (const Parse::Tree::PackagingNames& import : imports) {
    if (import.inline_body_id.has_value()) {
      // Expand `import Cpp inline "code";` directly into the specified code.
      auto code_token = context.parse_tree().node_token(import.inline_body_id);
      AppendInlineCode(context, code_stream, code_token,
                       context.string_literal_values().Get(
                           context.tokens().GetStringLiteralValue(code_token)));
      // TODO: Inject a clang pragma here to produce an error if there are
      // unclosed scopes at the end of this inline C++ fragment.
    } else if (import.library_id.has_value()) {
      // Translate `import Cpp library "foo.h";` into `#include "foo.h"`.
      GenerateLineMarker(context, code_stream,
                         context.tokens().GetLineNumber(
                             context.parse_tree().node_token(import.node_id)));
      auto name = context.string_literal_values().Get(import.library_id);
      if (name.starts_with('<') && name.ends_with('>')) {
        code_stream << "#include <"
                    << FormatEscaped(name.drop_front().drop_back()) << ">\n";
      } else {
        code_stream << "#include \"" << FormatEscaped(name) << "\"\n";
      }
    }
  }
  return code_stream.TakeStr();
}

namespace {

// A wrapper around a clang::CompilerInvocation that allows us to make a shallow
// copy of most of the invocation and only make a deep copy of the parts that we
// want to change.
//
// clang::CowCompilerInvocation almost allows this, but doesn't derive from
// CompilerInvocation or support shallow copies from a CompilerInvocation, so is
// not useful to us as we can't build an ASTUnit from it.
class ShallowCopyCompilerInvocation : public clang::CompilerInvocation {
 public:
  explicit ShallowCopyCompilerInvocation(
      const clang::CompilerInvocation& invocation) {
    shallow_copy_assign(invocation);

    // Make a deep copy of options that we modify.
    FrontendOpts = std::make_shared<clang::FrontendOptions>(*FrontendOpts);
    PPOpts = std::make_shared<clang::PreprocessorOptions>(*PPOpts);
  }
};

// Provides clang AST nodes representing Carbon SemIR entities.
class CarbonExternalASTSource : public SemIR::ReadOnlyASTSource {
 public:
  explicit CarbonExternalASTSource(Context* context)
      : ReadOnlyASTSource(context->sem_ir()), context_(context) {}

  // Builds the top-level C++ namespace `Carbon` and adds it to the translation
  // unit.
  auto BuildCarbonNamespace() -> void;

  // Look up decls for `decl_name` inside `decl_context`, adding the decls to
  // `decl_context`. Returns true if any decls were added.
  auto FindExternalVisibleDeclsByName(
      const clang::DeclContext* decl_context, clang::DeclarationName decl_name,
      const clang::DeclContext* original_decl_context) -> bool override;

  auto LoadExternalSpecializations(
      const clang::Decl* decl,
      llvm::ArrayRef<clang::TemplateArgument> template_args) -> bool override {
    const auto* function_template_decl =
        llvm::dyn_cast<clang::FunctionTemplateDecl>(decl);
    if (!function_template_decl) {
      return false;
    }

    return ExportFunctionSpecializationToCpp(
        *context_,
        const_cast<clang::FunctionTemplateDecl*>(function_template_decl),
        template_args);
  }

  auto CompleteType(clang::TagDecl* tag_decl) -> void override;

  auto layoutRecordType(
      const clang::RecordDecl* record_decl, uint64_t& size, uint64_t& alignment,
      llvm::DenseMap<const clang::FieldDecl*, uint64_t>& field_offsets,
      llvm::DenseMap<const clang::CXXRecordDecl*, clang::CharUnits>&
          base_offsets,
      llvm::DenseMap<const clang::CXXRecordDecl*, clang::CharUnits>&
          vbase_offsets) -> bool override;

  auto isA(const void* class_id) const -> bool override {
    return class_id == &id || ReadOnlyASTSource::isA(class_id);
  }
  static auto classof(const ExternalASTSource* s) -> bool {
    return s->isA(&id);
  }

 private:
  // Map a Carbon entity to a Clang NamedDecl. Returns null if the entity cannot
  // currently be represented in C++.
  auto MapInstIdToClangDeclOrType(LookupResult lookup)
      -> std::variant<clang::NamedDecl*, clang::QualType>;

  auto GetOrExportFunctionToCpp(SemIR::InstId target_inst_id,
                                SemIR::FunctionId function_id)
      -> clang::NamedDecl*;
  // Get a current best-effort location for the current position within C++
  // processing.
  auto GetCurrentCppLocId() -> SemIR::LocId {
    auto* cpp_context = context_->cpp_context();
    CARBON_CHECK(cpp_context);

    // Use the current token location when parsing.
    auto clang_source_loc = cpp_context->parser().getCurToken().getLocation();
    if (auto& code_synthesis_contexts =
            cpp_context->sema().CodeSynthesisContexts;
        !code_synthesis_contexts.empty()) {
      // Use the current point of instantiation during template instantiation.
      clang_source_loc = code_synthesis_contexts.back().PointOfInstantiation;
    }

    return AddImportIRInst(context_->sem_ir(), clang_source_loc);
  }

  // For LLVM RTTI.
  static char id;

  Check::Context* context_;
};

char CarbonExternalASTSource::id;

}  // namespace

auto CarbonExternalASTSource::MapInstIdToClangDeclOrType(LookupResult lookup)
    -> std::variant<clang::NamedDecl*, clang::QualType> {
  auto target_inst_id = lookup.scope_result.target_inst_id();
  auto target_const_id = context_->constant_values().Get(target_inst_id);
  auto target_inst = context_->constant_values().GetInst(target_const_id);

  if (target_inst.type_id() == SemIR::TypeType::TypeId) {
    auto type_id =
        context_->types().GetTypeIdForTypeConstantId(target_const_id);
    auto type = MapToCppType(*context_, type_id);
    if (type.isNull()) {
      context_->TODO(GetCurrentCppLocId(), "interop with unsupported type");
      return nullptr;
    }
    return type;
  }

  CARBON_KIND_SWITCH(target_inst) {
    case CARBON_KIND(SemIR::Namespace namespace_info): {
      auto* decl_context =
          ExportNameScopeToCpp(*context_, SemIR::LocId(target_inst_id),
                               namespace_info.name_scope_id);
      if (!decl_context) {
        return nullptr;
      }
      if (isa<clang::TranslationUnitDecl>(decl_context)) {
        context_->TODO(GetCurrentCppLocId(),
                       "interop with translation unit decl");
        return nullptr;
      }
      return cast<clang::NamedDecl>(decl_context);
    }
    case SemIR::StructValue::Kind: {
      auto callee = GetCallee(context_->sem_ir(), target_inst_id);
      auto* callee_function = std::get_if<SemIR::CalleeFunction>(&callee);
      if (!callee_function) {
        return nullptr;
      }

      return GetOrExportFunctionToCpp(target_inst_id,
                                      callee_function->function_id);
    }
    case CARBON_KIND(SemIR::FieldDecl field_decl): {
      return ExportFieldToCpp(*context_, target_inst_id, field_decl);
    }
    case CARBON_KIND(SemIR::VarStorage var_storage): {
      return ExportVarToCpp(*context_, target_inst_id, var_storage);
    }
    default:
      return nullptr;
  }
}

auto CarbonExternalASTSource::GetOrExportFunctionToCpp(
    SemIR::InstId target_inst_id, SemIR::FunctionId function_id)
    -> clang::NamedDecl* {
  SemIR::Function& function = context_->functions().Get(function_id);
  if (const auto* clang_decl =
          context_->clang_decls().Lookup(function.first_decl_id())) {
    return cast<clang::NamedDecl>(clang_decl->decl());
  }

  auto* named_decl =
      ExportFunctionToCpp(*context_, SemIR::LocId(target_inst_id), function_id);
  if (!named_decl) {
    return nullptr;
  }

  if (auto* function_template_decl =
          llvm::dyn_cast<clang::FunctionTemplateDecl>(named_decl)) {
    context_->clang_decls().Add(
        {.key = SemIR::ClangDeclKey::ForNonFunctionDecl(function_template_decl),
         .inst_id = function.first_decl_id()});
    return function_template_decl;
  }

  auto* clang_function_decl = llvm::cast<clang::FunctionDecl>(named_decl);

  SemIR::ClangDeclSignature thunk_signature;
  thunk_signature.kind = SemIR::ClangDeclSignature::Normal;
  thunk_signature.num_params =
      static_cast<int32_t>(clang_function_decl->getNumParams());
  thunk_signature.passing_modes.assign(
      thunk_signature.num_params,
      SemIR::ClangDeclSignature::PassingMode::ByValue);
  context_->clang_decls().Add(
      {.key = SemIR::ClangDeclKey::ForFunctionDecl(
           clang_function_decl,
           context_->clang_decl_signatures().Add(std::move(thunk_signature))),
       .inst_id = function.first_decl_id()});
  return clang_function_decl;
}

auto CarbonExternalASTSource::BuildCarbonNamespace() -> void {
  static const llvm::StringLiteral carbon_namespace_name = "Carbon";
  auto& ast_context = context_->ast_context();
  auto* identifier = &ast_context.Idents.get(carbon_namespace_name);

  auto* decl_context = ast_context.getTranslationUnitDecl();

  // Check if it already exists.
  clang::NamespaceDecl* carbon_cpp_namespace = nullptr;
  auto lookup_result = decl_context->lookup(identifier);
  if (!lookup_result.empty()) {
    carbon_cpp_namespace = cast<clang::NamespaceDecl>(lookup_result.front());
  } else {
    // Create it if it doesn't exist.
    carbon_cpp_namespace = clang::NamespaceDecl::Create(
        ast_context, decl_context, /*Inline=*/false, clang::SourceLocation(),
        clang::SourceLocation(), identifier, /*PrevDecl=*/nullptr,
        /*Nested=*/false);
    decl_context->addDecl(carbon_cpp_namespace);

    // We provide custom lookup results within this namespace.
    carbon_cpp_namespace->setHasExternalVisibleStorage();
  }

  // Register this file's package scope as corresponding to the `Carbon`
  // namespace in C++.
  // TODO: For mangling purposes, include the package as a sub-namespace.
  auto key = SemIR::ClangDeclKey::ForNonFunctionDecl(carbon_cpp_namespace);
  auto clang_decl_id = context_->clang_decls().Add(
      {.key = key, .inst_id = SemIR::Namespace::PackageInstId});
  context_->name_scopes()
      .Get(SemIR::NameScopeId::Package)
      .set_clang_decl_context_id(clang_decl_id, /*is_cpp_scope=*/false);
}

auto CarbonExternalASTSource::FindExternalVisibleDeclsByName(
    const clang::DeclContext* decl_context, clang::DeclarationName decl_name,
    const clang::DeclContext* /*OriginalDC*/) -> bool {
  // Find the Carbon declaration corresponding to this Clang declaration.
  auto* decl = cast<clang::Decl>(
      const_cast<clang::DeclContext*>(decl_context->getPrimaryContext()));
  if (isa<clang::FunctionDecl>(decl)) {
    // Functions don't meaningfully have visible decls, but bail out early since
    // we can't form a `ClangDeclKey` for a function in the abstract.
    return false;
  }
  auto key = SemIR::ClangDeclKey::ForNonFunctionDecl(decl);
  auto decl_id = context_->clang_decls().LookupId(key);
  if (!decl_id.has_value()) {
    return false;
  }
  auto clang_decl = context_->clang_decls().Get(decl_id);
  if (clang_decl.is_imported) {
    // This is imported from C++, presumably from a Clang AST file, so it's not
    // our responsibility to provide its name lookup results.
    return false;
  }

  llvm::SmallVector<Check::LookupScope> lookup_scopes;

  // LocId::None seems fine here because we shouldn't produce any diagnostics
  // here - completeness should've been checked by clang before this point.
  if (!AppendLookupScopesForConstant(
          *context_, SemIR::LocId::None,
          context_->constant_values().Get(clang_decl.inst_id),
          SemIR::ConstantId::None, /*extended_scope=*/false, &lookup_scopes)) {
    return false;
  }

  clang::IdentifierInfo* identifier = nullptr;
  switch (decl_name.getNameKind()) {
    case clang::DeclarationName::Identifier: {
      identifier = decl_name.getAsIdentifierInfo();
      break;
    }
    case clang::DeclarationName::CXXConstructorName: {
      // The Carbon counterpart of a constructor is a function whose name
      // matches the class name.
      identifier =
          llvm::cast<clang::CXXRecordDecl>(decl_context)->getIdentifier();
      break;
    }
    default:
      return false;
  }

  auto name_id = AddIdentifierName(*context_, identifier->getName());

  // `required=false` so Carbon doesn't diagnose a failure, let Clang diagnose
  // it or even SFINAE.
  LookupResult result =
      LookupQualifiedName(*context_, SemIR::LocId::None, name_id, lookup_scopes,
                          /*required=*/false);
  if (!result.scope_result.is_found()) {
    return false;
  }

  // Map the found Carbon entity to a Clang NamedDecl.
  CARBON_KIND_SWITCH(MapInstIdToClangDeclOrType(result)) {
    case CARBON_KIND(clang::NamedDecl* clang_decl): {
      if (clang_decl) {
        SetExternalVisibleDeclsForName(decl_context, decl_name, {clang_decl});
        return true;
      } else {
        SetNoExternalVisibleDeclsForName(decl_context, decl_name);
        return false;
      }
    }

    case CARBON_KIND(clang::QualType type): {
      // Create a typedef declaration to model the type result.
      // TODO: If the type is a tag type that was declared with this name in
      // this context, use the tag decl directly.
      auto& ast_context = context_->ast_context();
      auto loc = GetCppLocation(
          *context_, SemIR::LocId(result.scope_result.target_inst_id()));
      auto* typedef_decl = clang::TypedefDecl::Create(
          ast_context, const_cast<clang::DeclContext*>(decl_context), loc, loc,
          identifier, ast_context.getTrivialTypeSourceInfo(type, loc));
      if (isa<clang::CXXRecordDecl>(decl_context)) {
        typedef_decl->setAccess(
            MapToCppAccess(result.scope_result.access_kind()));
      }
      SetExternalVisibleDeclsForName(decl_context, decl_name, {typedef_decl});
      return true;
    }
  }
}

auto CarbonExternalASTSource::CompleteType(clang::TagDecl* tag_decl) -> void {
  auto* class_decl = dyn_cast<clang::CXXRecordDecl>(tag_decl);
  if (!class_decl) {
    // TODO: If we start producing clang EnumTypes, we may have to handle them
    // here too.
    return;
  }

  auto carbon_class_info =
      SemIR::GetAsCarbonOwnedClass(context_->sem_ir(), tag_decl);
  if (!carbon_class_info) {
    return;
  }
  auto& [class_type_id, class_type] = *carbon_class_info;

  auto context_fn = [](DiagnosticContextBuilder& /*builder*/) -> void {};
  if (!RequireCompleteType(*context_, class_type_id, GetCurrentCppLocId(),
                           context_fn)) {
    return;
  }

  auto& class_info = context_->classes().Get(class_type.class_id);
  class_decl->startDefinition();
  CARBON_CHECK(class_decl->hasDefinition());

  // If the Carbon class is final, mark the C++ class as also being `final`.
  // Abstract classes are handled when generating the destructor declaration.
  if (class_info.inheritance_kind == SemIR::Class::InheritanceKind::Final) {
    // TODO: Find the location of the `final` modifier and use it here.
    class_decl->addAttr(clang::FinalAttr::Create(
        context_->ast_context(),
        GetCppLocation(*context_, SemIR::LocId(class_info.definition_id))));
  }

  // If the Carbon class has a base class that we can map into C++, add that as
  // a C++ base class.
  auto base_type_id =
      class_info.GetBaseType(context_->sem_ir(), class_type.specific_id);
  if (base_type_id.has_value()) {
    auto base_loc = GetCppLocation(*context_, SemIR::LocId(class_info.base_id));
    if (auto base_type = MapToCppType(*context_, base_type_id);
        !base_type.isNull() && base_type->isStructureOrClassType() &&
        !context_->clang_sema().RequireCompleteType(
            base_loc, base_type, clang::diag::err_incomplete_base_class)) {
      bool is_virtual = false;
      bool is_base_of_class = true;
      clang::CXXBaseSpecifier base(
          base_loc, is_virtual, is_base_of_class, clang::AS_public,
          context_->ast_context().getTrivialTypeSourceInfo(base_type, base_loc),
          /*EllipsisLoc=*/clang::SourceLocation());
      clang::CXXBaseSpecifier* bases[1] = {&base};
      CARBON_CHECK(class_decl->hasDefinition());
      class_decl->setBases(bases, 1);
    }
  }

  ExportAllFieldsToCpp(*context_, class_info);

  class_decl->addDecl(ExportDestructorToCpp(*context_, class_info, class_decl));

  // TODO: Import any special member functions that affect class properties.

  // Virtual functions whose definitions we have deferred generating until the
  // class is complete.
  struct PendingVirtualFunction {
    SemIR::LocId loc_id;
    SemIR::FunctionId function_id;
    clang::CXXMethodDecl* method_decl;
  };
  llvm::SmallVector<PendingVirtualFunction> pending_virtual_functions;

  if (class_info.vtable_decl_id.has_value()) {
    auto vtable_inst_block = context_->inst_blocks().Get(
        context_->vtables()
            .Get(context_->insts()
                     .GetAs<SemIR::VtableDecl>(class_info.vtable_decl_id)
                     .vtable_id)
            .virtual_functions_id);
    for (auto vtable_entry_id : vtable_inst_block) {
      if (!vtable_entry_id.has_value()) {
        continue;
      }

      const auto callee_function =
          GetCalleeAsFunction(context_->sem_ir(), vtable_entry_id);
      const SemIR::Function& function =
          context_->functions().Get(callee_function.function_id);

      // If this is a member of a base class, nothing to do here.
      if (function.parent_scope_id != class_info.scope_id) {
        continue;
      }
      auto* method_decl =
          cast_or_null<clang::CXXMethodDecl>(ExportVirtualFunctionDeclToCpp(
              *context_, SemIR::LocId(vtable_entry_id), class_decl,
              callee_function.function_id));
      if (!method_decl) {
        continue;
      }
      context_->clang_sema().AddOverriddenMethods(class_decl, method_decl);
      context_->clang_decls().Add(
          {.key = SemIR::ClangDeclKey::ForFunctionDecl(
               method_decl,
               MakeVirtualFunctionSignature(*context_, method_decl)),
           .inst_id = function.first_decl_id()});
      // An abstract function has no definition, so it doesn't need a thunk.
      if (function.virtual_modifier ==
          SemIR::Function::VirtualModifier::Abstract) {
        continue;
      }
      pending_virtual_functions.push_back(
          {.loc_id = SemIR::LocId(vtable_entry_id),
           .function_id = callee_function.function_id,
           .method_decl = method_decl});
    }
  }
  class_decl->completeDefinition();

  // Now the class is complete, we can define the virtual function thunks.
  for (auto virtual_fn : pending_virtual_functions) {
    DefineExportedVirtualFunction(*context_, virtual_fn.loc_id,
                                  virtual_fn.function_id,
                                  virtual_fn.method_decl);
  }
}

auto CarbonExternalASTSource::layoutRecordType(
    const clang::RecordDecl* record_decl, uint64_t& size, uint64_t& alignment,
    llvm::DenseMap<const clang::FieldDecl*, uint64_t>& field_offsets,
    llvm::DenseMap<const clang::CXXRecordDecl*, clang::CharUnits>& base_offsets,
    llvm::DenseMap<const clang::CXXRecordDecl*, clang::CharUnits>&
        vbase_offsets) -> bool {
  auto carbon_class_info =
      SemIR::GetAsCarbonOwnedClass(context_->sem_ir(), record_decl);
  if (!carbon_class_info) {
    return false;
  }
  auto& [class_type_id, class_type] = *carbon_class_info;

  // Clang should not have asked for the layout of an incomplete type, but check
  // now to be sure, and to generate a specific definition if needed.
  // TODO: Add a test for layout of a specific class once they're supported in
  // general.
  CompleteTypeOrCheckFail(*context_, class_type_id);

  auto& class_info = context_->classes().Get(class_type.class_id);
  ExportAllFieldsToCpp(*context_, class_info);

  return ReadOnlyASTSource::layoutRecordType(
      record_decl, size, alignment, field_offsets, base_offsets, vbase_offsets);
}

// Parses a sequence of top-level declarations and forms a corresponding
// representation in the Clang AST. Unlike clang::ParseAST, does not finish the
// translation unit when EOF is reached.
static auto ParseTopLevelDecls(clang::Parser& parser,
                               clang::ASTConsumer& consumer) -> void {
  // Don't allow C++20 module declarations in inline Cpp code fragments.
  auto module_import_state = clang::Sema::ModuleImportState::NotACXX20Module;

  // Parse top-level declarations until we see EOF. Do not parse EOF, as that
  // will cause the parser to end the translation unit prematurely.
  while (parser.getCurToken().isNot(clang::tok::eof)) {
    clang::Parser::DeclGroupPtrTy decl_group;
    bool eof = parser.ParseTopLevelDecl(decl_group, module_import_state);
    CARBON_CHECK(!eof, "Should not parse decls at EOF");
    if (decl_group && !consumer.HandleTopLevelDecl(decl_group.get())) {
      // If the consumer rejects the declaration, bail out of parsing.
      //
      // TODO: In this case, we shouldn't parse any more declarations even in
      // separate inline C++ fragments. But our current AST consumer only ever
      // returns true.
      break;
    }
  }
}

// Generate a Clang module corresponding to the current Carbon file.
static auto CreateModuleForFile(CppDomain& domain, const SemIR::File& file)
    -> clang::Module* {
  // TODO: Consider creating a parent module to hold all Carbon modules.
  // Consider naming the module after the package and library rather than using
  // the filename.
  auto& module_map = domain.clang_instance()
                         .getPreprocessor()
                         .getHeaderSearchInfo()
                         .getModuleMap();
  return module_map.createModule(file.filename(), /*Parent=*/nullptr,
                                 /*IsFramework=*/false, /*IsExplicit=*/true);
}

// Parse the tokens that have been injected into the preprocessor in the given
// context.
static auto ParseInjectedTokens(CppContext& cpp_context) -> void {
  clang::Sema& sema = cpp_context.sema();
  clang::Parser& parser = cpp_context.parser();
  CARBON_CHECK(parser.getCurToken().is(clang::tok::eof));
  parser.ConsumeToken();
  ParseTopLevelDecls(parser, sema.getASTConsumer());
}

// Injects the C++ code in `buffer` into the Clang preprocessor. Returns the
// file ID of the injected buffer.
static auto InjectBuffer(CppContext& cpp_context, llvm::StringRef contents,
                         llvm::StringRef name, clang::SourceLocation import_loc)
    -> clang::FileID {
  auto buffer = llvm::MemoryBuffer::getMemBufferCopy(contents, name);

  clang::Preprocessor& preprocessor = cpp_context.sema().getPreprocessor();
  clang::FileID file_id =
      preprocessor.getSourceManager().createFileID(std::move(buffer));
  if (preprocessor.EnterSourceFile(file_id, nullptr, import_loc)) {
    CARBON_FATAL("Failed to enter buffer");
  }

  return file_id;
}

// Injects code to import the given set of headers into Clang and parses it as
// top-level declarations.
static auto ParseImports(Context& context,
                         llvm::ArrayRef<Parse::Tree::PackagingNames> imports)
    -> void {
  auto* cpp_context = context.cpp_context();
  CARBON_CHECK(cpp_context);

  // Inject the imports-as-#includes buffer.
  auto file_id = InjectBuffer(*cpp_context,
                              GenerateCppIncludesHeaderCode(context, imports),
                              "<shared cpp imports>", clang::SourceLocation());

  // Enter the module for this file.
  auto& preprocessor = cpp_context->sema().getPreprocessor();
  auto* mod = CreateModuleForFile(cpp_context->domain(), context.sem_ir());
  auto loc = preprocessor.getSourceManager().getLocForStartOfFile(file_id);
  preprocessor.EnterSubmodule(mod, loc, /*ForPragma=*/false);
  preprocessor.EnterAnnotationToken(loc, clang::tok::annot_module_begin, mod);

  ParseInjectedTokens(*cpp_context);
}

// Leave the current Clang module.
static auto LeaveModule(Context& context, clang::SourceLocation loc) -> void {
  CARBON_CHECK(loc.isValid());

  auto* cpp_context = context.cpp_context();
  CARBON_CHECK(cpp_context);

  auto& preprocessor = cpp_context->sema().getPreprocessor();
  auto* mod = preprocessor.LeaveSubmodule(/*ForPragma=*/false);
  CARBON_CHECK(mod);

  // We *should* only need to enter one annotation token, but Clang has some
  // error recovery where Sema enters and never leaves an additional module if
  // it sees a `module;` directive in the source. So recover from this by
  // leaving modules until we find the preprocessor's module.
  while (true) {
    auto* sema_mod = cpp_context->sema().getCurrentModule();
    CARBON_CHECK(sema_mod, "Sema prematurely exited Carbon module");

    preprocessor.EnterAnnotationToken(loc, clang::tok::annot_module_end,
                                      sema_mod);
    ParseInjectedTokens(*cpp_context);
    if (sema_mod == mod) {
      break;
    }
  }
}

namespace {

// An action and a set of registered Clang callbacks used to generate an AST
// from a set of Cpp imports.
class GenerateASTAction : public clang::ASTFrontendAction {
 public:
  explicit GenerateASTAction(llvm::StringRef filename,
                             llvm::LLVMContext* llvm_context)
      : filename_(filename), llvm_context_(llvm_context) {}

  auto code_generator() const -> clang::CodeGenerator* {
    return code_generator_;
  }

  auto TakeParser() -> std::unique_ptr<clang::Parser> {
    return std::move(parser_);
  }

 protected:
  auto CreateASTConsumer(clang::CompilerInstance& clang_instance,
                         llvm::StringRef /*file*/)
      -> std::unique_ptr<clang::ASTConsumer> override {
    if (!llvm_context_) {
      return std::make_unique<clang::ASTConsumer>();
    }
    auto code_generator =
        std::unique_ptr<clang::CodeGenerator>(clang::CreateLLVMCodeGen(
            clang_instance.getDiagnostics(), filename_,
            clang_instance.getVirtualFileSystemPtr(),
            clang_instance.getHeaderSearchOpts(),
            clang_instance.getPreprocessorOpts(),
            clang_instance.getCodeGenOpts(), *llvm_context_));
    code_generator_ = code_generator.get();
    return code_generator;
  }

  auto BeginSourceFileAction(clang::CompilerInstance& /*clang_instance*/)
      -> bool override {
    return true;
  }

  // Parse the imports and inline C++ fragments. This is notionally very similar
  // to `clang::ParseAST`, which `ASTFrontendAction::ExecuteAction` calls, but
  // this version doesn't parse C++20 modules and stops just before reaching the
  // end of the translation unit.
  auto ExecuteAction() -> void override {
    clang::CompilerInstance& clang_instance = getCompilerInstance();
    clang_instance.createSema(getTranslationUnitKind(),
                              /*CompletionConsumer=*/nullptr);

    parser_ = std::make_unique<clang::Parser>(clang_instance.getPreprocessor(),
                                              clang_instance.getSema(),
                                              /*SkipFunctionBodies=*/false);

    clang_instance.getPreprocessor().enableIncrementalProcessing();
    clang_instance.getPreprocessor().EnterMainSourceFile();
    parser_->Initialize();

    if (auto* source = clang_instance.getASTContext().getExternalSource()) {
      source->StartTranslationUnit(&clang_instance.getASTConsumer());
    }

    clang_instance.getSema().ActOnStartOfTranslationUnit();

    ParseTopLevelDecls(*parser_, clang_instance.getASTConsumer());
  }

 private:
  std::string filename_;
  llvm::LLVMContext* llvm_context_;
  clang::CodeGenerator* code_generator_ = nullptr;
  std::unique_ptr<clang::Parser> parser_;
};

}  // namespace

// Initializes the Clang state by building a new compiler invocation,
// creating a diagnostics engine, and parsing a dummy main file containing a
// semicolon. Returns the initialized state, or null on failure.
auto InitializeCppDomain(
    Diagnostics::Consumer& consumer, llvm::StringRef filename,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
    llvm::LLVMContext* llvm_context,
    std::shared_ptr<clang::CompilerInvocation> base_invocation)
    -> std::unique_ptr<CppDomain> {
  std::shared_ptr<clang::CompilerInstance> clang_instance;
  llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> diags;

  // Build a new invocation.
  auto invocation =
      std::make_shared<ShallowCopyCompilerInvocation>(*base_invocation);

  // Ask Clang to not leak memory.
  invocation->getFrontendOpts().DisableFree = false;

  // Build a diagnostics engine.
  diags = clang::CompilerInstance::createDiagnostics(
      *fs, invocation->getDiagnosticOpts(),
      MakeDiagnosticConsumer(consumer, invocation).release(),
      /*ShouldOwnClient=*/true);

  // Ensure any diagnostics emitted in this function are flushed before we
  // return.
  auto on_exit =
      llvm::scope_exit([&]() { FlushDiagnosticConsumer(*diags->getClient()); });

  // Extract the input from the frontend invocation and make sure it makes
  // sense.
  const auto& inputs = invocation->getFrontendOpts().Inputs;
  CARBON_CHECK(inputs.size() == 1 &&
               inputs[0].getKind().getLanguage() == clang::Language::CXX &&
               inputs[0].getKind().getFormat() == clang::InputKind::Source);
  llvm::StringRef file_name = inputs[0].getFile();

  // Remap the input file to a dummy buffer containing a semicolon to start
  // with an empty AST. Clang requires at least one token in the main file
  // to avoid assertion failures if it later encounters module declarations.
  // TODO: See if we can fix this by injecting code into the main file rather
  // than entering nested buffers.
  auto empty_buffer = llvm::MemoryBuffer::getMemBuffer(";");
  invocation->getPreprocessorOpts().addRemappedFile(file_name,
                                                    empty_buffer.release());

  clang_instance = std::make_shared<clang::CompilerInstance>(invocation);

  clang_instance->setDiagnostics(diags);
  clang_instance->setVirtualFileSystem(fs);
  clang_instance->createFileManager();
  clang_instance->createSourceManager();
  if (!clang_instance->createTarget()) {
    return nullptr;
  }

  GenerateASTAction action(filename, llvm_context);
  if (!action.BeginSourceFile(*clang_instance, inputs[0])) {
    return nullptr;
  }

  auto& ast = clang_instance->getASTContext();

  // Create an AST reader before we set up our own source. Clang does this
  // automatically later if we don't do it now, and will overwrite our external
  // source with its own when it does so.
  clang_instance->createASTReader();

  // Always build a multiplex source, even if there's only one child
  // source. During lowering, the `CarbonExternalASTSource` can no longer be
  // used (because it uses `Check::Context`), so a `ReadOnlyASTSource` is
  // installed instead. However, clang internally keeps pointers to the
  // top-level `ExternalASTSource` installed via `setExternalSource`, and
  // those pointers aren't updated if `setExternalSource` is called again. By
  // using `MultiplexExternalSemaSource`, we can keep the top-level
  // `ExternalASTSource` pointer the same, and only update its children.
  auto multiplex_source_ref_cnt_ptr =
      llvm::makeIntrusiveRefCnt<clang::MultiplexExternalSemaSource>();
  auto* multiplex_source = cast<clang::MultiplexExternalSemaSource>(
      multiplex_source_ref_cnt_ptr.get());
  if (auto* existing_source = llvm::cast_or_null<clang::ExternalSemaSource>(
          ast.getExternalSource())) {
    multiplex_source->AddSource(existing_source);
  }
  ast.setExternalSource(std::move(multiplex_source_ref_cnt_ptr));

  if (llvm::Error error = action.Execute()) {
    // `Execute` currently never fails, but its contract allows it to.
    CARBON_FATAL("Failed to execute clang action: {0}",
                 llvm::toString(std::move(error)));
  }

  auto parser = action.TakeParser();
  CARBON_CHECK(parser);

  return std::make_unique<CppDomain>(std::move(clang_instance),
                                     std::move(parser), action.code_generator(),
                                     llvm_context);
}

auto GenerateAst(Context& context,
                 llvm::ArrayRef<Parse::Tree::PackagingNames> imports,
                 CppDomain& domain) -> bool {
  CARBON_CHECK(!context.cpp_context());
  CARBON_CHECK(!context.sem_ir().cpp_file());

  // Register an annotation scope to flush any Clang diagnostics when we
  // return. This ensures C++ diagnostics get flushed before `diags` is
  // destroyed, and that diagnostics created here don't interleave with later
  // Carbon diagnostics.
  Diagnostics::AnnotationScope annotate_diagnostics(&context.emitter(),
                                                    [](auto& /*builder*/) {});

  auto clang_instance = domain.clang_instance_ptr();

  // Set up CppFile for the current SemIR::File.
  auto cpp_file =
      std::make_unique<SemIR::CppFile>(clang_instance, domain.llvm_context());
  if (domain.code_generator()) {
    cpp_file->SetCodeGenerator(domain.code_generator());
  }
  context.sem_ir().set_cpp_file(std::move(cpp_file));

  // Set up CppContext for the current Context.
  context.set_cpp_context(std::make_unique<CppContext>(
      domain, MakeContextDiagnosticListener(
                  *clang_instance->getDiagnostics().getClient(), context)));

  // The AST context is now available, so the mangle context (used to compute
  // stable identities for imported C++ types) can be created.
  context.sem_ir().cpp_file()->CreateMangleContext();

  // Add an external source referring to this context.
  auto* multiplex_source = cast<clang::MultiplexExternalSemaSource>(
      context.ast_context().getExternalSource());
  auto ast_source =
      llvm::makeIntrusiveRefCnt<CarbonExternalASTSource>(&context);
  multiplex_source->AddSource(ast_source);

  // Map the package scope to the Carbon namespace.
  ast_source->BuildCarbonNamespace();

  // Parse the imports-as-#includes buffer.
  ParseImports(context, imports);
  return true;
}

auto InjectAstFromInlineCode(Context& context, SemIR::LocId loc_id,
                             llvm::StringRef source_code) -> void {
  auto* cpp_context = context.cpp_context();
  CARBON_CHECK(cpp_context);

  RawStringOstream code_stream;
  AppendInlineCode(context, code_stream,
                   context.parse_tree().node_token(loc_id.node_id()),
                   source_code);

  // Clang will have generated a suitable error if this fails. There's nothing
  // more to do here.
  InjectBuffer(*cpp_context, code_stream.TakeStr(), "<inline c++>",
               GetCppLocation(context, loc_id));
  ParseInjectedTokens(*cpp_context);
}

auto FinishAst(Context& context) -> void {
  if (!context.cpp_context()) {
    return;
  }

  // Leave the module we entered to encapsulate the contents of this Carbon
  // file.
  auto end_loc_id =
      SemIR::LocId(*(context.sem_ir().parse_tree().postorder().end() - 1));
  // Shuffle the end of file location back by one character to work around a
  // Clang bug: if we give Clang the end-of-file location, it will replace the
  // location with the include location without checking whether the file was
  // actually included, and then crash because it picked an invalid location!
  // There is always at least one token in a file with a `Cpp` import, so this
  // location adjustment is safe.
  LeaveModule(context,
              GetCppLocation(context, end_loc_id).getLocWithOffset(-1));

  // Finalize the per-Context AST fragment. The final ActOnEndOfTranslationUnit
  // call for the CppDomain is performed in FinalizeCppDomain once all files
  // sharing the domain have been checked.
  context.cpp_context()->sema().ActOnEndOfTranslationUnitFragment(
      clang::TUFragmentKind::Normal);
  FlushDiagnosticConsumer(
      *context.cpp_context()->sema().getDiagnostics().getClient());
  context.emitter().Flush();

  // Remove the `CarbonExternalASTSource` installed in `GenerateAst` and
  // replace it with a `ReadOnlyASTSource`. This is necessary because
  // the source may be accessed later during lowering, but the
  // `CarbonExternalASTSource` has a pointer to `Check::Context` that
  // will not remain valid.
  auto* multiplex_source = cast<clang::MultiplexExternalSemaSource>(
      context.ast_context().getExternalSource());
  multiplex_source->EraseIf([](const auto& src) {
    return llvm::isa<CarbonExternalASTSource>(src.get());
  });
  multiplex_source->AddSource(
      llvm::makeIntrusiveRefCnt<SemIR::ReadOnlyASTSource>(context.sem_ir()));

  // We don't call FrontendAction::EndSourceFile, because that destroys the AST.
  context.set_cpp_context(nullptr);
}

auto FinalizeCppDomain(CppDomain& domain) -> void {
  if (domain.clang_instance_ptr()) {
    domain.clang_instance().getSema().ActOnEndOfTranslationUnit();
    FlushDiagnosticConsumer(
        *domain.clang_instance().getDiagnostics().getClient());
  }
}

}  // namespace Carbon::Check
