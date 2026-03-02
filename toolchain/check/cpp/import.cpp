// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/import.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/UnresolvedSet.h"
#include "clang/AST/VTableBuilder.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "common/check.h"
#include "common/ostream.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/base/int.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/base/value_ids.h"
#include "toolchain/check/call.h"
#include "toolchain/check/class.h"
#include "toolchain/check/context.h"
#include "toolchain/check/control_flow.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/core_identifier.h"
#include "toolchain/check/cpp/access.h"
#include "toolchain/check/cpp/custom_type_mapping.h"
#include "toolchain/check/cpp/generate_ast.h"
#include "toolchain/check/cpp/location.h"
#include "toolchain/check/cpp/macros.h"
#include "toolchain/check/cpp/thunk.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/function.h"
#include "toolchain/check/import.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/literal.h"
#include "toolchain/check/member_access.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/operator.h"
#include "toolchain/check/pattern.h"
#include "toolchain/check/pattern_match.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/check/unused.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/clang_decl.h"
#include "toolchain/sem_ir/class.h"
#include "toolchain/sem_ir/cpp_file.h"
#include "toolchain/sem_ir/cpp_overload_set.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/pattern.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Adds the name to the scope with the given `access_kind` and `inst_id`.
// `inst_id` must have a value.
static auto AddNameToScope(Context& context, SemIR::NameScopeId scope_id,
                           SemIR::NameId name_id, SemIR::AccessKind access_kind,
                           SemIR::InstId inst_id) -> void {
  CARBON_CHECK(inst_id.has_value());
  context.name_scopes().Get(scope_id).AddRequired(
      {.name_id = name_id,
       .result = SemIR::ScopeLookupResult::MakeFound(inst_id, access_kind)});
}

// Maps a Clang name to a Carbon `NameId`.
static auto AddIdentifierName(Context& context, llvm::StringRef name)
    -> SemIR::NameId {
  return SemIR::NameId::ForIdentifier(context.identifiers().Add(name));
}

// Adds the given source location and an `ImportIRInst` referring to it in
// `ImportIRId::Cpp`.
static auto AddImportIRInst(SemIR::File& file,
                            clang::SourceLocation clang_source_loc)
    -> SemIR::ImportIRInstId {
  SemIR::ClangSourceLocId clang_source_loc_id =
      file.clang_source_locs().Add(clang_source_loc);
  return file.import_ir_insts().Add(SemIR::ImportIRInst(clang_source_loc_id));
}

// Adds a namespace for the `Cpp` import and returns its `NameScopeId`.
static auto AddNamespace(Context& context, PackageNameId cpp_package_id,
                         llvm::ArrayRef<Parse::Tree::PackagingNames> imports)
    -> SemIR::NameScopeId {
  return AddImportNamespaceToScope(
             context,
             GetSingletonType(context, SemIR::NamespaceType::TypeInstId),
             SemIR::NameId::ForPackageName(cpp_package_id),
             SemIR::NameScopeId::Package,
             /*diagnose_duplicate_namespace=*/false,
             [&]() {
               return AddInst<SemIR::ImportCppDecl>(
                   context,
                   context.parse_tree().As<Parse::ImportDeclId>(
                       imports.front().node_id),
                   {});
             })
      .add_result.name_scope_id;
}

auto ImportCpp(Context& context,
               llvm::ArrayRef<Parse::Tree::PackagingNames> imports,
               llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
               llvm::LLVMContext* llvm_context,
               std::shared_ptr<clang::CompilerInvocation> invocation) -> void {
  if (imports.empty()) {
    // TODO: Consider always having a (non-null) AST even if there are no Cpp
    // imports.
    return;
  }

  PackageNameId package_id = imports.front().package_id;
  CARBON_CHECK(
      llvm::all_of(imports, [&](const Parse::Tree::PackagingNames& import) {
        return import.package_id == package_id;
      }));

  auto name_scope_id = AddNamespace(context, package_id, imports);
  SemIR::NameScope& name_scope = context.name_scopes().Get(name_scope_id);
  name_scope.set_is_closed_import(true);

  if (GenerateAst(context, imports, fs, llvm_context, std::move(invocation))) {
    name_scope.set_clang_decl_context_id(context.clang_decls().Add(
        {.key = SemIR::ClangDeclKey(
             context.ast_context().getTranslationUnitDecl()),
         .inst_id = name_scope.inst_id()}));
  } else {
    name_scope.set_has_error();
  }
}

// Returns the Clang `DeclContext` for the given name scope. Return the
// translation unit decl if no scope is provided.
static auto GetDeclContext(Context& context, SemIR::NameScopeId scope_id)
    -> clang::DeclContext* {
  if (!scope_id.has_value()) {
    return context.ast_context().getTranslationUnitDecl();
  }
  auto scope_clang_decl_context_id =
      context.name_scopes().Get(scope_id).clang_decl_context_id();
  return dyn_cast<clang::DeclContext>(
      context.clang_decls().Get(scope_clang_decl_context_id).key.decl);
}

// Returns true if the given Clang declaration is the implicit injected class
// name within the class.
static auto IsDeclInjectedClassName(Context& context,
                                    SemIR::NameScopeId scope_id,
                                    SemIR::NameId name_id,
                                    const clang::NamedDecl* named_decl)
    -> bool {
  if (!named_decl->isImplicit()) {
    return false;
  }

  const auto* record_decl = dyn_cast<clang::CXXRecordDecl>(named_decl);
  if (!record_decl) {
    return false;
  }

  const SemIR::ClangDecl& clang_decl = context.clang_decls().Get(
      context.name_scopes().Get(scope_id).clang_decl_context_id());
  const auto* scope_record_decl =
      cast<clang::CXXRecordDecl>(clang_decl.key.decl);

  const clang::ASTContext& ast_context = context.ast_context();
  CARBON_CHECK(ast_context.getCanonicalTagType(scope_record_decl) ==
               ast_context.getCanonicalTagType(record_decl));

  auto class_decl = context.insts().GetAs<SemIR::ClassDecl>(clang_decl.inst_id);
  CARBON_CHECK(name_id == context.classes().Get(class_decl.class_id).name_id);
  return true;
}

// Performs a qualified name lookup of the identifier in the given scope.
// Returns the lookup result if lookup was successful.
static auto ClangLookupName(Context& context, SemIR::NameScopeId scope_id,
                            clang::IdentifierInfo* identifier_name)
    -> std::optional<clang::LookupResult> {
  CARBON_CHECK(identifier_name, "Identifier name is empty");
  clang::Sema& sema = context.clang_sema();

  // TODO: Map the LocId of the lookup to a clang SourceLocation and provide it
  // here so that clang's diagnostics can point into the carbon code that uses
  // the name.
  clang::LookupResult lookup(
      sema,
      clang::DeclarationNameInfo(clang::DeclarationName(identifier_name),
                                 clang::SourceLocation()),
      clang::Sema::LookupNameKind::LookupOrdinaryName);

  bool found =
      sema.LookupQualifiedName(lookup, GetDeclContext(context, scope_id));

  if (!found) {
    return std::nullopt;
  }

  return lookup;
}

// Returns whether `decl` already mapped to an instruction.
static auto IsClangDeclImported(Context& context, SemIR::ClangDeclKey key)
    -> bool {
  return context.clang_decls().Lookup(key).has_value();
}

// If `decl` already mapped to an instruction, returns that instruction.
// Otherwise returns `None`.
static auto LookupClangDeclInstId(Context& context, SemIR::ClangDeclKey key)
    -> SemIR::InstId {
  const auto& clang_decls = context.clang_decls();
  if (auto context_clang_decl_id = clang_decls.Lookup(key);
      context_clang_decl_id.has_value()) {
    return clang_decls.Get(context_clang_decl_id).inst_id;
  }
  return SemIR::InstId::None;
}

// Returns the parent of the given declaration. Skips declaration types we
// ignore.
static auto GetParentDecl(clang::Decl* clang_decl) -> clang::Decl* {
  auto* parent_dc = clang_decl->getDeclContext();
  while (!parent_dc->isLookupContext()) {
    parent_dc = parent_dc->getParent();
  }
  return cast<clang::Decl>(parent_dc);
}

// Returns the given declaration's parent scope. Assumes the parent declaration
// was already imported.
static auto GetParentNameScopeId(Context& context, clang::Decl* clang_decl)
    -> SemIR::NameScopeId {
  auto* parent_decl = GetParentDecl(clang_decl);

  if (auto* tag_decl = dyn_cast<clang::TagDecl>(parent_decl)) {
    auto class_inst_id =
        LookupClangDeclInstId(context, SemIR::ClangDeclKey(tag_decl));
    CARBON_CHECK(class_inst_id.has_value());
    auto class_inst = context.insts().Get(class_inst_id);
    auto class_id = SemIR::ClassId::None;
    if (auto class_decl = class_inst.TryAs<SemIR::ClassDecl>()) {
      // Common case: the tag was imported as a new Carbon class.
      class_id = class_decl->class_id;
    } else {
      // Rare case: the tag was imported as an existing Carbon class. This
      // happens for C++ classes that get mapped to Carbon prelude types, such
      // as `std::string_view`.
      // TODO: In this case, should we import the C++ class declaration and use
      // it as the parent, rather than using the existing Carbon class?
      class_id = class_inst.As<SemIR::ClassType>().class_id;
    }
    return context.classes().Get(class_id).scope_id;
  }

  if (isa<clang::NamespaceDecl, clang::TranslationUnitDecl>(parent_decl)) {
    auto namespace_inst_id = LookupClangDeclInstId(
        context, SemIR::ClangDeclKey::ForNonFunctionDecl(parent_decl));
    CARBON_CHECK(namespace_inst_id.has_value());
    return context.insts()
        .GetAs<SemIR::Namespace>(namespace_inst_id)
        .name_scope_id;
  }

  CARBON_FATAL("Unexpected kind of parent {0}", parent_decl->getDeclKindName());
}

// Imports a namespace declaration from Clang to Carbon. If successful, returns
// the new Carbon namespace declaration `InstId`. If the declaration was already
// imported, returns the mapped instruction.
static auto ImportNamespaceDecl(Context& context,
                                clang::NamespaceDecl* clang_decl)
    -> SemIR::InstId {
  auto key = SemIR::ClangDeclKey(clang_decl);
  // Check if the declaration is already mapped.
  if (SemIR::InstId existing_inst_id = LookupClangDeclInstId(context, key);
      existing_inst_id.has_value()) {
    return existing_inst_id;
  }
  auto result = AddImportNamespace(
      context, GetSingletonType(context, SemIR::NamespaceType::TypeInstId),
      AddIdentifierName(context, clang_decl->getName()),
      GetParentNameScopeId(context, clang_decl),
      /*import_id=*/SemIR::InstId::None);
  context.name_scopes()
      .Get(result.name_scope_id)
      .set_clang_decl_context_id(
          context.clang_decls().Add({.key = key, .inst_id = result.inst_id}));
  return result.inst_id;
}

// Creates a class declaration for the given class name in the given scope.
// Returns the `InstId` for the declaration.
static auto BuildClassDecl(Context& context,
                           SemIR::ImportIRInstId import_ir_inst_id,
                           SemIR::NameScopeId parent_scope_id,
                           SemIR::NameId name_id)
    -> std::tuple<SemIR::ClassId, SemIR::TypeInstId> {
  // Add the class declaration.
  auto class_decl = SemIR::ClassDecl{.type_id = SemIR::TypeType::TypeId,
                                     .class_id = SemIR::ClassId::None,
                                     .decl_block_id = SemIR::InstBlockId::None};
  auto class_decl_id = AddPlaceholderImportedInstInNoBlock(
      context,
      MakeImportedLocIdAndInst(context, import_ir_inst_id, class_decl));

  SemIR::Class class_info = {
      {.name_id = name_id,
       .parent_scope_id = parent_scope_id,
       .generic_id = SemIR::GenericId::None,
       .first_param_node_id = Parse::NodeId::None,
       .last_param_node_id = Parse::NodeId::None,
       .pattern_block_id = SemIR::InstBlockId::None,
       .implicit_param_patterns_id = SemIR::InstBlockId::None,
       .param_patterns_id = SemIR::InstBlockId::None,
       .is_extern = false,
       .extern_library_id = SemIR::LibraryNameId::None,
       .non_owning_decl_id = SemIR::InstId::None,
       .first_owning_decl_id = class_decl_id},
      {// `.self_type_id` depends on the ClassType, so is set below.
       .self_type_id = SemIR::TypeId::None,
       // TODO: Support Dynamic classes.
       // TODO: Support Final classes.
       .inheritance_kind = SemIR::Class::Base}};

  class_decl.class_id = context.classes().Add(class_info);

  // Write the class ID into the ClassDecl.
  ReplaceInstBeforeConstantUse(context, class_decl_id, class_decl);

  SetClassSelfType(context, class_decl.class_id);

  return {class_decl.class_id, context.types().GetAsTypeInstId(class_decl_id)};
}

// Imports a tag declaration from Clang to Carbon. This covers classes (which
// includes structs and unions) as well as enums. If successful, returns the new
// Carbon class declaration `InstId`.
static auto ImportTagDecl(Context& context, clang::TagDecl* clang_decl)
    -> SemIR::InstId {
  auto import_ir_inst_id =
      AddImportIRInst(context.sem_ir(), clang_decl->getLocation());

  auto [class_id, class_inst_id] = BuildClassDecl(
      context, import_ir_inst_id, GetParentNameScopeId(context, clang_decl),
      AddIdentifierName(context, clang_decl->getName()));

  // TODO: The caller does the same lookup. Avoid doing it twice.
  auto key = SemIR::ClangDeclKey(clang_decl);
  auto clang_decl_id =
      context.clang_decls().Add({.key = key, .inst_id = class_inst_id});

  // Name lookup into the Carbon class looks in the C++ class definition.
  auto& class_info = context.classes().Get(class_id);
  class_info.scope_id = context.name_scopes().Add(
      class_inst_id, SemIR::NameId::None, class_info.parent_scope_id);
  context.name_scopes()
      .Get(class_info.scope_id)
      .set_clang_decl_context_id(clang_decl_id);

  return class_inst_id;
}

// Determines the Carbon inheritance kind to use for a C++ class definition.
static auto GetInheritanceKind(clang::CXXRecordDecl* class_def)
    -> SemIR::Class::InheritanceKind {
  if (class_def->isUnion()) {
    // Treat all unions as final classes to match their C++ semantics. While we
    // could support this, the author of a C++ union has no way to mark their
    // type as `final` to prevent it, and so we assume the intent was to
    // disallow inheritance.
    return SemIR::Class::Final;
  }

  if (class_def->hasAttr<clang::FinalAttr>()) {
    // The class is final in C++; don't allow Carbon types to derive from it.
    // Note that such a type might also be abstract in C++; we treat final as
    // taking precedence.
    //
    // We could also treat classes with a final destructor as being final, as
    // Clang does when determining whether a class is "effectively final", but
    // to keep our rules simpler we do not.
    return SemIR::Class::Final;
  }

  if (class_def->getNumVBases()) {
    // TODO: We treat classes with virtual bases as final for now. We use the
    // layout of the class including its virtual bases as its Carbon type
    // layout, so we wouldn't behave correctly if we derived from it.
    return SemIR::Class::Final;
  }

  if (class_def->isAbstract()) {
    // If the class has any abstract members, it's abstract.
    return SemIR::Class::Abstract;
  }

  // Allow inheritance from any other C++ class type.
  return SemIR::Class::Base;
}

// Checks that the specified finished class definition is valid and builds and
// returns a corresponding complete type witness instruction.
static auto ImportClassObjectRepr(Context& context, SemIR::ClassId class_id,
                                  SemIR::ImportIRInstId import_ir_inst_id,
                                  SemIR::TypeInstId class_type_inst_id,
                                  const clang::CXXRecordDecl* clang_def)
    -> SemIR::TypeInstId {
  if (clang_def->isInvalidDecl()) {
    // Clang already diagnosed this error.
    return SemIR::ErrorInst::TypeInstId;
  }

  // For now, if the class is empty and an aggregate, produce an empty struct as
  // the object representation. This allows our tests to continue to pass while
  // we don't properly support initializing imported C++ classes. We only do
  // this for aggregates so that non-aggregate classes are not incorrectly
  // initializable from `{}`.
  // TODO: Remove this.
  if (clang_def->isEmpty() && !clang_def->getNumBases() &&
      clang_def->isAggregate()) {
    return context.types().GetAsTypeInstId(AddInst(
        context,
        MakeImportedLocIdAndInst(
            context, import_ir_inst_id,
            SemIR::StructType{.type_id = SemIR::TypeType::TypeId,
                              .fields_id = SemIR::StructTypeFieldsId::Empty})));
  }

  const auto& clang_layout =
      context.ast_context().getASTRecordLayout(clang_def);

  llvm::SmallVector<uint64_t> layout;
  llvm::SmallVector<SemIR::StructTypeField> fields;

  static_assert(SemIR::CustomLayoutId::SizeIndex == 0);
  layout.push_back(clang_layout.getSize().getQuantity());

  static_assert(SemIR::CustomLayoutId::AlignIndex == 1);
  layout.push_back(clang_layout.getAlignment().getQuantity());

  static_assert(SemIR::CustomLayoutId::FirstFieldIndex == 2);

  // TODO: Import vptr(s).

  // The kind of base class we've picked so far. These are ordered in increasing
  // preference order.
  enum class BaseKind {
    None,
    Empty,
    NonEmpty,
    Polymorphic,
  };
  BaseKind base_kind = BaseKind::None;

  // Import bases.
  for (const auto& base : clang_def->bases()) {
    if (base.isVirtual()) {
      // If the base is virtual, skip it from the layout. We don't know where it
      // will actually appear within the complete object layout, as a pointer to
      // this class might point to a derived type that puts the vbase in a
      // different place.
      // TODO: Track that the virtual base existed. Support derived-to-vbase
      // conversions by generating a clang AST fragment.
      continue;
    }

    auto [base_type_inst_id, base_type_id] =
        ImportCppType(context, import_ir_inst_id, base.getType());
    if (!base_type_id.has_value()) {
      // TODO: If the base class's type can't be mapped, skip it.
      continue;
    }

    auto base_decl_id = AddInst(
        context,
        MakeImportedLocIdAndInst(
            context, import_ir_inst_id,
            SemIR::BaseDecl{.type_id = GetUnboundElementType(
                                context, class_type_inst_id, base_type_inst_id),
                            .base_type_inst_id = base_type_inst_id,
                            .index = SemIR::ElementIndex(fields.size())}));

    auto* base_class = base.getType()->getAsCXXRecordDecl();
    CARBON_CHECK(base_class, "Base class {0} is not a class",
                 base.getType().getAsString());

    // If there's a unique "best" base class, treat it as a Carbon base class
    // too.
    // TODO: Improve handling for the case where the class has multiple base
    // classes.
    BaseKind kind = base_class->isPolymorphic() ? BaseKind::Polymorphic
                    : base_class->isEmpty()     ? BaseKind::Empty
                                                : BaseKind::NonEmpty;
    auto& class_info = context.classes().Get(class_id);
    if (kind > base_kind) {
      // This base is better than the previous best.
      class_info.base_id = base_decl_id;
      base_kind = kind;
    } else if (kind == base_kind) {
      // Multiple base classes of this kind: no unique best.
      class_info.base_id = SemIR::InstId::None;
    }

    // TODO: If the base class has virtual bases, the size of the type that we
    // add to the layout here will be the full size of the class (including
    // virtual bases), whereas the size actually occupied by this base class is
    // only the nvsize (excluding virtual bases).
    auto base_offset = base.isVirtual()
                           ? clang_layout.getVBaseClassOffset(base_class)
                           : clang_layout.getBaseClassOffset(base_class);
    layout.push_back(base_offset.getQuantity());
    fields.push_back(
        {.name_id = SemIR::NameId::Base, .type_inst_id = base_type_inst_id});
  }

  // Import fields.
  for (auto* decl : clang_def->decls()) {
    auto* field = dyn_cast<clang::FieldDecl>(decl);

    // Track the chain of fields from the class to this field. This chain is
    // only one element long unless the field is a member of an anonymous struct
    // or union.
    clang::NamedDecl* single_field_chain[1] = {field};
    llvm::ArrayRef<clang::NamedDecl*> chain = single_field_chain;

    // If this isn't a field, it might be an indirect field in an anonymous
    // struct or union.
    if (!field) {
      auto* indirect_field = dyn_cast<clang::IndirectFieldDecl>(decl);
      if (!indirect_field) {
        continue;
      }
      chain = indirect_field->chain();
      field = indirect_field->getAnonField();
    }

    if (field->isBitField()) {
      // TODO: Add a representation for named bitfield members.
      continue;
    }

    if (field->isAnonymousStructOrUnion()) {
      // Fields within an anonymous structure or union will be added via their
      // IndirectFieldDecls.
      continue;
    }

    auto field_name_id = AddIdentifierName(context, field->getName());
    auto [field_type_inst_id, field_type_id] =
        ImportCppType(context, import_ir_inst_id, field->getType());
    if (!field_type_inst_id.has_value()) {
      // TODO: For now, just skip over fields whose types we can't map.
      continue;
    }

    // Create a field now, as we know the index to use.
    // TODO: Consider doing this lazily instead.
    auto field_decl_id = AddInst(
        context, MakeImportedLocIdAndInst(
                     context, import_ir_inst_id,
                     SemIR::FieldDecl{
                         .type_id = GetUnboundElementType(
                             context, class_type_inst_id, field_type_inst_id),
                         .name_id = field_name_id,
                         .index = SemIR::ElementIndex(fields.size())}));
    // The imported SemIR::FieldDecl represents the original declaration `decl`,
    // which is either the field or the indirect field declaration.
    auto key = SemIR::ClangDeclKey::ForNonFunctionDecl(decl);
    context.clang_decls().Add({.key = key, .inst_id = field_decl_id});

    // Compute the offset to the field that appears directly in the class.
    uint64_t offset = clang_layout.getFieldOffset(
        cast<clang::FieldDecl>(chain.front())->getFieldIndex());

    // If this is an indirect field, walk the path and accumulate the offset to
    // the named field.
    for (auto* inner_decl : chain.drop_front()) {
      auto* inner_field = cast<clang::FieldDecl>(inner_decl);
      const auto& inner_layout =
          context.ast_context().getASTRecordLayout(inner_field->getParent());
      offset += inner_layout.getFieldOffset(inner_field->getFieldIndex());
    }

    layout.push_back(
        context.ast_context().toCharUnitsFromBits(offset).getQuantity());
    fields.push_back(
        {.name_id = field_name_id, .type_inst_id = field_type_inst_id});
  }

  // TODO: Add a field to prevent tail padding reuse if necessary.

  return AddTypeInst(context,
                     MakeImportedLocIdAndInst<SemIR::CustomLayoutType>(
                         context, import_ir_inst_id,
                         {.type_id = SemIR::TypeType::TypeId,
                          .fields_id = context.struct_type_fields().Add(fields),
                          .layout_id = context.custom_layouts().Add(layout)}));
}

// Creates a Carbon class definition based on the information in the given Clang
// class declaration, which is assumed to be for a class definition.
static auto BuildClassDefinition(Context& context,
                                 SemIR::ImportIRInstId import_ir_inst_id,
                                 SemIR::ClassId class_id,
                                 SemIR::TypeInstId class_inst_id,
                                 clang::CXXRecordDecl* clang_def) -> void {
  auto& class_info = context.classes().Get(class_id);
  CARBON_CHECK(!class_info.has_definition_started());
  class_info.definition_id = class_inst_id;

  context.inst_block_stack().Push();

  class_info.inheritance_kind = GetInheritanceKind(clang_def);

  // Compute the class's object representation.
  auto object_repr_id = ImportClassObjectRepr(
      context, class_id, import_ir_inst_id, class_inst_id, clang_def);
  class_info.complete_type_witness_id = AddInst(
      context,
      MakeImportedLocIdAndInst<SemIR::CompleteTypeWitness>(
          context, import_ir_inst_id,
          {.type_id = GetSingletonType(context, SemIR::WitnessType::TypeInstId),
           .object_repr_type_inst_id = object_repr_id}));

  class_info.body_block_id = context.inst_block_stack().Pop();
}

// Computes and returns the Carbon type to use as the object representation of
// the given C++ enum type. This is a builtin int type matching the enum's
// representation.
static auto ImportEnumObjectRepresentation(
    Context& context, SemIR::ImportIRInstId import_ir_inst_id,
    clang::EnumDecl* enum_decl) -> SemIR::TypeInstId {
  auto int_type = enum_decl->getIntegerType();
  CARBON_CHECK(!int_type.isNull(), "incomplete enum type {0}",
               enum_decl->getNameAsString());

  auto int_kind = int_type->isSignedIntegerType() ? SemIR::IntKind::Signed
                                                  : SemIR::IntKind::Unsigned;
  auto bit_width_id = GetOrAddInst(
      context, MakeImportedLocIdAndInst<SemIR::IntValue>(
                   context, import_ir_inst_id,
                   {.type_id = GetSingletonType(
                        context, SemIR::IntLiteralType::TypeInstId),
                    .int_id = context.ints().AddUnsigned(llvm::APInt(
                        64, context.ast_context().getIntWidth(int_type)))}));
  return context.types().GetAsTypeInstId(
      GetOrAddInst(context, SemIR::LocIdAndInst::NoLoc(SemIR::IntType{
                                .type_id = SemIR::TypeType::TypeId,
                                .int_kind = int_kind,
                                .bit_width_id = bit_width_id})));
}

// Creates a Carbon class definition based on the information in the given Clang
// enum declaration.
static auto BuildEnumDefinition(Context& context,
                                SemIR::ImportIRInstId import_ir_inst_id,
                                SemIR::ClassId class_id,
                                SemIR::TypeInstId class_inst_id,
                                clang::EnumDecl* enum_decl) -> void {
  auto& class_info = context.classes().Get(class_id);
  CARBON_CHECK(!class_info.has_definition_started());
  class_info.definition_id = class_inst_id;

  context.inst_block_stack().Push();

  // Don't allow inheritance from C++ enums, to match the behavior in C++.
  class_info.inheritance_kind = SemIR::Class::Final;

  // Compute the enum type's object representation. An enum is an adapter for
  // the corresponding builtin integer type.
  auto object_repr_id =
      ImportEnumObjectRepresentation(context, import_ir_inst_id, enum_decl);
  class_info.adapt_id = AddInst(
      context, MakeImportedLocIdAndInst(
                   context, import_ir_inst_id,
                   SemIR::AdaptDecl{.adapted_type_inst_id = object_repr_id}));
  class_info.complete_type_witness_id = AddInst(
      context,
      MakeImportedLocIdAndInst<SemIR::CompleteTypeWitness>(
          context, import_ir_inst_id,
          {.type_id = GetSingletonType(context, SemIR::WitnessType::TypeInstId),
           .object_repr_type_inst_id = object_repr_id}));

  class_info.body_block_id = context.inst_block_stack().Pop();
}

// Imports an enumerator declaration from Clang to Carbon.
static auto ImportEnumConstantDecl(Context& context,
                                   clang::EnumConstantDecl* enumerator_decl)
    -> SemIR::InstId {
  auto key = SemIR::ClangDeclKey(enumerator_decl);
  CARBON_CHECK(!IsClangDeclImported(context, key));

  // Find the enclosing enum type.
  auto enum_key = SemIR::ClangDeclKey(
      cast<clang::EnumDecl>(enumerator_decl->getDeclContext()));
  auto type_inst_id = LookupClangDeclInstId(context, enum_key);
  auto type_id = context.types().GetTypeIdForTypeInstId(type_inst_id);

  // Build a corresponding IntValue.
  auto int_id = context.ints().Add(enumerator_decl->getInitVal());
  auto import_ir_inst_id =
      AddImportIRInst(context.sem_ir(), enumerator_decl->getLocation());
  auto inst_id = AddInstInNoBlock(
      context,
      MakeImportedLocIdAndInst<SemIR::IntValue>(
          context, import_ir_inst_id, {.type_id = type_id, .int_id = int_id}));
  context.imports().push_back(inst_id);
  context.clang_decls().Add({.key = key, .inst_id = inst_id});
  return inst_id;
}

// Mark the given `key` as failed in `clang_decls`.
static auto MarkFailedDecl(Context& context, SemIR::ClangDeclKey key) {
  context.clang_decls().Add({.key = key, .inst_id = SemIR::ErrorInst::InstId});
}

// Creates an integer type of the given size.
static auto MakeIntType(Context& context, IntId size_id, bool is_signed)
    -> TypeExpr {
  auto type_inst_id = MakeIntTypeLiteral(
      context, Parse::NodeId::None,
      is_signed ? SemIR::IntKind::Signed : SemIR::IntKind::Unsigned, size_id);
  return ExprAsType(context, Parse::NodeId::None, type_inst_id);
}

static auto MakeCppCompatType(Context& context, SemIR::LocId loc_id,
                              CoreIdentifier name) -> TypeExpr {
  return ExprAsType(
      context, loc_id,
      LookupNameInCore(context, loc_id, {CoreIdentifier::CppCompat, name}));
}

// Maps a C++ builtin integer type to a Carbon `Core.CppCompat` type.
static auto MapBuiltinCppCompatIntegerType(Context& context,
                                           unsigned int cpp_width,
                                           unsigned int carbon_width,
                                           CoreIdentifier cpp_compat_name)
    -> TypeExpr {
  if (cpp_width != carbon_width) {
    return TypeExpr::None;
  }

  return MakeCppCompatType(context, Parse::NodeId::None, cpp_compat_name);
}

// Maps a C++ builtin integer type to a Carbon type.
// TODO: Handle integer types that map to named aliases.
static auto MapBuiltinIntegerType(Context& context, SemIR::LocId loc_id,
                                  clang::QualType qual_type,
                                  const clang::BuiltinType& type) -> TypeExpr {
  clang::ASTContext& ast_context = context.ast_context();
  unsigned width = ast_context.getIntWidth(qual_type);
  bool is_signed = type.isSignedInteger();
  auto int_n_type = ast_context.getIntTypeForBitwidth(width, is_signed);
  if (clang::ASTContext::hasSameType(qual_type, int_n_type)) {
    TypeExpr type_expr =
        MakeIntType(context, context.ints().Add(width), is_signed);
    // Try to make sure integer types of 32 or 64 bits are complete so we can
    // check against them when deciding whether we need to generate a thunk.
    if (width == 32 || width == 64) {
      SemIR::TypeId type_id = type_expr.type_id;
      if (!context.types().IsComplete(type_id)) {
        TryToCompleteType(context, type_id, loc_id);
      }
    }
    return type_expr;
  }
  if (clang::ASTContext::hasSameType(qual_type, ast_context.CharTy)) {
    return ExprAsType(context, Parse::NodeId::None,
                      MakeCharTypeLiteral(context, Parse::NodeId::None));
  }
  if (clang::ASTContext::hasSameType(qual_type, ast_context.LongTy)) {
    return MapBuiltinCppCompatIntegerType(context, width, 32,
                                          CoreIdentifier::Long32);
  }
  if (clang::ASTContext::hasSameType(qual_type, ast_context.UnsignedLongTy)) {
    return MapBuiltinCppCompatIntegerType(context, width, 32,
                                          CoreIdentifier::ULong32);
  }
  if (clang::ASTContext::hasSameType(qual_type, ast_context.LongLongTy)) {
    return MapBuiltinCppCompatIntegerType(context, width, 64,
                                          CoreIdentifier::LongLong64);
  }
  if (clang::ASTContext::hasSameType(qual_type,
                                     ast_context.UnsignedLongLongTy)) {
    return MapBuiltinCppCompatIntegerType(context, width, 64,
                                          CoreIdentifier::ULongLong64);
  }
  return TypeExpr::None;
}

static auto MapNullptrType(Context& context, SemIR::LocId loc_id) -> TypeExpr {
  return MakeCppCompatType(context, loc_id, CoreIdentifier::NullptrT);
}

// Maps a C++ builtin type to a Carbon type.
// TODO: Support more builtin types.
static auto MapBuiltinType(Context& context, SemIR::LocId loc_id,
                           clang::QualType qual_type,
                           const clang::BuiltinType& type) -> TypeExpr {
  clang::ASTContext& ast_context = context.ast_context();
  if (type.isBooleanType()) {
    CARBON_CHECK(ast_context.hasSameType(qual_type, ast_context.BoolTy));
    return ExprAsType(context, Parse::NodeId::None,
                      context.types().GetTypeInstId(GetSingletonType(
                          context, SemIR::BoolType::TypeInstId)));
  }
  if (type.isInteger()) {
    return MapBuiltinIntegerType(context, loc_id, qual_type, type);
  }
  if (type.isFloatingPoint()) {
    if (type.isFloat16Type() || type.isFloat32Type() || type.isDoubleType() ||
        type.isFloat128Type()) {
      return ExprAsType(
          context, Parse::NodeId::None,
          MakeFloatTypeLiteral(
              context, Parse::NodeId::None,
              context.ints().Add(ast_context.getTypeSize(qual_type))));
    }
    // TODO: Handle floating-point types that map to named aliases.
  } else if (type.isVoidType()) {
    return MakeCppCompatType(context, loc_id, CoreIdentifier::VoidBase);
  } else if (type.isNullPtrType()) {
    return MapNullptrType(context, loc_id);
  }

  return TypeExpr::None;
}

// Determines whether record_decl is a C++ class that has a custom mapping into
// Carbon, and if so, returns the corresponding Carbon type. Otherwise returns
// None.
static auto LookupCustomRecordType(Context& context,
                                   const clang::CXXRecordDecl* record_decl)
    -> TypeExpr {
  switch (GetCustomCppTypeMapping(record_decl)) {
    case CustomCppTypeMapping::None:
      return TypeExpr::None;

    case CustomCppTypeMapping::Str:
      return MakeStringType(
          context,
          AddImportIRInst(context.sem_ir(), record_decl->getLocation()));
  }
}

// Maps a C++ tag type (class, struct, union, enum) to a Carbon type.
static auto MapTagType(Context& context, const clang::TagType& type)
    -> TypeExpr {
  auto* tag_decl = type.getDecl();
  CARBON_CHECK(tag_decl);

  // Check if the declaration is already mapped.
  auto key = SemIR::ClangDeclKey(tag_decl);
  SemIR::InstId tag_inst_id = LookupClangDeclInstId(context, key);
  if (!tag_inst_id.has_value()) {
    if (auto* record_decl = dyn_cast<clang::CXXRecordDecl>(tag_decl)) {
      auto custom_type = LookupCustomRecordType(context, record_decl);
      if (custom_type.inst_id.has_value()) {
        context.clang_decls().Add({.key = key, .inst_id = custom_type.inst_id});
        return custom_type;
      }
    }

    tag_inst_id = ImportTagDecl(context, tag_decl);
  }
  SemIR::TypeInstId record_type_inst_id =
      context.types().GetAsTypeInstId(tag_inst_id);
  return {
      // TODO: inst_id's location should be the location of the usage, not
      // the location of the type definition. Possibly we should synthesize a
      // NameRef inst, to match how this would work in Carbon code.
      .inst_id = record_type_inst_id,
      .type_id = context.types().GetTypeIdForTypeInstId(record_type_inst_id)};
}

// Maps a C++ type that is not a wrapper type such as a pointer to a Carbon
// type.
// TODO: Support more types.
static auto MapNonWrapperType(Context& context, SemIR::LocId loc_id,
                              clang::QualType type) -> TypeExpr {
  if (const auto* builtin_type = type->getAs<clang::BuiltinType>()) {
    return MapBuiltinType(context, loc_id, type, *builtin_type);
  }

  if (const auto* tag_type = type->getAs<clang::TagType>()) {
    return MapTagType(context, *tag_type);
  }

  CARBON_CHECK(!type.hasQualifiers() && !type->isPointerType(),
               "Should not see wrapper types here");

  return TypeExpr::None;
}

// Maps a qualified C++ type to a Carbon type.
static auto MapQualifiedType(Context& context, clang::QualType type,
                             TypeExpr type_expr) -> TypeExpr {
  auto quals = type.getQualifiers();

  if (quals.hasConst()) {
    auto type_id = GetConstType(context, type_expr.inst_id);
    type_expr = TypeExpr::ForUnsugared(context, type_id);
    quals.removeConst();
  }

  // TODO: Support other qualifiers.
  if (!quals.empty()) {
    return TypeExpr::None;
  }

  return type_expr;
}

// Returns true if the type has the `_Nonnull` attribute.
static auto IsClangTypeNonNull(clang::QualType type) -> bool {
  auto nullability = type->getNullability();
  return nullability.has_value() &&
         *nullability == clang::NullabilityKind::NonNull;
}

// Like `clang::QualType::getUnqualifiedType()`, retrieves the unqualified
// variant of the given type, but preserves `_Nonnull`.
static auto ClangGetUnqualifiedTypePreserveNonNull(
    Context& context, clang::QualType original_type) -> clang::QualType {
  clang::QualType type = original_type.getUnqualifiedType();
  // Preserve non-nullability.
  if (IsClangTypeNonNull(original_type) && !IsClangTypeNonNull(type)) {
    type = context.ast_context().getAttributedType(
        clang::NullabilityKind::NonNull, type, type);
  }
  return type;
}

// Returns the type `Core.Optional(T)`, where  `T` is described by
// `inner_type_inst_id`.
static auto MakeOptionalType(Context& context, SemIR::LocId loc_id,
                             SemIR::InstId inner_type_inst_id) -> TypeExpr {
  auto fn_inst_id = LookupNameInCore(context, loc_id, CoreIdentifier::Optional);
  auto call_id = PerformCall(context, loc_id, fn_inst_id, {inner_type_inst_id});
  return ExprAsType(context, loc_id, call_id);
}

// Maps a C++ pointer type to a Carbon pointer type.
static auto MapPointerType(Context& context, SemIR::LocId loc_id,
                           clang::QualType type, TypeExpr pointee_type_expr)
    -> TypeExpr {
  CARBON_CHECK(type->isPointerType());

  bool optional =
      !IsClangTypeNonNull(type) &&
      // If the type was produced by C++ template substitution, then we assume
      // it was deduced from a Carbon pointer type, so it's non-null.
      !type->getAs<clang::SubstTemplateTypeParmType>();

  TypeExpr pointer_type_expr = TypeExpr::ForUnsugared(
      context, GetPointerType(context, pointee_type_expr.inst_id));
  if (optional) {
    pointer_type_expr =
        MakeOptionalType(context, loc_id, pointer_type_expr.inst_id);
  }
  return pointer_type_expr;
}

// Maps a C++ reference type to a Carbon type. We map all references to
// pointers for now. Note that when mapping function parameters and return
// types, a different rule is used; see MapParameterType for details.
// TODO: Revisit this and decide what we really want to do here.
static auto MapReferenceType(Context& context, clang::QualType type,
                             TypeExpr referenced_type_expr) -> TypeExpr {
  CARBON_CHECK(type->isReferenceType());
  SemIR::TypeId pointer_type_id =
      GetPointerType(context, referenced_type_expr.inst_id);
  pointer_type_id =
      GetConstType(context, context.types().GetTypeInstId(pointer_type_id));
  return TypeExpr::ForUnsugared(context, pointer_type_id);
}

// Maps a C++ type to a Carbon type. `type` should not be canonicalized because
// we check for pointer nullability and nullability will be lost by
// canonicalization.
static auto MapType(Context& context, SemIR::LocId loc_id, clang::QualType type)
    -> TypeExpr {
  // Unwrap any type modifiers and wrappers.
  llvm::SmallVector<clang::QualType> wrapper_types;
  while (true) {
    clang::QualType orig_type = type;
    if (type.hasQualifiers()) {
      type = ClangGetUnqualifiedTypePreserveNonNull(context, type);
    } else if (type->isPointerType()) {
      type = type->getPointeeType();
    } else if (type->isReferenceType()) {
      type = type.getNonReferenceType();
    } else {
      break;
    }
    wrapper_types.push_back(orig_type);
  }

  auto mapped = MapNonWrapperType(context, loc_id, type);

  for (auto wrapper : llvm::reverse(wrapper_types)) {
    if (!mapped.inst_id.has_value() ||
        mapped.type_id == SemIR::ErrorInst::TypeId) {
      break;
    }

    if (wrapper.hasQualifiers()) {
      mapped = MapQualifiedType(context, wrapper, mapped);
    } else if (wrapper->isPointerType()) {
      mapped = MapPointerType(context, loc_id, wrapper, mapped);
    } else if (wrapper->isReferenceType()) {
      mapped = MapReferenceType(context, wrapper, mapped);
    } else {
      CARBON_FATAL("Unexpected wrapper type {0}", wrapper.getAsString());
    }
  }

  return mapped;
}

namespace {
// Information about how to map a C++ parameter type into Carbon.
struct ParameterTypeInfo {
  // The type to use for the Carbon parameter.
  TypeExpr type;
  // Whether to build a `ref` pattern.
  bool want_ref_pattern;
};
}  // namespace

// Given the type of a C++ function parameter, returns information about the
// type to use for the corresponding Carbon parameter.
//
// Note that if the parameter has a type for which `IsSimpleAbiType` returns
// true, we must produce a parameter type that has the same calling convention
// as the C++ type.
static auto MapParameterType(Context& context, SemIR::LocId loc_id,
                             clang::QualType param_type) -> ParameterTypeInfo {
  ParameterTypeInfo info = {.type = TypeExpr::None, .want_ref_pattern = false};

  // Perform some custom mapping for parameters of reference type:
  //
  //   * `T& x` -> `ref x: T`.
  //   * `const T& x` -> `x: T`.
  //   * `T&& x` -> `x: T`.
  //
  // TODO: For the `&&` mapping, we allow an rvalue reference to bind to a
  // durable reference expression. This should not be allowed.
  if (param_type->isReferenceType()) {
    clang::QualType pointee_type = param_type->getPointeeType();
    if (param_type->isLValueReferenceType()) {
      if (pointee_type.isConstQualified()) {
        // TODO: Consider only doing this if `const` is the only qualifier. For
        // now, any other qualifier will fail when mapping the type.
        auto split_type = pointee_type.getSplitUnqualifiedType();
        split_type.Quals.removeConst();
        pointee_type = context.ast_context().getQualifiedType(split_type);
      } else {
        // The reference will map to a `ref` pattern.
        info.want_ref_pattern = true;
      }
    }
    param_type = pointee_type;
  }

  info.type = MapType(context, loc_id, param_type);
  return info;
}

// Returns a block for the implicit parameters of the given function
// declaration. Because function templates are not yet supported, this currently
// only contains the `self` parameter. On error, produces a diagnostic and
// returns None.
static auto MakeImplicitParamPatternsBlockId(
    Context& context, SemIR::LocId loc_id,
    const clang::FunctionDecl& clang_decl) -> SemIR::InstBlockId {
  const auto* method_decl = dyn_cast<clang::CXXMethodDecl>(&clang_decl);
  if (!method_decl || method_decl->isStatic() ||
      isa<clang::CXXConstructorDecl>(clang_decl)) {
    return SemIR::InstBlockId::Empty;
  }

  // Build a `self` parameter from the object parameter.
  BeginSubpattern(context);

  clang::QualType param_type =
      method_decl->getFunctionObjectParameterReferenceType();
  auto param_info = MapParameterType(context, loc_id, param_type);
  auto [type_inst_id, type_id] = param_info.type;
  SemIR::ExprRegionId type_expr_region_id =
      EndSubpatternAsExpr(context, type_inst_id);

  if (!type_id.has_value()) {
    context.TODO(loc_id,
                 llvm::formatv("Unsupported: object parameter type: {0}",
                               param_type.getAsString()));
    return SemIR::InstBlockId::None;
  }

  // TODO: Fill in a location once available.
  auto pattern_id = AddParamPattern(context, loc_id, SemIR::NameId::SelfValue,
                                    type_expr_region_id, type_id,
                                    param_info.want_ref_pattern);

  return context.inst_blocks().Add({pattern_id});
}

// Returns a block id for the explicit parameters of the given function
// declaration. If the function declaration has no parameters, it returns
// `SemIR::InstBlockId::Empty`. In the case of an unsupported parameter type, it
// produces an error and returns `SemIR::InstBlockId::None`. `signature`
// specifies how to convert the C++ signature to the Carbon signature.
// TODO: Consider refactoring to extract and reuse more logic from
// `HandleAnyBindingPattern()`.
static auto MakeParamPatternsBlockId(Context& context, SemIR::LocId loc_id,
                                     const clang::FunctionDecl& clang_decl,
                                     SemIR::ClangDeclKey::Signature signature)
    -> SemIR::InstBlockId {
  llvm::SmallVector<SemIR::InstId> param_ids;
  llvm::SmallVector<SemIR::InstId> param_type_ids;
  param_ids.reserve(signature.num_params);
  param_type_ids.reserve(signature.num_params);
  CARBON_CHECK(static_cast<int>(clang_decl.getNumNonObjectParams()) >=
                   signature.num_params,
               "Function has fewer parameters than requested: {0} < {1}",
               clang_decl.getNumNonObjectParams(), signature.num_params);
  const auto* function_type =
      clang_decl.getType()->castAs<clang::FunctionProtoType>();
  for (int i : llvm::seq(signature.num_params)) {
    const auto* param = clang_decl.getNonObjectParameter(i);
    clang::QualType orig_param_type = function_type->getParamType(
        clang_decl.hasCXXExplicitFunctionObjectParameter() + i);

    // The parameter type is decayed but hasn't necessarily had its qualifiers
    // removed.
    // TODO: The presence of qualifiers here is probably a Clang bug.
    clang::QualType param_type =
        ClangGetUnqualifiedTypePreserveNonNull(context, orig_param_type);

    // Mark the start of a region of insts, needed for the type expression
    // created later with the call of `EndSubpatternAsExpr()`.
    BeginSubpattern(context);
    auto param_info = MapParameterType(context, loc_id, param_type);
    auto [type_inst_id, type_id] = param_info.type;
    // Type expression of the binding pattern - a single-entry/single-exit
    // region that allows control flow in the type expression e.g. fn F(x: if C
    // then i32 else i64).
    SemIR::ExprRegionId type_expr_region_id =
        EndSubpatternAsExpr(context, type_inst_id);

    if (!type_id.has_value()) {
      context.TODO(loc_id, llvm::formatv("Unsupported: parameter type: {0}",
                                         orig_param_type.getAsString()));
      return SemIR::InstBlockId::None;
    }

    llvm::StringRef param_name = param->getName();
    SemIR::NameId name_id =
        param_name.empty()
            // Translate an unnamed parameter to an underscore to
            // match Carbon's naming of unnamed/unused function params.
            ? SemIR::NameId::Underscore
            : AddIdentifierName(context, param_name);

    SemIR::LocId param_loc_id =
        AddImportIRInst(context.sem_ir(), param->getLocation());

    // TODO: Add template support.
    SemIR::InstId pattern_id =
        AddParamPattern(context, param_loc_id, name_id, type_expr_region_id,
                        type_id, param_info.want_ref_pattern);
    param_ids.push_back(pattern_id);
    param_type_ids.push_back(type_inst_id);
  }

  switch (signature.kind) {
    case SemIR::ClangDeclKey::Signature::Normal: {
      // Use the converted parameter list as-is.
      break;
    }

    case SemIR::ClangDeclKey::Signature::TuplePattern: {
      // Replace the parameters with a single tuple pattern containing the
      // converted parameter list.
      auto param_block_id = context.inst_blocks().Add(param_ids);
      auto tuple_pattern_type_id =
          GetPatternType(context, GetTupleType(context, param_type_ids));
      SemIR::InstId pattern_id = AddPatternInst(
          context,
          SemIR::LocIdAndInst::UncheckedLoc(
              loc_id, SemIR::TuplePattern{.type_id = tuple_pattern_type_id,
                                          .elements_id = param_block_id}));
      param_ids = {pattern_id};
      break;
    }
  }

  return context.inst_blocks().Add(param_ids);
}

// Returns the return `TypeExpr` of the given function declaration. In case of
// an unsupported return type, returns `SemIR::ErrorInst::InstId`. Constructors
// are treated as returning a class instance.
// TODO: Support more return types.
static auto GetReturnTypeExpr(Context& context, SemIR::LocId loc_id,
                              clang::FunctionDecl* clang_decl)
    -> Context::FormExpr {
  auto make_init_form = [&](SemIR::TypeInstId type_component_inst_id) {
    SemIR::InitForm inst = {
        .type_id = SemIR::FormType::TypeId,
        .type_component_inst_id = type_component_inst_id,
        .index = context.full_pattern_stack().NextCallParamIndex()};
    return context.constant_values().GetInstId(TryEvalInst(context, inst));
  };
  auto make_ref_form = [&](SemIR::TypeInstId type_component_inst_id) {
    SemIR::RefForm inst = {.type_id = SemIR::FormType::TypeId,
                           .type_component_inst_id = type_component_inst_id};
    return context.constant_values().GetInstId(TryEvalInst(context, inst));
  };
  clang::QualType orig_ret_type = clang_decl->getReturnType();
  if (!orig_ret_type->isVoidType()) {
    bool is_reference = orig_ret_type->isReferenceType();
    if (is_reference) {
      orig_ret_type = orig_ret_type->getPointeeType();
    }
    // TODO: We should eventually map reference returns to non-pointer types
    // here. We should return by `ref` for `T&` return types once `ref` return
    // is implemented.
    auto [orig_type_inst_id, type_id] = MapType(context, loc_id, orig_ret_type);
    if (!orig_type_inst_id.has_value()) {
      context.TODO(loc_id, llvm::formatv("Unsupported: return type: {0}",
                                         orig_ret_type.getAsString()));
      return Context::FormExpr::Error;
    }
    Context::FormExpr result = {
        .form_inst_id = is_reference ? make_ref_form(orig_type_inst_id)
                                     : make_init_form(orig_type_inst_id),
        .type_component_inst_id = orig_type_inst_id,
        .type_component_id = type_id};

    return result;
  }

  auto* ctor = dyn_cast<clang::CXXConstructorDecl>(clang_decl);
  if (!ctor) {
    // void.
    return {.form_inst_id = SemIR::InstId::None,
            .type_component_inst_id = SemIR::TypeInstId::None,
            .type_component_id = SemIR::TypeId::None};
  }

  // TODO: Make this a `PartialType`.
  SemIR::TypeInstId record_type_inst_id = context.types().GetAsTypeInstId(
      LookupClangDeclInstId(context, SemIR::ClangDeclKey(ctor->getParent())));
  return {.form_inst_id = make_init_form(record_type_inst_id),
          .type_component_inst_id = record_type_inst_id,
          .type_component_id =
              context.types().GetTypeIdForTypeInstId(record_type_inst_id)};
}

// Information about a function's declared return type, corresponding to the
// fields of SemIR::Function with the same names.
struct ReturnInfo {
  SemIR::TypeInstId return_type_inst_id;
  SemIR::InstId return_form_inst_id;
  SemIR::InstBlockId return_patterns_id;
};

// Returns information about the declared return type of the given function
// declaration. In case of an unsupported return type, it produces a diagnostic,
// and the returned return_type_inst_id will be `SemIR::ErrorInst::InstId`.
// Constructors are treated as returning a class instance.
static auto GetReturnInfo(Context& context, SemIR::LocId loc_id,
                          clang::FunctionDecl* clang_decl) -> ReturnInfo {
  auto [form_inst_id, type_inst_id, type_id] =
      GetReturnTypeExpr(context, loc_id, clang_decl);
  if (!form_inst_id.has_value()) {
    // void.
    return {.return_type_inst_id = SemIR::TypeInstId::None,
            .return_form_inst_id = SemIR::InstId::None,
            .return_patterns_id = SemIR::InstBlockId::None};
  }
  if (form_inst_id == SemIR::ErrorInst::InstId) {
    return {.return_type_inst_id = SemIR::ErrorInst::TypeInstId,
            .return_form_inst_id = SemIR::ErrorInst::InstId,
            .return_patterns_id = SemIR::InstBlockId::None};
  }
  auto pattern_type_id = GetPatternType(context, type_id);
  clang::SourceLocation return_type_loc =
      clang_decl->getReturnTypeSourceRange().getBegin();
  if (return_type_loc.isInvalid()) {
    // TODO: While `getReturnTypeSourceRange()` should work, it seems broken for
    // trailing return type. See
    // https://github.com/llvm/llvm-project/issues/162649. Until this is fixed,
    // we fallback to `getTypeSpecStartLoc()`.
    return_type_loc = clang_decl->getTypeSpecStartLoc();
  }
  SemIR::ImportIRInstId return_type_import_ir_inst_id =
      AddImportIRInst(context.sem_ir(), return_type_loc);
  auto return_patterns_id = SemIR::InstBlockId::Empty;
  if (auto init_form =
          context.insts().TryGetAs<SemIR::InitForm>(form_inst_id)) {
    SemIR::InstId return_slot_pattern_id = AddPatternInst(
        context, MakeImportedLocIdAndInst(
                     context, return_type_import_ir_inst_id,
                     SemIR::ReturnSlotPattern({.type_id = pattern_type_id,
                                               .type_inst_id = type_inst_id})));
    auto param_pattern_id = AddPatternInst(
        context,
        MakeImportedLocIdAndInst(
            context, return_type_import_ir_inst_id,
            SemIR::OutParamPattern({.type_id = pattern_type_id,
                                    .subpattern_id = return_slot_pattern_id,
                                    .index = init_form->index})));
    return_patterns_id = context.inst_blocks().Add({param_pattern_id});
  }
  return {.return_type_inst_id = type_inst_id,
          .return_form_inst_id = form_inst_id,
          .return_patterns_id = return_patterns_id};
}

namespace {
// Represents the insts and inst blocks associated with the parameters and
// returns of a function declaration, corresponding to the fields of
// SemIR::Function with the same names.
struct FunctionSignatureInsts {
  SemIR::InstBlockId implicit_param_patterns_id;
  SemIR::InstBlockId param_patterns_id;
  SemIR::TypeInstId return_type_inst_id;
  SemIR::InstId return_form_inst_id;
  SemIR::InstBlockId return_patterns_id;
  SemIR::InstBlockId call_param_patterns_id;
  SemIR::InstBlockId call_params_id;
};
}  // namespace

// Creates the insts and inst blocks that represent the parameters and returns
// of the given C++ function's Carbon counterpart, including emitting a callee
// pattern match to create the `Call` parameters, and returns a
// FunctionSignatureInsts containing the results. Produces a diagnostic and
// returns `std::nullopt` if the function declaration has an unsupported
// parameter type. `signature` specifies how to convert the C++ function
// signature to the Carbon function signature.
static auto CreateFunctionSignatureInsts(
    Context& context, SemIR::LocId loc_id, clang::FunctionDecl* clang_decl,
    SemIR::ClangDeclKey::Signature signature)
    -> std::optional<FunctionSignatureInsts> {
  context.full_pattern_stack().PushFullPattern(
      FullPatternStack::Kind::ImplicitParamList);
  std::optional pop = llvm::scope_exit(
      [&context] { context.full_pattern_stack().PopFullPattern(); });
  auto implicit_param_patterns_id =
      MakeImplicitParamPatternsBlockId(context, loc_id, *clang_decl);
  if (!implicit_param_patterns_id.has_value()) {
    return std::nullopt;
  }
  context.full_pattern_stack().EndImplicitParamList();
  auto param_patterns_id =
      MakeParamPatternsBlockId(context, loc_id, *clang_decl, signature);
  if (!param_patterns_id.has_value()) {
    return std::nullopt;
  }
  auto [return_type_inst_id, return_form_inst_id, return_patterns_id] =
      GetReturnInfo(context, loc_id, clang_decl);
  if (return_type_inst_id == SemIR::ErrorInst::TypeInstId) {
    return std::nullopt;
  }
  pop.reset();

  auto [call_param_patterns_id, call_params_id] =
      CalleePatternMatch(context, implicit_param_patterns_id, param_patterns_id,
                         return_patterns_id);

  return {{.implicit_param_patterns_id = implicit_param_patterns_id,
           .param_patterns_id = param_patterns_id,
           .return_type_inst_id = return_type_inst_id,
           .return_form_inst_id = return_form_inst_id,
           .return_patterns_id = return_patterns_id,
           .call_param_patterns_id = call_param_patterns_id,
           .call_params_id = call_params_id}};
}

// Returns the Carbon function name for the given function.
static auto GetFunctionName(Context& context, clang::FunctionDecl* clang_decl)
    -> SemIR::NameId {
  switch (clang_decl->getDeclName().getNameKind()) {
    case clang::DeclarationName::CXXConstructorName: {
      auto key = SemIR::ClangDeclKey(
          cast<clang::CXXConstructorDecl>(clang_decl)->getParent());
      return context.classes()
          .Get(context.insts()
                   .GetAs<SemIR::ClassDecl>(LookupClangDeclInstId(context, key))
                   .class_id)
          .name_id;
    }

    case clang::DeclarationName::CXXDestructorName: {
      return SemIR::NameId::CppDestructor;
    }

    case clang::DeclarationName::CXXOperatorName:
    case clang::DeclarationName::CXXConversionFunctionName: {
      return SemIR::NameId::CppOperator;
    }

    default: {
      return AddIdentifierName(context, clang_decl->getName());
    }
  }
}

// Creates a `FunctionDecl` and a `Function` without C++ thunk information.
// Returns std::nullopt on failure.
//
// The given Clang declaration is assumed to:
// * Have not been imported before.
// * Be of supported type (ignoring parameters).
//
// `signature` specifies how to convert the C++ function signature to the Carbon
// function signature.
static auto ImportFunction(Context& context, SemIR::LocId loc_id,
                           SemIR::ImportIRInstId import_ir_inst_id,
                           clang::FunctionDecl* clang_decl,
                           SemIR::ClangDeclKey::Signature signature)
    -> std::optional<SemIR::FunctionId> {
  StartFunctionSignature(context);

  auto function_params_insts =
      CreateFunctionSignatureInsts(context, loc_id, clang_decl, signature);

  auto [pattern_block_id, decl_block_id] =
      FinishFunctionSignature(context, /*check_unused=*/false);

  if (!function_params_insts.has_value()) {
    return std::nullopt;
  }

  auto virtual_modifier = SemIR::Function::VirtualModifier::None;
  int32_t virtual_index = -1;
  if (auto* method_decl = dyn_cast<clang::CXXMethodDecl>(clang_decl)) {
    if (method_decl->size_overridden_methods()) {
      virtual_modifier = SemIR::Function::VirtualModifier::Override;
    } else if (method_decl->isVirtual()) {
      virtual_modifier = SemIR::Function::VirtualModifier::Virtual;
    }
    if (virtual_modifier != SemIR::Function::VirtualModifier::None) {
      // TODO: Add support for Microsoft/non-Itanium vtables.
      virtual_index = dyn_cast<clang::ItaniumVTableContext>(
                          context.ast_context().getVTableContext())
                          ->getMethodVTableIndex(method_decl);
    }
  }

  auto [decl_id, function_id] = MakeFunctionDecl(
      context, import_ir_inst_id, decl_block_id, /*build_generic=*/false,
      /*is_definition=*/false,
      SemIR::Function{
          {
              .name_id = GetFunctionName(context, clang_decl),
              .parent_scope_id = GetParentNameScopeId(context, clang_decl),
              .generic_id = SemIR::GenericId::None,
              .first_param_node_id = Parse::NodeId::None,
              .last_param_node_id = Parse::NodeId::None,
              .pattern_block_id = pattern_block_id,
              .implicit_param_patterns_id =
                  function_params_insts->implicit_param_patterns_id,
              .param_patterns_id = function_params_insts->param_patterns_id,
              .is_extern = false,
              .extern_library_id = SemIR::LibraryNameId::None,
              .non_owning_decl_id = SemIR::InstId::None,
              // Set by `MakeFunctionDecl`.
              .first_owning_decl_id = SemIR::InstId::None,
          },
          {
              .call_param_patterns_id =
                  function_params_insts->call_param_patterns_id,
              .call_params_id = function_params_insts->call_params_id,
              .return_type_inst_id = function_params_insts->return_type_inst_id,
              .return_form_inst_id = function_params_insts->return_form_inst_id,
              .return_patterns_id = function_params_insts->return_patterns_id,
              .virtual_modifier = virtual_modifier,
              .virtual_index = virtual_index,
              .self_param_id = FindSelfPattern(
                  context, function_params_insts->implicit_param_patterns_id),
          }});
  context.imports().push_back(decl_id);

  context.functions().Get(function_id).clang_decl_id =
      context.clang_decls().Add(
          {.key = SemIR::ClangDeclKey::ForFunctionDecl(clang_decl, signature),
           .inst_id = decl_id});

  return function_id;
}

// Imports a C++ function, returning a corresponding Carbon function.
// `signature` specifies how to convert the C++ function signature to the Carbon
// function signature. `signature.num_params` may be less than the number of
// parameters that the C++ function has if default arguments are available for
// the trailing parameters.
static auto ImportFunctionDecl(Context& context, SemIR::LocId loc_id,
                               clang::FunctionDecl* clang_decl,
                               SemIR::ClangDeclKey::Signature signature)
    -> SemIR::InstId {
  auto key = SemIR::ClangDeclKey::ForFunctionDecl(clang_decl, signature);

  // Check if the declaration is already mapped.
  if (SemIR::InstId existing_inst_id = LookupClangDeclInstId(context, key);
      existing_inst_id.has_value()) {
    return existing_inst_id;
  }

  if (clang_decl->isVariadic()) {
    context.TODO(loc_id, "Unsupported: Variadic function");
    MarkFailedDecl(context, key);
    return SemIR::ErrorInst::InstId;
  }

  if (clang_decl->getTemplatedKind() ==
      clang::FunctionDecl::TK_FunctionTemplate) {
    context.TODO(loc_id, "Unsupported: Template function");
    MarkFailedDecl(context, key);
    return SemIR::ErrorInst::InstId;
  }

  auto import_ir_inst_id =
      AddImportIRInst(context.sem_ir(), clang_decl->getLocation());

  CARBON_CHECK(clang_decl->getFunctionType()->isFunctionProtoType(),
               "Not Prototype function (non-C++ code)");
  auto function_id =
      ImportFunction(context, loc_id, import_ir_inst_id, clang_decl, signature);
  if (!function_id) {
    MarkFailedDecl(context, key);
    return SemIR::ErrorInst::InstId;
  }

  SemIR::Function& function_info = context.functions().Get(*function_id);
  if (IsCppThunkRequired(context, function_info)) {
    Diagnostics::AnnotationScope annotate_diagnostics(
        &context.emitter(), [&](auto& builder) {
          CARBON_DIAGNOSTIC(InCppThunk, Note,
                            "in thunk for C++ function used here");
          builder.Note(loc_id, InCppThunk);
        });

    if (clang::FunctionDecl* thunk_clang_decl =
            BuildCppThunk(context, function_info)) {
      if (auto thunk_function_id = ImportFunction(
              context, loc_id, import_ir_inst_id, thunk_clang_decl,
              {.num_params =
                   static_cast<int32_t>(thunk_clang_decl->getNumParams())})) {
        SemIR::InstId thunk_function_decl_id =
            context.functions().Get(*thunk_function_id).first_owning_decl_id;
        function_info.SetHasCppThunk(thunk_function_decl_id);
      }
    }
  } else {
    // Inform Clang that the function has been referenced. This will trigger
    // instantiation if needed.
    context.clang_sema().MarkFunctionReferenced(GetCppLocation(context, loc_id),
                                                clang_decl);

    // If the function is trivial, mark it as being a builtin if possible.
    if (clang_decl->isTrivial()) {
      // Trivial destructors map to a "no_op" builtin.
      if (isa<clang::CXXDestructorDecl>(clang_decl)) {
        function_info.SetBuiltinFunction(SemIR::BuiltinFunctionKind::NoOp);
      }
      // TODO: Should we model a trivial default constructor as performing
      // value-initialization (zero-initializing all fields) or
      // default-initialization (leaving fields uniniitalized)? Either way we
      // could model that effect as a builtin.
      // TODO: Add a builtin to model trivial copies.
    }
  }

  return function_info.first_owning_decl_id;
}

namespace {
// An item to be imported in an import worklist.
// TODO: If worklists ever become particularly large, consider changing this
// to use a `PointerIntPair`.
struct ImportItem {
  // A declaration that we want to import.
  SemIR::ClangDeclKey decl_key;
  // Whether we have added `decl`'s dependencies to the worklist.
  bool added_dependencies;
};
// A worklist of declarations to import.
using ImportWorklist = llvm::SmallVector<ImportItem>;
}  // namespace

// Adds the given declaration to our list of declarations to import.
static auto AddDependentDecl(Context& context, SemIR::ClangDeclKey decl,
                             ImportWorklist& worklist) -> void {
  if (!IsClangDeclImported(context, decl)) {
    worklist.push_back({.decl_key = decl, .added_dependencies = false});
  }
}

// Finds all decls that need to be imported before importing the given type and
// adds them to the given set.
static auto AddDependentUnimportedTypeDecls(Context& context,
                                            clang::QualType type,
                                            ImportWorklist& worklist) -> void {
  while (true) {
    if (type->isPointerType() || type->isReferenceType()) {
      type = type->getPointeeType();
    } else if (const clang::ArrayType* array_type =
                   type->getAsArrayTypeUnsafe()) {
      type = array_type->getElementType();
    } else {
      break;
    }
  }

  if (const auto* tag_type = type->getAs<clang::TagType>()) {
    AddDependentDecl(context, SemIR::ClangDeclKey(tag_type->getDecl()),
                     worklist);
  }
}

// Finds all decls that need to be imported before importing the given function
// and adds them to the given set.
static auto AddDependentUnimportedFunctionDecls(
    Context& context, const clang::FunctionDecl& clang_decl,
    SemIR::ClangDeclKey::Signature signature, ImportWorklist& worklist)
    -> void {
  const auto* function_type =
      clang_decl.getType()->castAs<clang::FunctionProtoType>();
  for (int i : llvm::seq(clang_decl.hasCXXExplicitFunctionObjectParameter() +
                         signature.num_params)) {
    AddDependentUnimportedTypeDecls(context, function_type->getParamType(i),
                                    worklist);
  }
  AddDependentUnimportedTypeDecls(context, clang_decl.getReturnType(),
                                  worklist);
}

// Finds all decls that need to be imported before importing the given
// declaration and adds them to the given set.
static auto AddDependentUnimportedDecls(Context& context,
                                        SemIR::ClangDeclKey key,
                                        ImportWorklist& worklist) -> void {
  clang::Decl* clang_decl = key.decl;
  if (auto* clang_function_decl = clang_decl->getAsFunction()) {
    AddDependentUnimportedFunctionDecls(context, *clang_function_decl,
                                        key.signature, worklist);
  } else if (auto* type_decl = dyn_cast<clang::TypeDecl>(clang_decl)) {
    if (!isa<clang::TagDecl>(clang_decl)) {
      AddDependentUnimportedTypeDecls(
          context, type_decl->getASTContext().getTypeDeclType(type_decl),
          worklist);
    }
  } else if (auto* var_decl = dyn_cast<clang::VarDecl>(clang_decl)) {
    AddDependentUnimportedTypeDecls(context, var_decl->getType(), worklist);
  }
  auto* parent = GetParentDecl(clang_decl);
  if (llvm::isa_and_nonnull<clang::TagDecl, clang::NamespaceDecl,
                            clang::TranslationUnitDecl>(parent)) {
    AddDependentDecl(context, SemIR::ClangDeclKey::ForNonFunctionDecl(parent),
                     worklist);
  }
}

static auto ImportVarDecl(Context& context, SemIR::LocId loc_id,
                          clang::VarDecl* var_decl) -> SemIR::InstId {
  if (SemIR::InstId existing_inst_id =
          LookupClangDeclInstId(context, SemIR::ClangDeclKey(var_decl));
      existing_inst_id.has_value()) {
    return existing_inst_id;
  }

  // Extract type and name.
  clang::QualType var_type = var_decl->getType();
  SemIR::TypeId var_type_id = MapType(context, loc_id, var_type).type_id;
  if (!var_type_id.has_value()) {
    context.TODO(loc_id, llvm::formatv("Unsupported: var type: {0}",
                                       var_type.getAsString()));
    return SemIR::ErrorInst::InstId;
  }
  SemIR::NameId var_name_id = AddIdentifierName(context, var_decl->getName());

  // Create an entity name to identify this variable.
  SemIR::EntityNameId entity_name_id = context.entity_names().Add(
      {.name_id = var_name_id,
       .parent_scope_id = GetParentNameScopeId(context, var_decl),
       .is_unused = false});

  // Create `RefBindingPattern` and `VarPattern`. Mirror the behavior of
  // import_ref and don't create a `NameBindingDecl` here; we'd never use it for
  // anything.
  SemIR::TypeId pattern_type_id = GetPatternType(context, var_type_id);
  SemIR::InstId binding_pattern_inst_id =
      AddInstInNoBlock<SemIR::RefBindingPattern>(
          context, loc_id,
          {.type_id = pattern_type_id, .entity_name_id = entity_name_id});
  context.imports().push_back(binding_pattern_inst_id);
  auto pattern_id = AddInstInNoBlock<SemIR::VarPattern>(
      context, Parse::VariablePatternId::None,
      {.type_id = pattern_type_id, .subpattern_id = binding_pattern_inst_id});
  context.imports().push_back(pattern_id);

  // Create the imported storage for the global. We intentionally use the
  // untyped form of `AddInstInNoBlock` to bypass the check on adding an
  // instruction that requires a cleanup, because we don't want a cleanup here!
  SemIR::InstId var_storage_inst_id = AddInstInNoBlock(
      context, {loc_id, SemIR::VarStorage{.type_id = var_type_id,
                                          .pattern_id = pattern_id}});
  context.imports().push_back(var_storage_inst_id);

  // Register the variable so we don't create it again, and track the
  // corresponding declaration to use for mangling.
  auto clang_decl_id = context.clang_decls().Add(
      {.key = SemIR::ClangDeclKey(var_decl), .inst_id = var_storage_inst_id});
  context.cpp_global_names().Add({.key = {.entity_name_id = entity_name_id},
                                  .clang_decl_id = clang_decl_id});

  // Inform Clang that the variable has been referenced.
  context.clang_sema().MarkVariableReferenced(GetCppLocation(context, loc_id),
                                              var_decl);

  return var_storage_inst_id;
}

static auto ImportTemplateDecl(Context& context,
                               clang::TemplateDecl* template_decl)
    -> SemIR::InstId {
  auto key = SemIR::ClangDeclKey(template_decl);

  // TODO: Avoid doing this lookup both here and in the insertion below.
  if (SemIR::InstId existing_inst_id = LookupClangDeclInstId(context, key);
      existing_inst_id.has_value()) {
    return existing_inst_id;
  }

  // Add a placeholder instruction to resolve cycle between the clang
  // declaration and the type.
  auto import_loc_id =
      AddImportIRInst(context.sem_ir(), template_decl->getLocation());
  SemIR::StructValue value = {.type_id = SemIR::TypeId::None,
                              .elements_id = SemIR::InstBlockId::Empty};
  auto inst_id = AddPlaceholderImportedInstInNoBlock(
      context, MakeImportedLocIdAndInst(context, import_loc_id, value));

  // Create a type for the constant value.
  auto name_id = context.entity_names().Add(
      {.name_id = AddIdentifierName(context, template_decl->getName()),
       .parent_scope_id = GetParentNameScopeId(context, template_decl)});
  auto decl_id = context.clang_decls().Add({.key = key, .inst_id = inst_id});
  value.type_id = GetCppTemplateNameType(context, name_id, decl_id);

  // Update the value with its type.
  ReplaceInstBeforeConstantUse(context, inst_id, value);
  return inst_id;
}

// Imports a declaration from Clang to Carbon. Returns the instruction for the
// new Carbon declaration, which will be an ErrorInst on failure. Assumes all
// dependencies have already been imported.
static auto ImportDeclAfterDependencies(Context& context, SemIR::LocId loc_id,
                                        SemIR::ClangDeclKey key)
    -> SemIR::InstId {
  clang::Decl* clang_decl = key.decl;
  if (auto* clang_function_decl = clang_decl->getAsFunction()) {
    return ImportFunctionDecl(context, loc_id, clang_function_decl,
                              key.signature);
  }
  if (auto* clang_namespace_decl = dyn_cast<clang::NamespaceDecl>(clang_decl)) {
    return ImportNamespaceDecl(context, clang_namespace_decl);
  }
  if (auto* type_decl = dyn_cast<clang::TypeDecl>(clang_decl)) {
    auto type = clang_decl->getASTContext().getTypeDeclType(type_decl);
    auto type_inst_id = MapType(context, loc_id, type).inst_id;
    if (!type_inst_id.has_value()) {
      context.TODO(AddImportIRInst(context.sem_ir(), type_decl->getLocation()),
                   llvm::formatv("Unsupported: Type declaration: {0}",
                                 type.getAsString()));
      return SemIR::ErrorInst::InstId;
    }
    context.clang_decls().Add({.key = key, .inst_id = type_inst_id});
    return type_inst_id;
  }
  if (isa<clang::FieldDecl, clang::IndirectFieldDecl>(clang_decl)) {
    // Usable fields get imported as a side effect of importing the class.
    if (SemIR::InstId existing_inst_id = LookupClangDeclInstId(context, key);
        existing_inst_id.has_value()) {
      return existing_inst_id;
    }
    context.TODO(AddImportIRInst(context.sem_ir(), clang_decl->getLocation()),
                 "Unsupported: field declaration has unhandled type or kind");
    return SemIR::ErrorInst::InstId;
  }
  if (auto* enum_const_decl = dyn_cast<clang::EnumConstantDecl>(clang_decl)) {
    return ImportEnumConstantDecl(context, enum_const_decl);
  }
  if (auto* var_decl = dyn_cast<clang::VarDecl>(clang_decl)) {
    return ImportVarDecl(context, loc_id, var_decl);
  }
  if (auto* template_decl = dyn_cast<clang::TemplateDecl>(clang_decl)) {
    return ImportTemplateDecl(context, template_decl);
  }

  context.TODO(AddImportIRInst(context.sem_ir(), clang_decl->getLocation()),
               llvm::formatv("Unsupported: Declaration type {0}",
                             clang_decl->getDeclKindName()));
  return SemIR::ErrorInst::InstId;
}

// Attempts to import a set of declarations. Returns `false` if an error was
// produced, `true` otherwise.
static auto ImportDeclSet(Context& context, SemIR::LocId loc_id,
                          ImportWorklist& worklist) -> bool {
  // Walk the dependency graph in depth-first order, and import declarations
  // once we've imported all of their dependencies.
  while (!worklist.empty()) {
    auto& item = worklist.back();
    if (!item.added_dependencies) {
      // Skip items we've already imported. We checked this when initially
      // adding the item to the worklist, but it might have been added to the
      // worklist twice before the first time we visited it. For example, this
      // happens for `fn F(a: Cpp.T, b: Cpp.T)`.
      if (IsClangDeclImported(context, item.decl_key)) {
        worklist.pop_back();
        continue;
      }

      // First time visiting this declaration (preorder): add its dependencies
      // to the work list.
      item.added_dependencies = true;
      AddDependentUnimportedDecls(context, item.decl_key, worklist);
    } else {
      // Second time visiting this declaration (postorder): its dependencies are
      // already imported, so we can import it now.
      auto decl_key = worklist.pop_back_val().decl_key;
      auto inst_id = ImportDeclAfterDependencies(context, loc_id, decl_key);
      CARBON_CHECK(inst_id.has_value());
      if (inst_id == SemIR::ErrorInst::InstId) {
        return false;
      }
      CARBON_CHECK(IsClangDeclImported(context, decl_key));
    }
  }

  return true;
}

auto ImportCppDecl(Context& context, SemIR::LocId loc_id,
                   SemIR::ClangDeclKey key) -> SemIR::InstId {
  // Collect dependencies by walking the dependency graph in depth-first order.
  ImportWorklist worklist;
  AddDependentDecl(context, key, worklist);
  if (!ImportDeclSet(context, loc_id, worklist)) {
    return SemIR::ErrorInst::InstId;
  }
  return LookupClangDeclInstId(context, key);
}

auto ImportCppType(Context& context, SemIR::LocId loc_id, clang::QualType type)
    -> TypeExpr {
  // Collect dependencies by walking the dependency graph in depth-first order.
  ImportWorklist worklist;
  AddDependentUnimportedTypeDecls(context, type, worklist);
  if (!ImportDeclSet(context, loc_id, worklist)) {
    return {.inst_id = SemIR::ErrorInst::TypeInstId,
            .type_id = SemIR::ErrorInst::TypeId};
  }
  return MapType(context, loc_id, type);
}

// Imports a Clang declaration into Carbon and adds that name into the
// `NameScope`.
static auto ImportNameDeclIntoScope(Context& context, SemIR::LocId loc_id,
                                    SemIR::NameScopeId scope_id,
                                    SemIR::NameId name_id,
                                    SemIR::ClangDeclKey key,
                                    SemIR::AccessKind access_kind)
    -> SemIR::ScopeLookupResult {
  SemIR::InstId inst_id = ImportCppDecl(context, loc_id, key);
  if (!inst_id.has_value()) {
    return SemIR::ScopeLookupResult::MakeNotFound();
  }
  AddNameToScope(context, scope_id, name_id, access_kind, inst_id);
  return SemIR::ScopeLookupResult::MakeWrappedLookupResult(inst_id,
                                                           access_kind);
}

// Returns true if the scope is the top `Cpp` scope.
static auto IsTopCppScope(Context& context, SemIR::NameScopeId scope_id)
    -> bool {
  const SemIR::NameScope& name_scope = context.name_scopes().Get(scope_id);
  CARBON_CHECK(name_scope.is_cpp_scope());
  return name_scope.parent_scope_id() == SemIR::NameScopeId::Package;
}

// For a builtin name like `Cpp.long`, returns the associated type.
static auto LookupBuiltinName(Context& context, SemIR::LocId loc_id,
                              SemIR::NameScopeId scope_id,
                              SemIR::NameId name_id) -> SemIR::InstId {
  if (!IsTopCppScope(context, scope_id)) {
    return SemIR::InstId::None;
  }

  auto name = context.names().GetAsStringIfIdentifier(name_id);
  if (!name) {
    return SemIR::InstId::None;
  }

  const clang::ASTContext& ast_context = context.ast_context();

  // List of types based on
  // https://github.com/carbon-language/carbon-lang/blob/trunk/proposals/p5448.md#details
  auto builtin_type =
      llvm::StringSwitch<clang::QualType>(*name)
          .Case("signed_char", ast_context.SignedCharTy)
          .Case("short", ast_context.ShortTy)
          .Case("int", ast_context.IntTy)
          .Case("long", ast_context.LongTy)
          .Case("long_long", ast_context.LongLongTy)
          .Case("unsigned_char", ast_context.UnsignedCharTy)
          .Case("unsigned_short", ast_context.UnsignedShortTy)
          .Case("unsigned_int", ast_context.UnsignedIntTy)
          .Case("unsigned_long", ast_context.UnsignedLongTy)
          .Case("unsigned_long_long", ast_context.UnsignedLongLongTy)
          .Case("float", ast_context.FloatTy)
          .Case("double", ast_context.DoubleTy)
          .Case("long_double", ast_context.LongDoubleTy)
          .Case("void", ast_context.VoidTy)
          .Default(clang::QualType());
  if (builtin_type.isNull()) {
    if (*name == "nullptr") {
      // Map `Cpp.nullptr` to an uninitialized value of type `Core.CppNullptrT`.
      auto type_id = MapNullptrType(context, loc_id).type_id;
      return GetOrAddInst<SemIR::UninitializedValue>(
          context, SemIR::LocId::None, {.type_id = type_id});
    }
    return SemIR::InstId::None;
  }

  SemIR::InstId inst_id =
      MapNonWrapperType(context, loc_id, builtin_type).inst_id;
  if (!inst_id.has_value()) {
    context.TODO(loc_id, llvm::formatv("Unsupported: builtin type: {0}",
                                       builtin_type.getAsString()));
    return SemIR::ErrorInst::InstId;
  }
  return inst_id;
}

auto ImportCppOverloadSet(
    Context& context, SemIR::LocId loc_id, SemIR::NameScopeId scope_id,
    SemIR::NameId name_id, clang::CXXRecordDecl* naming_class,
    clang::UnresolvedSet<4>&& overload_set,
    clang::OverloadCandidateSet::OperatorRewriteInfo operator_rewrite_info)
    -> SemIR::InstId {
  SemIR::CppOverloadSetId overload_set_id = context.cpp_overload_sets().Add(
      SemIR::CppOverloadSet{.name_id = name_id,
                            .parent_scope_id = scope_id,
                            .naming_class = naming_class,
                            .candidate_functions = std::move(overload_set),
                            .operator_rewrite_info = operator_rewrite_info});
  auto overload_set_inst_id = AddInstInNoBlock<SemIR::CppOverloadSetValue>(
      context, loc_id,
      {.type_id = GetCppOverloadSetType(context, overload_set_id,
                                        SemIR::SpecificId::None),
       .overload_set_id = overload_set_id});

  context.imports().push_back(overload_set_inst_id);
  return overload_set_inst_id;
}

// Gets the best access for an overloaded function set. This is the access that
// we use for the overload set as a whole. More fine-grained checking is done
// after overload resolution.
static auto GetOverloadSetAccess(const clang::UnresolvedSet<4>& overload_set)
    -> SemIR::AccessKind {
  SemIR::AccessKind access_kind = SemIR::AccessKind::Private;
  for (clang::DeclAccessPair overload : overload_set.pairs()) {
    access_kind = std::min(access_kind, MapCppAccess(overload));
    if (access_kind == SemIR::AccessKind::Public) {
      break;
    }
  }
  return access_kind;
}

// Imports an overload set from Clang to Carbon and adds the name into the
// `NameScope`.
static auto ImportOverloadSetIntoScope(Context& context, SemIR::LocId loc_id,
                                       SemIR::NameScopeId scope_id,
                                       SemIR::NameId name_id,
                                       clang::CXXRecordDecl* naming_class,
                                       clang::UnresolvedSet<4>&& overload_set)
    -> SemIR::ScopeLookupResult {
  SemIR::AccessKind access_kind = GetOverloadSetAccess(overload_set);
  SemIR::InstId inst_id = ImportCppOverloadSet(
      context, loc_id, scope_id, name_id, naming_class, std::move(overload_set),
      /*operator_rewrite_info=*/{});
  AddNameToScope(context, scope_id, name_id, access_kind, inst_id);
  return SemIR::ScopeLookupResult::MakeWrappedLookupResult(inst_id,
                                                           access_kind);
}

// Imports the constructors for a given class name. The found constructors are
// imported as part of an overload set into the scope. Currently copy/move
// constructors are not imported.
static auto ImportConstructorsIntoScope(Context& context, SemIR::LocId loc_id,
                                        SemIR::NameScopeId scope_id,
                                        SemIR::NameId name_id)
    -> SemIR::ScopeLookupResult {
  auto* naming_class =
      cast<clang::CXXRecordDecl>(GetDeclContext(context, scope_id));
  clang::DeclContextLookupResult constructors_lookup =
      context.clang_sema().LookupConstructors(naming_class);

  clang::UnresolvedSet<4> overload_set;
  for (auto* decl : constructors_lookup) {
    auto info = clang::getConstructorInfo(decl);
    if (!info.Constructor || info.Constructor->isCopyOrMoveConstructor()) {
      continue;
    }
    overload_set.addDecl(info.FoundDecl, info.FoundDecl.getAccess());
  }
  if (overload_set.empty()) {
    return SemIR::ScopeLookupResult::MakeNotFound();
  }

  return ImportOverloadSetIntoScope(context, loc_id, scope_id, name_id,
                                    naming_class, std::move(overload_set));
}

// Attempts to import a builtin name from Clang to Carbon and adds the name into
// the scope.
static auto ImportBuiltinNameIntoScope(Context& context, SemIR::LocId loc_id,
                                       SemIR::NameScopeId scope_id,
                                       SemIR::NameId name_id)
    -> SemIR::ScopeLookupResult {
  SemIR::InstId builtin_inst_id =
      LookupBuiltinName(context, loc_id, scope_id, name_id);
  if (builtin_inst_id.has_value()) {
    AddNameToScope(context, scope_id, name_id, SemIR::AccessKind::Public,
                   builtin_inst_id);
    return SemIR::ScopeLookupResult::MakeWrappedLookupResult(
        builtin_inst_id, SemIR::AccessKind::Public);
  }
  return SemIR::ScopeLookupResult::MakeNotFound();
}

// Checks if the name scope is a class that is not complete.
static auto IsIncompleteClass(Context& context, SemIR::NameScopeId scope_id)
    -> bool {
  auto class_decl = context.insts().TryGetAs<SemIR::ClassDecl>(
      context.name_scopes().Get(scope_id).inst_id());
  return class_decl.has_value() &&
         !context.types().IsComplete(
             context.classes().Get(class_decl->class_id).self_type_id);
}

// Imports a macro definition into the scope. Currently supports only simple
// object-like macros that expand to a constant integer value.
// TODO: Add support for other macro types and non-integer literal values.
static auto ImportMacro(Context& context, SemIR::LocId loc_id,
                        SemIR::NameScopeId scope_id, SemIR::NameId name_id,
                        clang::MacroInfo* macro_info)
    -> SemIR::ScopeLookupResult {
  auto inst_id =
      TryEvaluateMacroToConstant(context, loc_id, name_id, macro_info);
  if (inst_id == SemIR::ErrorInst::InstId) {
    return SemIR::ScopeLookupResult::MakeNotFound();
  }

  AddNameToScope(context, scope_id, name_id, SemIR::AccessKind::Public,
                 inst_id);
  return SemIR::ScopeLookupResult::MakeWrappedLookupResult(
      inst_id, SemIR::AccessKind::Public);
}

// Looks up a macro definition in the top-level `Cpp` scope. Returns nullptr if
// the macro is not found or if it is a builtin macro, function-like macro or a
// macro used for header guards.
// TODO: Function-like and builtin macros are currently not supported and their
// support still needs to be clarified.
static auto LookupMacro(Context& context, SemIR::NameScopeId scope_id,
                        clang::IdentifierInfo* identifier_info)
    -> clang::MacroInfo* {
  if (!IsTopCppScope(context, scope_id)) {
    return nullptr;
  }
  CARBON_CHECK(identifier_info, "Identifier info is empty");
  clang::MacroInfo* macro_info =
      context.clang_sema().getPreprocessor().getMacroInfo(identifier_info);
  if (macro_info && !macro_info->isUsedForHeaderGuard() &&
      !macro_info->isFunctionLike() && !macro_info->isBuiltinMacro()) {
    return macro_info;
  }

  return nullptr;
}

auto GetClangIdentifierInfo(Context& context, SemIR::NameId name_id)
    -> clang::IdentifierInfo* {
  std::optional<llvm::StringRef> string_name =
      context.names().GetAsStringIfIdentifier(name_id);
  if (!string_name) {
    return nullptr;
  }
  clang::IdentifierInfo* identifier_info =
      context.clang_sema().getPreprocessor().getIdentifierInfo(*string_name);
  return identifier_info;
}

auto ImportNameFromCpp(Context& context, SemIR::LocId loc_id,
                       SemIR::NameScopeId scope_id, SemIR::NameId name_id)
    -> SemIR::ScopeLookupResult {
  Diagnostics::AnnotationScope annotate_diagnostics(
      &context.emitter(), [&](auto& builder) {
        CARBON_DIAGNOSTIC(InCppNameLookup, Note,
                          "in `Cpp` name lookup for `{0}`", SemIR::NameId);
        builder.Note(loc_id, InCppNameLookup, name_id);
      });
  if (IsIncompleteClass(context, scope_id)) {
    return SemIR::ScopeLookupResult::MakeError();
  }

  clang::IdentifierInfo* identifier_info =
      GetClangIdentifierInfo(context, name_id);
  if (!identifier_info) {
    return SemIR::ScopeLookupResult::MakeNotFound();
  }

  if (clang::MacroInfo* macro_info =
          LookupMacro(context, scope_id, identifier_info)) {
    return ImportMacro(context, loc_id, scope_id, name_id, macro_info);
  }
  auto lookup = ClangLookupName(context, scope_id, identifier_info);
  if (!lookup) {
    return ImportBuiltinNameIntoScope(context, loc_id, scope_id, name_id);
  }
  // Access checks are performed separately by the Carbon name lookup logic.
  lookup->suppressAccessDiagnostics();

  if (lookup->isOverloadedResult() ||
      (lookup->isSingleResult() &&
       lookup->getFoundDecl()->isFunctionOrFunctionTemplate())) {
    clang::UnresolvedSet<4> overload_set;
    overload_set.append(lookup->begin(), lookup->end());
    return ImportOverloadSetIntoScope(context, loc_id, scope_id, name_id,
                                      lookup->getNamingClass(),
                                      std::move(overload_set));
  }

  if (!lookup->isSingleResult()) {
    // Clang will diagnose ambiguous lookup results for us.
    if (!lookup->isAmbiguous()) {
      context.TODO(loc_id,
                   llvm::formatv("Unsupported: Lookup succeeded but couldn't "
                                 "find a single result; LookupResultKind: {0}",
                                 static_cast<int>(lookup->getResultKind())));
    }
    context.name_scopes().AddRequiredName(scope_id, name_id,
                                          SemIR::ErrorInst::InstId);
    return SemIR::ScopeLookupResult::MakeError();
  }
  if (IsDeclInjectedClassName(context, scope_id, name_id,
                              lookup->getFoundDecl())) {
    return ImportConstructorsIntoScope(context, loc_id, scope_id, name_id);
  }
  auto key = SemIR::ClangDeclKey::ForNonFunctionDecl(lookup->getFoundDecl());
  return ImportNameDeclIntoScope(context, loc_id, scope_id, name_id, key,
                                 MapCppAccess(lookup->begin().getPair()));
}

auto ImportClassDefinitionForClangDecl(Context& context,
                                       SemIR::ClassId class_id,
                                       SemIR::ClangDeclId clang_decl_id)
    -> bool {
  SemIR::CppFile* cpp_file = context.sem_ir().cpp_file();
  CARBON_CHECK(cpp_file);

  auto* clang_decl =
      cast<clang::TagDecl>(context.clang_decls().Get(clang_decl_id).key.decl);
  auto class_inst_id = context.types().GetAsTypeInstId(
      context.classes().Get(class_id).first_owning_decl_id);

  clang::SourceLocation loc = clang_decl->getLocation();
  // Ask Clang whether the type is complete. This triggers template
  // instantiation if necessary.
  clang::DiagnosticErrorTrap trap(cpp_file->diagnostics());
  if (!context.cpp_context()->sema().isCompleteType(
          loc, context.ast_context().getCanonicalTagType(clang_decl))) {
    // Type is incomplete. Nothing more to do, but tell the caller if we
    // produced an error.
    return !trap.hasErrorOccurred();
  }

  auto import_ir_inst_id =
      context.insts().GetCanonicalLocId(class_inst_id).import_ir_inst_id();

  if (auto* class_decl = dyn_cast<clang::CXXRecordDecl>(clang_decl)) {
    auto* class_def = class_decl->getDefinition();
    CARBON_CHECK(class_def, "Complete type has no definition");

    BuildClassDefinition(context, import_ir_inst_id, class_id, class_inst_id,
                         class_def);
  } else if (auto* enum_decl = dyn_cast<clang::EnumDecl>(clang_decl)) {
    BuildEnumDefinition(context, import_ir_inst_id, class_id, class_inst_id,
                        enum_decl);
  }
  return true;
}

auto GetAsClangVarDecl(Context& context, SemIR::InstId inst_id)
    -> clang::VarDecl* {
  if (const auto& var_storage =
          context.insts().TryGetAs<SemIR::VarStorage>(inst_id)) {
    auto var_name_id = SemIR::GetFirstBindingNameFromPatternId(
        context.sem_ir(), var_storage->pattern_id);
    if (auto cpp_global_var_id = context.sem_ir().cpp_global_vars().Lookup(
            {.entity_name_id = var_name_id});
        cpp_global_var_id.has_value()) {
      SemIR::ClangDeclId clang_decl_id = context.sem_ir()
                                             .cpp_global_vars()
                                             .Get(cpp_global_var_id)
                                             .clang_decl_id;
      return cast<clang::VarDecl>(
          context.clang_decls().Get(clang_decl_id).key.decl);
    }
  }

  return nullptr;
}

}  // namespace Carbon::Check
