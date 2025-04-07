// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/merge.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/import.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

CARBON_DIAGNOSTIC(RedeclPrevDecl, Note, "previously declared here");

// Diagnoses a redeclaration which is redundant.
static auto DiagnoseRedundant(Context& context, Lex::TokenKind decl_kind,
                              SemIR::NameId name_id, SemIRLoc new_loc,
                              SemIRLoc prev_loc) -> void {
  CARBON_DIAGNOSTIC(RedeclRedundant, Error,
                    "redeclaration of `{0} {1}` is redundant", Lex::TokenKind,
                    SemIR::NameId);
  context.emitter()
      .Build(new_loc, RedeclRedundant, decl_kind, name_id)
      .Note(prev_loc, RedeclPrevDecl)
      .Emit();
}

// Diagnoses a redefinition.
static auto DiagnoseRedef(Context& context, Lex::TokenKind decl_kind,
                          SemIR::NameId name_id, SemIRLoc new_loc,
                          SemIRLoc prev_loc) -> void {
  CARBON_DIAGNOSTIC(RedeclRedef, Error, "redefinition of `{0} {1}`",
                    Lex::TokenKind, SemIR::NameId);
  CARBON_DIAGNOSTIC(RedeclPrevDef, Note, "previously defined here");
  context.emitter()
      .Build(new_loc, RedeclRedef, decl_kind, name_id)
      .Note(prev_loc, RedeclPrevDef)
      .Emit();
}

// Diagnoses an `extern` versus non-`extern` mismatch.
static auto DiagnoseExternMismatch(Context& context, Lex::TokenKind decl_kind,
                                   SemIR::NameId name_id, SemIRLoc new_loc,
                                   SemIRLoc prev_loc) -> void {
  CARBON_DIAGNOSTIC(RedeclExternMismatch, Error,
                    "redeclarations of `{0} {1}` must match use of `extern`",
                    Lex::TokenKind, SemIR::NameId);
  context.emitter()
      .Build(new_loc, RedeclExternMismatch, decl_kind, name_id)
      .Note(prev_loc, RedeclPrevDecl)
      .Emit();
}

// Diagnoses `extern library` declared in a library importing the owned entity.
static auto DiagnoseExternLibraryInImporter(Context& context,
                                            Lex::TokenKind decl_kind,
                                            SemIR::NameId name_id,
                                            SemIRLoc new_loc, SemIRLoc prev_loc)
    -> void {
  CARBON_DIAGNOSTIC(ExternLibraryInImporter, Error,
                    "cannot declare imported `{0} {1}` as `extern library`",
                    Lex::TokenKind, SemIR::NameId);
  context.emitter()
      .Build(new_loc, ExternLibraryInImporter, decl_kind, name_id)
      .Note(prev_loc, RedeclPrevDecl)
      .Emit();
}

// Diagnoses `extern library` pointing to the wrong library.
static auto DiagnoseExternLibraryIncorrect(Context& context, SemIRLoc new_loc,
                                           SemIRLoc prev_loc) -> void {
  CARBON_DIAGNOSTIC(
      ExternLibraryIncorrect, Error,
      "declaration in {0} doesn't match `extern library` declaration",
      SemIR::LibraryNameId);
  CARBON_DIAGNOSTIC(ExternLibraryExpected, Note,
                    "previously declared with `extern library` here");
  context.emitter()
      .Build(new_loc, ExternLibraryIncorrect, context.sem_ir().library_id())
      .Note(prev_loc, ExternLibraryExpected)
      .Emit();
}

auto DiagnoseExternRequiresDeclInApiFile(Context& context, SemIRLoc loc)
    -> void {
  CARBON_DIAGNOSTIC(
      ExternRequiresDeclInApiFile, Error,
      "`extern` entities must have a declaration in the API file");
  context.emitter().Emit(loc, ExternRequiresDeclInApiFile);
}

auto DiagnoseIfInvalidRedecl(Context& context, Lex::TokenKind decl_kind,
                             SemIR::NameId name_id, RedeclInfo new_decl,
                             RedeclInfo prev_decl,
                             SemIR::ImportIRId import_ir_id) -> void {
  if (!import_ir_id.has_value()) {
    // Check for disallowed redeclarations in the same file.
    if (!new_decl.is_definition) {
      DiagnoseRedundant(context, decl_kind, name_id, new_decl.loc,
                        prev_decl.loc);
      return;
    }
    if (prev_decl.is_definition) {
      DiagnoseRedef(context, decl_kind, name_id, new_decl.loc, prev_decl.loc);
      return;
    }
    if (prev_decl.is_extern != new_decl.is_extern) {
      DiagnoseExternMismatch(context, decl_kind, name_id, new_decl.loc,
                             prev_decl.loc);
      return;
    }
    return;
  }

  if (import_ir_id == SemIR::ImportIRId::ApiForImpl) {
    // Check for disallowed redeclarations in the same library. Note that a
    // forward declaration in the impl is allowed.
    if (prev_decl.is_definition) {
      if (new_decl.is_definition) {
        DiagnoseRedef(context, decl_kind, name_id, new_decl.loc, prev_decl.loc);
      } else {
        DiagnoseRedundant(context, decl_kind, name_id, new_decl.loc,
                          prev_decl.loc);
      }
      return;
    }
    if (prev_decl.is_extern != new_decl.is_extern) {
      DiagnoseExternMismatch(context, decl_kind, name_id, new_decl.loc,
                             prev_decl.loc);
      return;
    }
    return;
  }

  // Check for disallowed redeclarations cross-library.
  if (new_decl.is_extern && context.sem_ir().is_impl()) {
    // We continue after issuing the "missing API declaration" diagnostic,
    // because it may still be helpful to note other issues with the
    // declarations.
    DiagnoseExternRequiresDeclInApiFile(context, new_decl.loc);
  }
  if (prev_decl.is_extern != new_decl.is_extern) {
    DiagnoseExternMismatch(context, decl_kind, name_id, new_decl.loc,
                           prev_decl.loc);
    return;
  }
  if (!prev_decl.extern_library_id.has_value()) {
    if (new_decl.extern_library_id.has_value()) {
      DiagnoseExternLibraryInImporter(context, decl_kind, name_id, new_decl.loc,
                                      prev_decl.loc);
    } else {
      DiagnoseRedundant(context, decl_kind, name_id, new_decl.loc,
                        prev_decl.loc);
    }
    return;
  }
  if (prev_decl.extern_library_id != SemIR::LibraryNameId::Error &&
      prev_decl.extern_library_id != context.sem_ir().library_id()) {
    DiagnoseExternLibraryIncorrect(context, new_decl.loc, prev_decl.loc);
    return;
  }
}

auto ReplacePrevInstForMerge(Context& context, SemIR::NameScopeId scope_id,
                             SemIR::NameId name_id, SemIR::InstId new_inst_id)
    -> void {
  auto& scope = context.name_scopes().Get(scope_id);
  auto entry_id = scope.Lookup(name_id);
  if (entry_id) {
    auto& result = scope.GetEntry(*entry_id).result;
    result = SemIR::ScopeLookupResult::MakeWrappedLookupResult(
        new_inst_id, result.access_kind());
  }
}

// Returns true if there was an error in declaring the entity, which will have
// previously been diagnosed.
static auto EntityHasParamError(Context& context, const DeclParams& info)
    -> bool {
  for (auto param_patterns_id :
       {info.implicit_param_patterns_id, info.param_patterns_id}) {
    if (param_patterns_id.has_value() &&
        param_patterns_id != SemIR::InstBlockId::Empty) {
      for (auto param_id : context.inst_blocks().Get(param_patterns_id)) {
        if (context.insts().Get(param_id).type_id() ==
            SemIR::ErrorInst::SingletonTypeId) {
          return true;
        }
      }
    }
  }
  return false;
}

// Returns false if a param differs for a redeclaration. The caller is expected
// to provide a diagnostic.
static auto CheckRedeclParam(Context& context, bool is_implicit_param,
                             int32_t param_index,
                             SemIR::InstId new_param_pattern_id,
                             SemIR::InstId prev_param_pattern_id,
                             SemIR::SpecificId prev_specific_id, bool diagnose,
                             bool check_self) -> bool {
  auto orig_new_param_pattern_id = new_param_pattern_id;
  auto orig_prev_param_pattern_id = prev_param_pattern_id;

  // TODO: Consider differentiating between type and name mistakes. For now,
  // taking the simpler approach because I also think we may want to refactor
  // params.
  CARBON_DIAGNOSTIC(
      RedeclParamPrevious, Note,
      "previous declaration's corresponding {0:implicit |}parameter here",
      Diagnostics::BoolAsSelect);
  auto emit_diagnostic = [&]() {
    if (!diagnose) {
      return;
    }
    CARBON_DIAGNOSTIC(RedeclParamDiffers, Error,
                      "redeclaration differs at {0:implicit |}parameter {1}",
                      Diagnostics::BoolAsSelect, int32_t);
    context.emitter()
        .Build(orig_new_param_pattern_id, RedeclParamDiffers, is_implicit_param,
               param_index + 1)
        .Note(orig_prev_param_pattern_id, RedeclParamPrevious,
              is_implicit_param)
        .Emit();
  };

  auto new_param_pattern = context.insts().Get(new_param_pattern_id);
  auto prev_param_pattern = context.insts().Get(prev_param_pattern_id);
  if (new_param_pattern.kind() != prev_param_pattern.kind()) {
    emit_diagnostic();
    return false;
  }

  if (new_param_pattern.Is<SemIR::AddrPattern>()) {
    new_param_pattern_id = new_param_pattern.As<SemIR::AddrPattern>().inner_id;
    new_param_pattern = context.insts().Get(new_param_pattern_id);
    prev_param_pattern_id =
        prev_param_pattern.As<SemIR::AddrPattern>().inner_id;
    prev_param_pattern = context.insts().Get(prev_param_pattern_id);
    if (new_param_pattern.kind() != prev_param_pattern.kind()) {
      emit_diagnostic();
      return false;
    }
  }

  if (new_param_pattern.Is<SemIR::AnyParamPattern>()) {
    new_param_pattern_id =
        new_param_pattern.As<SemIR::ValueParamPattern>().subpattern_id;
    new_param_pattern = context.insts().Get(new_param_pattern_id);
    prev_param_pattern_id =
        prev_param_pattern.As<SemIR::ValueParamPattern>().subpattern_id;
    prev_param_pattern = context.insts().Get(prev_param_pattern_id);
    if (new_param_pattern.kind() != prev_param_pattern.kind()) {
      emit_diagnostic();
      return false;
    }
  }

  auto new_name_id =
      context.entity_names()
          .Get(new_param_pattern.As<SemIR::AnyBindingPattern>().entity_name_id)
          .name_id;
  auto prev_name_id =
      context.entity_names()
          .Get(prev_param_pattern.As<SemIR::AnyBindingPattern>().entity_name_id)
          .name_id;

  if (!check_self && new_name_id == SemIR::NameId::SelfValue &&
      prev_name_id == SemIR::NameId::SelfValue) {
    return true;
  }

  auto prev_param_type_id = SemIR::GetTypeOfInstInSpecific(
      context.sem_ir(), prev_specific_id, prev_param_pattern_id);
  if (!context.types().AreEqualAcrossDeclarations(new_param_pattern.type_id(),
                                                  prev_param_type_id)) {
    if (!diagnose) {
      return false;
    }
    CARBON_DIAGNOSTIC(RedeclParamDiffersType, Error,
                      "type {3} of {0:implicit |}parameter {1} in "
                      "redeclaration differs from previous parameter type {2}",
                      Diagnostics::BoolAsSelect, int32_t, SemIR::TypeId,
                      SemIR::TypeId);
    context.emitter()
        .Build(orig_new_param_pattern_id, RedeclParamDiffersType,
               is_implicit_param, param_index + 1, prev_param_type_id,
               new_param_pattern.type_id())
        .Note(orig_prev_param_pattern_id, RedeclParamPrevious,
              is_implicit_param)
        .Emit();
    return false;
  }

  if (new_name_id != prev_name_id) {
    emit_diagnostic();
    return false;
  }

  return true;
}

// Returns false if the param refs differ for a redeclaration.
static auto CheckRedeclParams(Context& context, SemIRLoc new_decl_loc,
                              SemIR::InstBlockId new_param_patterns_id,
                              SemIRLoc prev_decl_loc,
                              SemIR::InstBlockId prev_param_patterns_id,
                              bool is_implicit_param,
                              SemIR::SpecificId prev_specific_id, bool diagnose,
                              bool check_self) -> bool {
  // This will often occur for empty params.
  if (new_param_patterns_id == prev_param_patterns_id) {
    return true;
  }

  // If exactly one of the parameter lists was present, they differ.
  if (new_param_patterns_id.has_value() != prev_param_patterns_id.has_value()) {
    if (!diagnose) {
      return false;
    }
    CARBON_DIAGNOSTIC(RedeclParamListDiffers, Error,
                      "redeclaration differs because of "
                      "{1:|missing }{0:implicit |}parameter list",
                      Diagnostics::BoolAsSelect, Diagnostics::BoolAsSelect);
    CARBON_DIAGNOSTIC(RedeclParamListPrevious, Note,
                      "previously declared "
                      "{1:with|without} {0:implicit |}parameter list",
                      Diagnostics::BoolAsSelect, Diagnostics::BoolAsSelect);
    context.emitter()
        .Build(new_decl_loc, RedeclParamListDiffers, is_implicit_param,
               new_param_patterns_id.has_value())
        .Note(prev_decl_loc, RedeclParamListPrevious, is_implicit_param,
              prev_param_patterns_id.has_value())
        .Emit();
    return false;
  }

  CARBON_CHECK(new_param_patterns_id.has_value() &&
               prev_param_patterns_id.has_value());
  const auto new_param_pattern_ids =
      context.inst_blocks().Get(new_param_patterns_id);
  const auto prev_param_pattern_ids =
      context.inst_blocks().Get(prev_param_patterns_id);
  if (new_param_pattern_ids.size() != prev_param_pattern_ids.size()) {
    if (!diagnose) {
      return false;
    }
    CARBON_DIAGNOSTIC(
        RedeclParamCountDiffers, Error,
        "redeclaration differs because of {0:implicit |}parameter count of {1}",
        Diagnostics::BoolAsSelect, int32_t);
    CARBON_DIAGNOSTIC(
        RedeclParamCountPrevious, Note,
        "previously declared with {0:implicit |}parameter count of {1}",
        Diagnostics::BoolAsSelect, int32_t);
    context.emitter()
        .Build(new_decl_loc, RedeclParamCountDiffers, is_implicit_param,
               new_param_pattern_ids.size())
        .Note(prev_decl_loc, RedeclParamCountPrevious, is_implicit_param,
              prev_param_pattern_ids.size())
        .Emit();
    return false;
  }
  for (auto [index, new_param_pattern_id, prev_param_pattern_id] :
       llvm::enumerate(new_param_pattern_ids, prev_param_pattern_ids)) {
    if (!CheckRedeclParam(context, is_implicit_param, index,
                          new_param_pattern_id, prev_param_pattern_id,
                          prev_specific_id, diagnose, check_self)) {
      return false;
    }
  }
  return true;
}

// Returns true if the two nodes represent the same syntax.
// TODO: Detect raw identifiers (will require token changes).
static auto IsNodeSyntaxEqual(Context& context, Parse::NodeId new_node_id,
                              Parse::NodeId prev_node_id) -> bool {
  if (context.parse_tree().node_kind(new_node_id) !=
      context.parse_tree().node_kind(prev_node_id)) {
    return false;
  }

  // TODO: Should there be a trivial way to check if we need to check spellings?
  // Identifiers and literals need their text checked for cross-file matching,
  // but not intra-file. Keywords and operators shouldn't need the token text
  // examined at all.
  auto new_spelling = context.tokens().GetTokenText(
      context.parse_tree().node_token(new_node_id));
  auto prev_spelling = context.tokens().GetTokenText(
      context.parse_tree().node_token(prev_node_id));
  return new_spelling == prev_spelling;
}

// Returns false if redeclaration parameter syntax doesn't match.
static auto CheckRedeclParamSyntax(Context& context,
                                   Parse::NodeId new_first_param_node_id,
                                   Parse::NodeId new_last_param_node_id,
                                   Parse::NodeId prev_first_param_node_id,
                                   Parse::NodeId prev_last_param_node_id,
                                   bool diagnose) -> bool {
  // Parse nodes may not always be available to compare.
  // TODO: Support cross-file syntax checks. Right now imports provide
  // `NodeId::None`, and we'll need to follow the declaration to its original
  // file to get the parse tree.
  if (!new_first_param_node_id.has_value() ||
      !prev_first_param_node_id.has_value()) {
    return true;
  }
  CARBON_CHECK(new_last_param_node_id.has_value(),
               "new_last_param_node_id.has_value should match "
               "new_first_param_node_id.has_value");
  CARBON_CHECK(prev_last_param_node_id.has_value(),
               "prev_last_param_node_id.has_value should match "
               "prev_first_param_node_id.has_value");
  Parse::Tree::PostorderIterator new_iter(new_first_param_node_id);
  Parse::Tree::PostorderIterator new_end(new_last_param_node_id);
  Parse::Tree::PostorderIterator prev_iter(prev_first_param_node_id);
  Parse::Tree::PostorderIterator prev_end(prev_last_param_node_id);
  // Done when one past the last node to check.
  ++new_end;
  ++prev_end;

  // Compare up to the shortest length.
  for (; new_iter != new_end && prev_iter != prev_end;
       ++new_iter, ++prev_iter) {
    auto new_node_id = *new_iter;
    auto prev_node_id = *prev_iter;
    if (!IsNodeSyntaxEqual(context, new_node_id, prev_node_id)) {
      // Skip difference if it is `Self as` vs. `as` in an `impl` declaration.
      // https://github.com/carbon-language/carbon-lang/blob/trunk/proposals/p3763.md#redeclarations
      auto new_node_kind = context.parse_tree().node_kind(new_node_id);
      auto prev_node_kind = context.parse_tree().node_kind(prev_node_id);
      if (new_node_kind == Parse::NodeKind::DefaultSelfImplAs &&
          prev_node_kind == Parse::NodeKind::SelfTypeNameExpr &&
          context.parse_tree().node_kind(prev_iter[1]) ==
              Parse::NodeKind::TypeImplAs) {
        ++prev_iter;
        continue;
      }
      if (prev_node_kind == Parse::NodeKind::DefaultSelfImplAs &&
          new_node_kind == Parse::NodeKind::SelfTypeNameExpr &&
          context.parse_tree().node_kind(new_iter[1]) ==
              Parse::NodeKind::TypeImplAs) {
        ++new_iter;
        continue;
      }
      if (!diagnose) {
        return false;
      }
      CARBON_DIAGNOSTIC(RedeclParamSyntaxDiffers, Error,
                        "redeclaration syntax differs here");
      CARBON_DIAGNOSTIC(RedeclParamSyntaxPrevious, Note,
                        "comparing with previous declaration here");
      context.emitter()
          .Build(new_node_id, RedeclParamSyntaxDiffers)
          .Note(prev_node_id, RedeclParamSyntaxPrevious)
          .Emit();
      return false;
    }
  }
  // The prefixes are the same, but the lengths may still be different. This is
  // only relevant for `impl` declarations where the final bracketing node is
  // not included in the range of nodes being compared, and in those cases
  // `diagnose` is false.
  if (new_iter != new_end) {
    CARBON_CHECK(!diagnose);
    return false;
  } else if (prev_iter != prev_end) {
    CARBON_CHECK(!diagnose);
    return false;
  }

  return true;
}

auto CheckRedeclParamsMatch(Context& context, const DeclParams& new_entity,
                            const DeclParams& prev_entity,
                            SemIR::SpecificId prev_specific_id, bool diagnose,
                            bool check_syntax, bool check_self) -> bool {
  if (EntityHasParamError(context, new_entity) ||
      EntityHasParamError(context, prev_entity)) {
    return false;
  }
  if (!CheckRedeclParams(
          context, new_entity.loc, new_entity.implicit_param_patterns_id,
          prev_entity.loc, prev_entity.implicit_param_patterns_id,
          /*is_implicit_param=*/true, prev_specific_id, diagnose, check_self)) {
    return false;
  }
  // Don't forward `check_self` here because it's extra cost, and `self` is only
  // allowed in implicit params.
  if (!CheckRedeclParams(context, new_entity.loc, new_entity.param_patterns_id,
                         prev_entity.loc, prev_entity.param_patterns_id,
                         /*is_implicit_param=*/false, prev_specific_id,
                         diagnose, /*check_self=*/true)) {
    return false;
  }
  if (check_syntax &&
      !CheckRedeclParamSyntax(context, new_entity.first_param_node_id,
                              new_entity.last_param_node_id,
                              prev_entity.first_param_node_id,
                              prev_entity.last_param_node_id, diagnose)) {
    return false;
  }
  return true;
}

}  // namespace Carbon::Check
