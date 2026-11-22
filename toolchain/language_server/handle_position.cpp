// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/raw_string_ostream.h"
#include "toolchain/language_server/handle.h"
#include "toolchain/language_server/handle_sem_ir_text.h"
#include "toolchain/language_server/position.h"
#include "toolchain/language_server/sem_ir_index.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/stringify.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::LanguageServer {

// Returns the type of `inst_id` rendered as Carbon source, or an empty string
// if it has no type. Instructions that aren't values, such as declarations of
// namespaces, have no type to show.
//
// TODO: `StringifyTypeOfInst` renders some types as placeholders such as
// `<type of F>` for a function and `<pattern for i32>` for a binding pattern,
// which is unhelpful as hover text. Show the signature for a function, and the
// bound type rather than the pattern type for a binding.
static auto StringifyTypeForHover(const SemIR::File& sem_ir,
                                  SemIR::InstId inst_id) -> std::string {
  if (!sem_ir.insts().Get(inst_id).type_id().has_value()) {
    return "";
  }
  return SemIR::StringifyTypeOfInst(sem_ir, inst_id);
}

// Returns the file a position-based query refers to. If the request names a
// file we don't know, produces an InvalidParams error and returns null.
template <typename ResponseType>
static auto FindFileOrFail(
    Context& context, const clang::clangd::TextDocumentPositionParams& params,
    llvm::function_ref<auto(llvm::Expected<ResponseType>)->void> on_done)
    -> const Context::File* {
  auto* file = context.LookupFile(params.textDocument.uri.file());
  if (!file) {
    on_done(llvm::make_error<clang::clangd::LSPError>(
        llvm::formatv("Unknown textDocument `{0}`",
                      params.textDocument.uri.file()),
        clang::clangd::ErrorCode::InvalidParams));
  }
  return file;
}

// Implements `textDocument/hover`:
// https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_hover
auto HandleHover(
    Context& context, const clang::clangd::TextDocumentPositionParams& params,
    llvm::function_ref<auto(llvm::Expected<clang::clangd::Hover>)->void>
        on_done) -> void {
  const auto* file =
      FindFileOrFail<clang::clangd::Hover>(context, params, on_done);
  if (!file) {
    return;
  }
  if (auto hover = GetSemIRTextHover(*file, params.position)) {
    on_done(std::move(*hover));
    return;
  }

  auto info = FindPositionInfo(*file, params.position);
  if (!info.has_inst()) {
    on_done(clang::clangd::Hover());
    return;
  }

  const auto& sem_ir = *info.file->sem_ir();
  RawStringOstream text;
  text << "```carbon\n" << info.file->tokens().GetTokenText(info.token);
  if (auto type = StringifyTypeForHover(sem_ir, info.inst_id); !type.empty()) {
    text << ": " << type;
  }
  text << "\n```";

  on_done(clang::clangd::Hover{
      .contents = {.kind = clang::clangd::MarkupKind::Markdown,
                   .value = text.TakeStr()},
      .range = GetTokenRange(info.file->tokens(), info.token)});
}

// Shared implementation of the goto-style requests, which differ only in which
// instruction they resolve to.
static auto HandleGoto(
    Context& context, const clang::clangd::TextDocumentPositionParams& params,
    SemIRTextGoto sem_ir_text_goto, bool use_type,
    llvm::function_ref<
        auto(llvm::Expected<std::vector<clang::clangd::Location>>)->void>
        on_done) -> void {
  const auto* file = FindFileOrFail<std::vector<clang::clangd::Location>>(
      context, params, on_done);
  if (!file) {
    return;
  }
  if (auto locations =
          GetSemIRTextLocations(*file, params.position, sem_ir_text_goto)) {
    on_done(std::move(*locations));
    return;
  }

  auto info = FindPositionInfo(*file, params.position);
  if (!info.has_inst()) {
    on_done(std::vector<clang::clangd::Location>());
    return;
  }

  const auto& sem_ir = *info.file->sem_ir();
  auto target_id = GetReferencedInst(sem_ir, info.inst_id);
  if (use_type) {
    auto type_id = sem_ir.insts().Get(target_id).type_id();
    if (!type_id.has_value()) {
      on_done(std::vector<clang::clangd::Location>());
      return;
    }
    target_id = sem_ir.types().GetTypeInstId(type_id);
  }

  std::vector<clang::clangd::Location> locations;
  if (auto location = GetInstLocation(*info.file, target_id)) {
    locations.push_back(*location);
  }
  on_done(std::move(locations));
}

// Implements `textDocument/definition`:
// https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_definition
auto HandleDefinition(
    Context& context, const clang::clangd::TextDocumentPositionParams& params,
    llvm::function_ref<
        auto(llvm::Expected<std::vector<clang::clangd::Location>>)->void>
        on_done) -> void {
  HandleGoto(context, params, SemIRTextGoto::Definition, /*use_type=*/false,
             on_done);
}

// Implements `textDocument/declaration`:
// https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_declaration
//
// In formatted SemIR this finds the source text the name was checked from,
// which is the closest thing an instruction has to a declaration, and is the
// more useful answer given `definition` already finds the instruction.
//
// Carbon separates declaration from definition, but SemIR resolves a name to a
// single entity instruction, so for Carbon source both requests answer alike.
// TODO: Point `definition` at the definition when an entity is declared in one
// place and defined in another.
auto HandleDeclaration(
    Context& context, const clang::clangd::TextDocumentPositionParams& params,
    llvm::function_ref<
        auto(llvm::Expected<std::vector<clang::clangd::Location>>)->void>
        on_done) -> void {
  HandleGoto(context, params, SemIRTextGoto::Source, /*use_type=*/false,
             on_done);
}

// Implements `textDocument/implementation`:
// https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_implementation
//
// In formatted SemIR this finds the value the name takes in each `specific` of
// the generic that defines it.
//
// TODO: For Carbon source, find the `impl`s of an interface, and the functions
// implementing an interface's associated functions.
auto HandleImplementation(
    Context& context, const clang::clangd::TextDocumentPositionParams& params,
    llvm::function_ref<
        auto(llvm::Expected<std::vector<clang::clangd::Location>>)->void>
        on_done) -> void {
  const auto* file = FindFileOrFail<std::vector<clang::clangd::Location>>(
      context, params, on_done);
  if (!file) {
    return;
  }
  auto locations =
      GetSemIRTextLocations(*file, params.position, SemIRTextGoto::Specifics);
  on_done(locations.value_or(std::vector<clang::clangd::Location>()));
}

// Implements `textDocument/typeDefinition`:
// https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_typeDefinition
auto HandleTypeDefinition(
    Context& context, const clang::clangd::TextDocumentPositionParams& params,
    llvm::function_ref<
        auto(llvm::Expected<std::vector<clang::clangd::Location>>)->void>
        on_done) -> void {
  // A SemIR name's type is written on its defining line, which `definition`
  // already finds, so there's nothing extra to offer for formatted SemIR.
  HandleGoto(context, params, SemIRTextGoto::Definition, /*use_type=*/true,
             on_done);
}

// Implements `textDocument/references`:
// https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_references
//
// Only finds references within the file being edited. Without a project-wide
// index there's no way to see other files, so results may be incomplete.
auto HandleReferences(
    Context& context, const clang::clangd::ReferenceParams& params,
    llvm::function_ref<
        auto(llvm::Expected<std::vector<clang::clangd::Location>>)->void>
        on_done) -> void {
  const auto* file = FindFileOrFail<std::vector<clang::clangd::Location>>(
      context, params, on_done);
  if (!file) {
    return;
  }
  if (auto locations = GetSemIRTextLocations(*file, params.position,
                                             SemIRTextGoto::References)) {
    on_done(std::move(*locations));
    return;
  }

  auto info = FindPositionInfo(*file, params.position);
  if (!info.has_inst()) {
    on_done(std::vector<clang::clangd::Location>());
    return;
  }

  const auto& sem_ir = *info.file->sem_ir();
  auto target_token =
      GetInstNameToken(*info.file, GetReferencedInst(sem_ir, info.inst_id));
  if (!target_token.has_value()) {
    on_done(std::vector<clang::clangd::Location>());
    return;
  }

  // This is the one request the token index can't serve: it needs every
  // instruction referring to an entity, which is the opposite direction from
  // the index. A scan is inherent, and cheap next to the compile that produced
  // the IR.
  std::vector<clang::clangd::Location> locations;
  if (params.context.includeDeclaration) {
    locations.push_back(
        {.uri = info.file->uri(),
         .range = GetTokenRange(info.file->tokens(), target_token)});
  }
  for (auto [inst_id, inst] : sem_ir.insts().enumerate()) {
    auto name_ref = inst.TryAs<SemIR::NameRef>();
    if (!name_ref ||
        GetInstNameToken(*info.file, name_ref->value_id) != target_token) {
      continue;
    }
    if (auto location = GetInstLocation(*info.file, inst_id)) {
      locations.push_back(*location);
    }
  }
  on_done(std::move(locations));
}

}  // namespace Carbon::LanguageServer
