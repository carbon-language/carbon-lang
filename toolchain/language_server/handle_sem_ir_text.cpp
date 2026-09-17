// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/language_server/handle_sem_ir_text.h"

#include <optional>
#include <utility>
#include <vector>

#include "common/raw_string_ostream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "toolchain/language_server/sem_ir_text.h"

namespace Carbon::LanguageServer {

// The language id the VS Code extension uses for formatted SemIR, so that
// hover text is highlighted the same way the output it was read from is. Note
// this is the language id from `package.json`, not the TextMate scope name
// `source.carbon-semir`.
static constexpr llvm::StringLiteral SemIRLanguage = "semir";

// Writes `text` as a fenced code block of formatted SemIR.
static auto WriteCodeBlock(llvm::raw_ostream& out, llvm::StringRef text)
    -> void {
  out << "```" << SemIRLanguage << "\n" << text << "\n```\n";
}

// Returns the 0-based line `line` of `text`, without its terminator, or an
// empty string if the text has no such line.
static auto GetLine(llvm::StringRef text, int line) -> llvm::StringRef {
  for (llvm::StringRef candidate : llvm::split(text, '\n')) {
    if (line-- == 0) {
      return candidate.rtrim('\r');
    }
  }
  return llvm::StringRef();
}

// Returns a definition written the way a reference to it from outside its
// scope would be, so that the scope it came from is visible. Only the first
// line is qualified: the rest is the body of the definition's brace group.
static auto Qualified(const SemIRTextDefinition& definition) -> std::string {
  RawStringOstream out;
  out << definition.scope << "." << definition.text;
  return out.TakeStr();
}

// Returns `text` with every line indented, so that it reads as subordinate to
// the line before it.
static auto Indent(llvm::StringRef text) -> std::string {
  RawStringOstream out;
  llvm::ListSeparator newline("\n");
  for (llvm::StringRef line : llvm::split(text, '\n')) {
    out << newline << "  " << line;
  }
  return out.TakeStr();
}

auto GetSemIRTextHover(const Context::File& file,
                       const clang::clangd::Position& position)
    -> std::optional<clang::clangd::Hover> {
  const auto* sem_ir_text = file.sem_ir_text();
  if (!sem_ir_text) {
    return std::nullopt;
  }
  auto name = sem_ir_text->Lookup(position);
  if (!name) {
    return std::nullopt;
  }
  if (name->definitions.empty() && !name->source_position) {
    // We found a name but have nothing to say about it, so say nothing rather
    // than showing an empty popup.
    return std::nullopt;
  }

  RawStringOstream text;

  // The definitions, written the way a reference to them from outside their
  // scope would be, so that the scope each one came from is visible.
  for (const auto& definition : name->definitions) {
    WriteCodeBlock(text, Qualified(definition));
  }

  // What the name becomes in each specific of the generic that defines it.
  if (!name->specific_values.empty()) {
    text << "\nSpecific values:\n\n";
    RawStringOstream values;
    llvm::ListSeparator newline("\n");
    for (const auto& value : name->specific_values) {
      values << newline << value.specific << " =>";
      if (value.value_definition) {
        // The value is nearly always a reference to a constant, and the name
        // of the constant says little, so show what it's defined as. It goes
        // on its own line because a definition can be long, or several lines.
        values << "\n" << Indent(Qualified(*value.value_definition));
      } else {
        values << " " << value.value;
      }
    }
    WriteCodeBlock(text, values.TakeStr());
  }

  // The source text the name was checked from.
  if (name->source_position) {
    auto source = GetLine(file.text(), name->source_position->line).trim();
    if (!source.empty()) {
      text << "\nSource:\n\n```carbon\n" << source << "\n```\n";
    }
  }

  return clang::clangd::Hover{
      .contents = {.kind = clang::clangd::MarkupKind::Markdown,
                   .value = text.TakeStr()},
      .range = name->range};
}

auto GetSemIRTextLocations(const Context::File& file,
                           const clang::clangd::Position& position,
                           SemIRTextGoto goto_kind)
    -> std::optional<std::vector<clang::clangd::Location>> {
  const auto* sem_ir_text = file.sem_ir_text();
  if (!sem_ir_text) {
    return std::nullopt;
  }
  auto name = sem_ir_text->Lookup(position);
  if (!name) {
    return std::nullopt;
  }

  std::vector<clang::clangd::Location> locations;
  auto add = [&](clang::clangd::Range range) {
    locations.push_back({.uri = file.uri(), .range = range});
  };

  switch (goto_kind) {
    case SemIRTextGoto::Definition:
      for (const auto& definition : name->definitions) {
        add(definition.range);
      }
      break;

    case SemIRTextGoto::Source:
      if (name->source_position) {
        // We know where the source text starts but not how far it runs, so
        // point at it rather than selecting it.
        add({.start = *name->source_position, .end = *name->source_position});
      }
      break;

    case SemIRTextGoto::Specifics:
      for (const auto& value : name->specific_values) {
        add(value.range);
      }
      break;

    case SemIRTextGoto::References:
      for (const auto& occurrence : name->occurrences) {
        add(occurrence);
      }
      break;
  }
  return locations;
}

}  // namespace Carbon::LanguageServer
