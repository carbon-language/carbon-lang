// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/language_server/sem_ir_text.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <tuple>
#include <utility>

#include "common/raw_string_ostream.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"

namespace Carbon::LanguageServer {

namespace {

// A line of formatted SemIR recovered from a `// CHECK:STDOUT:` line, and
// where it was written in the document.
struct ContentLine {
  // 0-based document line.
  int line;

  // 0-based document column where `text` begins.
  int column;

  // The SemIR, with the check prefix removed.
  llvm::StringRef text;
};

// The source location encoded in a name's `.loc<line>[_<column>]` suffix.
struct SourceLoc {
  // 1-based line number within the input file the SemIR was compiled from.
  int line;

  // 0-based column number.
  int column;
};

}  // namespace

// The prefix on a line of expected standard output in a test file.
static constexpr llvm::StringLiteral CheckPrefix = "// CHECK:STDOUT:";

// The prefix on a line that starts a new input file within a test file.
static constexpr llvm::StringLiteral SplitPrefix = "// ---";

// The prefix on the line that starts a new file's output within formatted
// SemIR. Note this is the same marker as `SplitPrefix` without the comment,
// because the output names the input files it came from.
static constexpr llvm::StringLiteral BlockPrefix = "--- ";

// The scope that type annotations are written in, whatever scope encloses them.
static constexpr llvm::StringLiteral ConstantsScope = "constants";

// Stands in for the scope of a top-level construct we didn't recognize, so
// that its names are still findable by the whole-block fallback.
static constexpr llvm::StringLiteral UnknownScope = "<unknown scope>";

// The text introducing the braces that hold a declared entity's names, as in
// `%F.decl: %F.type = fn_decl @F [concrete = constants.%F] { ... }`.
static constexpr llvm::StringLiteral DeclMarker = "_decl @";

// Separates the name from the value in a row of a `specific` block.
static constexpr llvm::StringLiteral SpecificSeparator = " => ";

// Starts the part of an interface or named constraint body that is written in
// the entity's `WithSelf` scope.
static constexpr llvm::StringLiteral WithSelfLabel = "!with Self:";

// Ends a `WithSelfLabel` region, returning to the entity's own scope.
static constexpr llvm::StringLiteral MembersLabel = "!members:";

// Appended to an entity's scope name to get the scope of its `!with Self:`
// region.
static constexpr llvm::StringLiteral WithSelfSuffix = ".WithSelf";

// Returns whether `c` can appear in a SemIR name or scope name. Note `.` is
// included: names are built from dot-separated segments, and a name can start
// with one, as in `%.loc12_34.1` and `%.Self`.
static auto IsNameChar(char c) -> bool {
  return llvm::isAlnum(c) || c == '_' || c == '.';
}

// Returns the end of the `%name` whose `%` is at `text[percent]`, or
// `percent + 1` if no name follows it.
static auto FindNameEnd(llvm::StringRef text, int percent) -> int {
  int end = percent + 1;
  while (end < static_cast<int>(text.size()) && IsNameChar(text[end])) {
    ++end;
  }
  // A trailing `.` belongs to the surrounding text rather than to the name.
  while (end > percent + 1 && text[end - 1] == '.') {
    --end;
  }
  return end;
}

// Returns the source location a name's `.loc<line>[_<column>]` suffix refers
// to, or `nullopt` if it has no such suffix.
static auto ParseLocSuffix(llvm::StringRef name) -> std::optional<SourceLoc> {
  auto [rest, last] = name.rsplit('.');

  // An all-digits final segment is a disambiguating counter, such as the `.1`
  // of `%T.loc5_16.1`, so the location is the segment before it.
  if (!last.empty() && llvm::all_of(last, llvm::isDigit)) {
    std::tie(rest, last) = rest.rsplit('.');
  }
  if (!last.consume_front("loc")) {
    return std::nullopt;
  }

  auto [line_text, column_text] = last.split('_');
  int line = 0;
  if (line_text.getAsInteger(10, line) || line <= 0) {
    return std::nullopt;
  }
  // A location with no column, such as `%i32.loc5`, refers to the whole line.
  int column = 1;
  if (!column_text.empty() &&
      (column_text.getAsInteger(10, column) || column <= 0)) {
    return std::nullopt;
  }
  return SourceLoc{.line = line, .column = column - 1};
}

// Returns the `@name` starting at `text[at]`, or an empty string if there
// isn't one there.
static auto ParseEntityName(llvm::StringRef text, size_t at)
    -> llvm::StringRef {
  size_t end = at + 1;
  while (end < text.size() && IsNameChar(text[end])) {
    ++end;
  }
  if (end == at + 1) {
    return llvm::StringRef();
  }
  return text.substr(at, end - at);
}

// Returns the scope that a top-level line introduces: a scope keyword for
// `constants {` and friends, and otherwise the entity name.
//
// The entity name is the first `@name` on the line, which works because every
// construct names itself before it mentions anything else:
// `generic fn @F(%T: type) {` introduces `@F` however its parameters are
// spelled, and `specific @F(constants.%i32) {` is written in `@F` too, because
// the names it maps are `@F`'s.
static auto ScopeNameFor(llvm::StringRef line) -> llvm::StringRef {
  for (llvm::StringRef keyword :
       {"constants", "imports", "generated", "file"}) {
    if (line.starts_with(keyword) &&
        line.drop_front(keyword.size()).ltrim(' ').starts_with("{")) {
      return keyword;
    }
  }

  size_t at = line.find('@');
  if (at == llvm::StringRef::npos) {
    return UnknownScope;
  }
  auto name = ParseEntityName(line, at);
  return name.empty() ? llvm::StringRef(UnknownScope) : name;
}

// Returns the entity scope that the braces on `line` hold, or an empty string
// if they hold the same scope as the line itself.
//
// The only lines that change scope without saying so are the declarations:
// the braces of `%F.decl: %F.type = fn_decl @F [...] { ... } { ... }` are
// lexically inside `file`, but the names they define belong to `@F`.
static auto DeclScopeFor(llvm::StringRef line) -> llvm::StringRef {
  size_t decl = line.find(DeclMarker);
  if (decl == llvm::StringRef::npos) {
    return llvm::StringRef();
  }
  return ParseEntityName(line, decl + DeclMarker.size() - 1);
}

// Returns whether a block's output looks like formatted SemIR, rather than
// some other part of the toolchain's output such as a lowering test's LLVM IR
// or a parse test's YAML.
static auto IsSemIR(llvm::ArrayRef<ContentLine> lines) -> bool {
  for (const auto& line : lines) {
    llvm::StringRef text = line.text;
    if (text == "constants {" || text == "imports {" || text == "generated {" ||
        text == "file {") {
      return true;
    }
    // A file whose top-level scopes are all empty still prints its entities.
    for (llvm::StringRef keyword :
         {"generic ", "specific ", "fn @", "class @", "interface @",
          "constraint @", "impl @", "final impl @", "vtable @"}) {
      if (text.starts_with(keyword)) {
        return true;
      }
    }
  }
  return false;
}

// Reads the formatted SemIR of one `--- file.carbon` block, and adds the
// definitions and references it finds to the index.
class SemIRText::BlockReader {
 public:
  explicit BlockReader(SemIRText& index, int32_t block)
      : index_(&index), block_(block) {}

  auto Read(llvm::ArrayRef<ContentLine> lines) -> void {
    lines_ = lines;
    for (int index = 0, size = lines.size(); index < size; ++index) {
      ReadLine(index);
    }
    // The output can be truncated mid-definition, so close whatever is still
    // open at the end of the block rather than dropping it.
    while (!open_definitions_.empty()) {
      FinishDefinition(open_definitions_.pop_back_val(), lines.size() - 1);
    }
    // References are resolved only once the whole block has been read, because
    // a name can be used before the line that defines it.
    Resolve();
  }

 private:
  // Where the parts of a line that need separate treatment are, as offsets
  // into the line's body.
  struct LineParts {
    // The `<type>` of a `%name: <type> = ...`, which is written in the
    // `constants` scope rather than in the line's own scope. `-1` for both if
    // the line has no type annotation.
    int type_begin = -1;
    int type_end = -1;

    // The `specific` row this line is, as an index into `specific_values_`, or
    // -1 if it isn't one.
    int32_t specific_value = -1;

    // Where the value of a `specific` row starts, or -1.
    int value_begin = -1;
  };

  // A definition whose brace group is still open.
  struct OpenDefinition {
    // Index into `definitions_`.
    int32_t definition;

    // Index into `lines_` of the line the definition starts on.
    int start_line;

    // The brace depth just before that line, which the depth returns to when
    // the definition's last brace group closes.
    int depth;
  };

  auto ReadLine(int index) -> void;
  auto ReadNames(const ContentLine& line, int body_offset, llvm::StringRef body,
                 llvm::StringRef scope, const LineParts& parts) -> void;
  auto AddDefinition(llvm::StringRef name, llvm::StringRef scope,
                     clang::clangd::Range range, llvm::StringRef text)
      -> int32_t;
  auto FinishDefinition(const OpenDefinition& open, int end_line) -> void;
  auto Resolve() -> void;
  auto LookupIn(llvm::StringRef scope, llvm::StringRef name) const
      -> std::optional<int32_t>;

  // Updates `scope_stack_` for the braces on `text`, pushing `nested_scope`
  // for each `{`. Text inside a string literal is skipped, so that a brace in
  // a name or path doesn't unbalance the stack.
  auto UpdateScopeStack(llvm::StringRef text, llvm::StringRef nested_scope)
      -> void {
    bool in_string = false;
    for (size_t i = 0; i < text.size(); ++i) {
      char c = text[i];
      if (in_string) {
        if (c == '\\') {
          ++i;
        } else if (c == '"') {
          in_string = false;
        }
      } else if (c == '"') {
        in_string = true;
      } else if (c == '{') {
        scope_stack_.push_back({.scope = nested_scope});
      } else if (c == '}' && !scope_stack_.empty()) {
        scope_stack_.pop_back();
      }
    }
  }

  // Handles the labels that change the scope of the rest of the brace group
  // they're in. An interface or named constraint body switches to the
  // `@Entity.WithSelf` scope at `!with Self:`, and back at `!members:`.
  auto UpdateScopeForLabel(llvm::StringRef body) -> void {
    if (scope_stack_.empty()) {
      return;
    }
    OpenScope& open = scope_stack_.back();
    if (body == WithSelfLabel) {
      open.saved_scope = open.scope;
      open.scope =
          index_->strings_.save(llvm::Twine(open.scope) + WithSelfSuffix);
    } else if (body == MembersLabel && !open.saved_scope.empty()) {
      open.scope = open.saved_scope;
      open.saved_scope = llvm::StringRef();
    }
  }

  SemIRText* index_;
  int32_t block_;

  // The content lines of the block being read.
  llvm::ArrayRef<ContentLine> lines_;

  // An open brace group and the scope its contents are written in.
  struct OpenScope {
    llvm::StringRef scope;

    // The scope to return to at `!members:`, or empty if we haven't passed a
    // `!with Self:` in this brace group.
    llvm::StringRef saved_scope;
  };

  // The scope of each open brace. Empty at the top level of the block, where
  // the next line introduces a scope of its own.
  llvm::SmallVector<OpenScope> scope_stack_;

  // Definitions whose brace groups haven't closed yet, innermost last.
  llvm::SmallVector<OpenDefinition> open_definitions_;

  // The header of the `specific` block being read, such as
  // `specific @F(constants.%i32)`, or empty if we aren't in one.
  llvm::StringRef specific_;

  llvm::SmallVector<PendingRef> pending_;
};

auto SemIRText::BlockReader::ReadLine(int index) -> void {
  const ContentLine& line = lines_[index];
  llvm::StringRef text = line.text;
  llvm::StringRef body = text.ltrim(' ');
  int body_offset = text.size() - body.size();

  // A label can change the scope of the rest of the brace group it's in, and
  // has no names or braces of its own, so handle it before anything else.
  if (body.starts_with("!")) {
    UpdateScopeForLabel(body);
    return;
  }

  // The scope the names on this line are written in, and the scope that any
  // braces on it open.
  llvm::StringRef scope;
  llvm::StringRef nested_scope;
  if (scope_stack_.empty()) {
    if (body.empty()) {
      return;
    }
    // A construct at the top level of the block introduces its own scope.
    scope = ScopeNameFor(body);
    nested_scope = scope;
    specific_ =
        body.starts_with("specific ") ? body.rtrim(" {") : llvm::StringRef();
  } else {
    scope = scope_stack_.back().scope;
    nested_scope = DeclScopeFor(body);
    if (nested_scope.empty()) {
      nested_scope = scope;
    }
  }

  // A line that starts with a name either defines it or, in a `specific`
  // block, gives it a value.
  LineParts parts;
  int32_t definition = -1;
  if (body.starts_with("%")) {
    int name_end = FindNameEnd(body, 0);
    llvm::StringRef name = body.substr(1, name_end - 1);
    llvm::StringRef rest = body.substr(name_end);
    clang::clangd::Range range = {
        .start = {.line = line.line, .character = line.column + body_offset},
        .end = {.line = line.line,
                .character = line.column + body_offset + name_end}};

    if (!name.empty() && !specific_.empty() &&
        rest.starts_with(SpecificSeparator)) {
      // A `specific` block doesn't define names; it maps its generic's names
      // to the values they take, so this is a reference into the generic.
      parts.specific_value = index_->specific_values_.size();
      parts.value_begin = name_end + SpecificSeparator.size();
      index_->specific_values_.push_back(
          {.info = {.range = range,
                    .specific = specific_,
                    .value = body.substr(parts.value_begin)}});
    } else if (!name.empty()) {
      definition = AddDefinition(name, scope, range, body);

      // In `%name: <type> = ...`, the type is written in the `constants`
      // scope: the formatter switches scope to print it, because a type is
      // nearly always a constant.
      if (rest.starts_with(":")) {
        size_t equals = body.find(" = ", name_end);
        if (equals != llvm::StringRef::npos) {
          parts.type_begin = name_end + 1;
          parts.type_end = equals;
        }
      }
    }
  }

  ReadNames(line, body_offset, body, scope, parts);

  int depth = scope_stack_.size();
  UpdateScopeStack(text, nested_scope);

  // A definition that leaves a brace group open continues onto later lines,
  // and isn't complete until the depth comes back to where it started. Note a
  // `fn_decl` has two brace groups, and the line between them dips to the
  // starting depth and back; because the depth is only checked once per line,
  // that doesn't end the definition.
  if (definition >= 0 && static_cast<int>(scope_stack_.size()) > depth) {
    open_definitions_.push_back(
        {.definition = definition, .start_line = index, .depth = depth});
  }
  while (!open_definitions_.empty() && static_cast<int>(scope_stack_.size()) <=
                                           open_definitions_.back().depth) {
    FinishDefinition(open_definitions_.pop_back_val(), index);
  }
}

auto SemIRText::BlockReader::FinishDefinition(const OpenDefinition& open,
                                              int end_line) -> void {
  llvm::StringRef first = lines_[open.start_line].text;
  size_t indent = first.size() - first.ltrim(' ').size();

  // Each line is a separate `// CHECK:STDOUT:` line in the document, so the
  // text has to be rebuilt rather than pointed at.
  RawStringOstream text;
  llvm::ListSeparator newline("\n");
  for (int i = open.start_line; i <= end_line; ++i) {
    llvm::StringRef line = lines_[i].text;
    text << newline
         << (line.size() >= indent ? line.drop_front(indent) : line.ltrim(' '));
  }
  index_->definitions_[open.definition].info.text =
      index_->strings_.save(text.TakeStr());
}

auto SemIRText::BlockReader::ReadNames(const ContentLine& line, int body_offset,
                                       llvm::StringRef body,
                                       llvm::StringRef scope,
                                       const LineParts& parts) -> void {
  int size = body.size();
  for (int i = 0; i < size; ++i) {
    if (body[i] != '%') {
      continue;
    }
    int name_end = FindNameEnd(body, i);
    if (name_end == i + 1) {
      continue;
    }

    // A name from another scope is written `scope.%name`, where the scope is
    // either a keyword or an `@`-prefixed entity name.
    int begin = i;
    llvm::StringRef explicit_scope;
    if (i > 0 && body[i - 1] == '.') {
      int scope_begin = i - 1;
      while (scope_begin > 0 && IsNameChar(body[scope_begin - 1])) {
        --scope_begin;
      }
      if (scope_begin > 0 && body[scope_begin - 1] == '@') {
        --scope_begin;
      }
      if (scope_begin < i - 1) {
        explicit_scope = body.substr(scope_begin, i - 1 - scope_begin);
        begin = scope_begin;
      }
    }

    int column = line.column + body_offset;
    // A `specific` row's value gets linked to the row only when this name
    // makes up the whole of it, so that a compound value such as
    // `%A.type (%A)` isn't reported as the definition of the row.
    bool is_whole_value =
        begin == parts.value_begin && name_end == static_cast<int>(body.size());
    pending_.push_back(
        {.ref = {.line = line.line,
                 .column_begin = column + begin,
                 .column_end = column + name_end,
                 .block = block_,
                 .name = body.substr(i + 1, name_end - i - 1),
                 .definitions_begin = 0,
                 .definitions_size = 0},
         .explicit_scope = explicit_scope,
         .context_scope = scope,
         .in_type = i >= parts.type_begin && name_end <= parts.type_end,
         .specific_value = i == 0 ? parts.specific_value : -1,
         .value_of_specific = is_whole_value ? parts.specific_value : -1});

    i = name_end - 1;
  }
}

auto SemIRText::BlockReader::AddDefinition(llvm::StringRef name,
                                           llvm::StringRef scope,
                                           clang::clangd::Range range,
                                           llvm::StringRef text) -> int32_t {
  int32_t index = index_->definitions_.size();
  index_->definitions_.push_back(
      {.info = {.range = range, .scope = scope, .text = text}});
  // Names are unique within a scope, but the output can be truncated or
  // regex-substituted, so keep the first of any duplicates.
  index_->blocks_[block_].scopes[scope].insert({name, index});
  return index;
}

auto SemIRText::BlockReader::LookupIn(llvm::StringRef scope,
                                      llvm::StringRef name) const
    -> std::optional<int32_t> {
  const auto& scopes = index_->blocks_[block_].scopes;
  auto scope_it = scopes.find(scope);
  if (scope_it == scopes.end()) {
    return std::nullopt;
  }
  auto name_it = scope_it->second.find(name);
  if (name_it == scope_it->second.end()) {
    return std::nullopt;
  }
  return name_it->second;
}

auto SemIRText::BlockReader::Resolve() -> void {
  for (auto& pending : pending_) {
    llvm::StringRef name = pending.ref.name;
    llvm::SmallVector<int32_t, 1> definitions;

    if (!pending.explicit_scope.empty()) {
      if (auto index = LookupIn(pending.explicit_scope, name)) {
        definitions.push_back(*index);
      }
    } else {
      // A bare name usually comes from the scope of the line it's written on,
      // except in a type annotation, where it comes from `constants`. Try the
      // likely scope first, then the other one.
      llvm::StringRef scopes[] = {pending.context_scope, ConstantsScope};
      if (pending.in_type) {
        std::swap(scopes[0], scopes[1]);
      }
      for (llvm::StringRef scope : scopes) {
        if (auto index = LookupIn(scope, name)) {
          definitions.push_back(*index);
          break;
        }
      }
    }

    if (definitions.empty()) {
      // The formatter has a few more places where it switches scope without
      // saying so, such as the constant value of a symbolic binding, which is
      // written in the generic that owns it, and it doesn't always print the
      // contents of a scope it names. Rather than model each of them, search
      // the whole block and report everything that matches.
      for (const auto& scope : index_->blocks_[block_].scopes) {
        auto it = scope.second.find(name);
        if (it != scope.second.end()) {
          definitions.push_back(it->second);
        }
      }
      // `StringMap` iteration order is unspecified; sorting puts the
      // definitions back into the order they were written in.
      llvm::sort(definitions);
    }

    pending.ref.definitions_begin = index_->ref_definitions_.size();
    pending.ref.definitions_size = definitions.size();
    llvm::append_range(index_->ref_definitions_, definitions);
    index_->refs_.push_back(pending.ref);

    if (pending.specific_value >= 0) {
      for (int32_t index : definitions) {
        index_->definitions_[index].specific_values.push_back(
            pending.specific_value);
      }
    }

    // An ambiguous value would be misleading to show as *the* definition, so
    // only record one we resolved to exactly one place.
    if (pending.value_of_specific >= 0 && definitions.size() == 1) {
      index_->specific_values_[pending.value_of_specific].value_definition =
          definitions.front();
    }
  }
}

// Recovers the formatted SemIR from the `// CHECK:STDOUT:` lines of `text`,
// and records the line each `// --- file.carbon` split was written on.
static auto ExtractContent(llvm::StringRef text,
                           llvm::SmallVectorImpl<ContentLine>& content,
                           llvm::StringMap<int>& split_lines) -> void {
  llvm::SmallVector<llvm::StringRef> lines;
  text.split(lines, '\n');

  for (auto [index, line] : llvm::enumerate(lines)) {
    llvm::StringRef body = line.rtrim('\r');
    int indent = body.size();
    body = body.ltrim(' ');
    indent -= body.size();

    if (body.consume_front(CheckPrefix)) {
      int column = indent + CheckPrefix.size();
      // An empty line of output has no space after the prefix.
      if (body.consume_front(" ")) {
        ++column;
      }
      content.push_back(
          {.line = static_cast<int>(index), .column = column, .text = body});
    } else if (body.consume_front(SplitPrefix)) {
      split_lines.insert({body.trim(), static_cast<int>(index)});
    }
  }
}

// Returns what to add to a SemIR line number to get the 0-based document line
// of the corresponding source text.
static auto FindSourceLineBase(llvm::StringRef name,
                               const llvm::StringMap<int>& split_lines)
    -> std::optional<int> {
  // Line 1 of a split is the line after its `// --- file.carbon` marker.
  if (auto it = split_lines.find(name); it != split_lines.end()) {
    return it->second;
  }
  if (split_lines.empty()) {
    // The test file isn't split, so it is itself the input file, and SemIR
    // line numbers are 1-based document line numbers.
    return -1;
  }
  // The input was pulled in by `// INCLUDE-FILE`, so its text is in a
  // different document, which we have no way to refer to from here.
  return std::nullopt;
}

SemIRText::SemIRText(llvm::StringRef text) {
  llvm::SmallVector<ContentLine> content;
  llvm::StringMap<int> split_lines;
  ExtractContent(text, content, split_lines);

  // Each `--- file.carbon` block is a separate compilation, so names in one
  // never refer to names in another. Read them independently.
  for (size_t begin = 0; begin < content.size();) {
    llvm::StringRef name;
    if (content[begin].text.starts_with(BlockPrefix)) {
      name = content[begin].text.drop_front(BlockPrefix.size()).trim();
      ++begin;
    }
    size_t end = begin;
    while (end < content.size() &&
           !content[end].text.starts_with(BlockPrefix)) {
      ++end;
    }

    auto lines = llvm::ArrayRef(content).slice(begin, end - begin);
    if (IsSemIR(lines)) {
      blocks_.push_back(
          {.source_line_base = FindSourceLineBase(name, split_lines)});
      BlockReader(*this, blocks_.size() - 1).Read(lines);
    }
    begin = end;
  }

  // Blocks and lines are read in order, so this is nearly always a no-op, but
  // `Lookup` depends on it.
  llvm::stable_sort(refs_, [](const Ref& lhs, const Ref& rhs) {
    return std::pair(lhs.line, lhs.column_begin) <
           std::pair(rhs.line, rhs.column_begin);
  });
}

auto SemIRText::FindRef(clang::clangd::Position position) const -> const Ref* {
  // References are sorted by position, so only the last one starting at or
  // before `position` can contain it.
  auto after = llvm::partition_point(refs_, [&](const Ref& ref) {
    return std::pair(ref.line, ref.column_begin) <=
           std::pair(position.line, position.character);
  });
  if (after == refs_.begin()) {
    return nullptr;
  }

  const Ref& ref = *std::prev(after);
  if (ref.line != position.line || position.character >= ref.column_end) {
    return nullptr;
  }
  return &ref;
}

auto SemIRText::Lookup(clang::clangd::Position position) const
    -> std::optional<SemIRTextNameRef> {
  const Ref* ref = FindRef(position);
  if (!ref) {
    return std::nullopt;
  }

  auto range_of = [](const Ref& ref) -> clang::clangd::Range {
    return {.start = {.line = ref.line, .character = ref.column_begin},
            .end = {.line = ref.line, .character = ref.column_end}};
  };
  auto definitions_of = [&](const Ref& ref) -> llvm::ArrayRef<int32_t> {
    return llvm::ArrayRef(ref_definitions_)
        .slice(ref.definitions_begin, ref.definitions_size);
  };

  SemIRTextNameRef result = {.range = range_of(*ref)};
  for (int32_t index : definitions_of(*ref)) {
    const Definition& definition = definitions_[index];
    result.definitions.push_back(definition.info);
    for (int32_t value : definition.specific_values) {
      const SpecificValue& specific_value = specific_values_[value];
      result.specific_values.push_back(specific_value.info);
      if (specific_value.value_definition >= 0) {
        result.specific_values.back().value_definition =
            definitions_[specific_value.value_definition].info;
      }
    }
  }

  // A name's `.loc<line>_<column>` suffix names the source text the
  // instruction was checked from, in the input file this block was compiled
  // from.
  if (auto loc = ParseLocSuffix(ref->name)) {
    if (auto base = blocks_[ref->block].source_line_base) {
      result.source_position = {.line = *base + loc->line,
                                .character = loc->column};
    }
  }

  for (const Ref& other : refs_) {
    // Two names denote the same thing when they resolve to the same
    // definitions. When neither resolves, fall back to matching the text,
    // which is still better than reporting nothing.
    bool same = definitions_of(*ref).empty() && definitions_of(other).empty()
                    ? ref->name == other.name
                    : definitions_of(*ref) == definitions_of(other);
    if (other.block == ref->block && same) {
      result.occurrences.push_back(range_of(other));
    }
  }

  return result;
}

}  // namespace Carbon::LanguageServer
