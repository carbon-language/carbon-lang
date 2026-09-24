// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LANGUAGE_SERVER_SEM_IR_TEXT_H_
#define CARBON_TOOLCHAIN_LANGUAGE_SERVER_SEM_IR_TEXT_H_

#include <optional>
#include <vector>

#include "clang-tools-extra/clangd/Protocol.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/StringSaver.h"

namespace Carbon::LanguageServer {

// The formatted SemIR that introduces a name, such as
// `%i32: type = class_type @Int, @Int(%int_32) [concrete]`.
struct SemIRTextDefinition {
  // The range of the `%name` the definition introduces.
  clang::clangd::Range range;

  // The scope the name belongs to: a keyword such as `constants`, or an entity
  // name such as `@F`.
  llvm::StringRef scope;

  // The defining text, with the indentation of its first line removed. This is
  // more than one line when the definition opens a brace group, as
  // `%foo: <namespace> = namespace [concrete] { ... }` does, in which case it
  // runs to the matching closing brace.
  llvm::StringRef text;
};

// A `%name => value` row of a `specific` block, giving the value that one name
// from a generic takes in one particular specific.
struct SemIRTextSpecificValue {
  // The range of the row's `%name`.
  clang::clangd::Range range;

  // The header of the enclosing `specific` block, such as
  // `specific @F(constants.%i32)`.
  llvm::StringRef specific;

  // The text to the right of the `=>`.
  llvm::StringRef value;

  // Where `value` is defined, when it is a single name reference that resolves
  // unambiguously. A specific's value is nearly always `constants.%something`,
  // and the name on its own says little, so this is what a reader wants to see.
  std::optional<SemIRTextDefinition> value_definition;
};

// What a `%name` written at some position in formatted SemIR refers to.
struct SemIRTextNameRef {
  // The range of the reference, including any `scope.` prefix.
  clang::clangd::Range range;

  // The definitions the name resolves to. Normally one; none if we couldn't
  // resolve it, and more than one if the name is ambiguous.
  llvm::SmallVector<SemIRTextDefinition, 1> definitions;

  // The value this name takes in each `specific` of the generic that defines
  // it. Empty unless the name is defined in a generic that has specifics.
  llvm::SmallVector<SemIRTextSpecificValue, 0> specific_values;

  // Where in this document the source text named by the name's
  // `.loc<line>_<column>` suffix lives. `nullopt` if the name has no such
  // suffix, or if we couldn't work out where the input file it refers to was
  // written.
  std::optional<clang::clangd::Position> source_position;

  // Every place this name is written within the same output block, including
  // its definition and this reference itself.
  llvm::SmallVector<clang::clangd::Range, 1> occurrences;
};

// An index of the formatted SemIR that a toolchain test file carries in its
// `// CHECK:STDOUT:` lines.
//
// This is a heuristic reader, not a parser: there is no grammar for formatted
// SemIR, and the format is free to change. It recognizes just enough structure
// to answer position-based requests -- which names exist, which scope each one
// belongs to, and where each is written -- and silently ignores anything it
// doesn't understand. A test file whose expected output isn't SemIR at all,
// such as a lowering test's LLVM IR, is skipped by `--- file` block rather
// than producing nonsense.
//
// The interesting structure, and the reason a plain text search isn't enough,
// is scoping. A name is written `scope.%name` when it comes from another scope
// and bare `%name` when it comes from the current one, so resolving a name
// means knowing which scope the line it's written on belongs to. That isn't
// simply the innermost `{`: the braces of a `fn_decl @F` sit inside `file` but
// hold `@F`'s names, and a type annotation is written in the `constants` scope
// whatever scope encloses it. Where the scope is still ambiguous, we fall back
// to searching the whole output block, and report every match.
class SemIRText {
 public:
  // Indexes the formatted SemIR in `text`, which must outlive this object.
  explicit SemIRText(llvm::StringRef text);

  // Returns whether any formatted SemIR was found.
  auto empty() const -> bool { return refs_.empty(); }

  // Returns what the name written at `position` refers to, or `nullopt` if
  // `position` isn't within a `%name` in formatted SemIR.
  auto Lookup(clang::clangd::Position position) const
      -> std::optional<SemIRTextNameRef>;

 private:
  // A name defined within one scope of one output block.
  struct Definition {
    SemIRTextDefinition info;

    // Indices into `specific_values_` of the rows giving this name's value in
    // each `specific` of the generic that defines it.
    llvm::SmallVector<int32_t, 0> specific_values;
  };

  // A `%name => value` row of a `specific` block.
  struct SpecificValue {
    SemIRTextSpecificValue info;

    // Where `info.value` is defined, as an index into `definitions_`, or -1 if
    // the value isn't a single name that resolved unambiguously.
    int32_t value_definition = -1;
  };

  // One `--- file.carbon` section of the output. Sections are independent:
  // each is a separate compilation with its own names, so a name in one never
  // refers to a name in another.
  struct Block {
    // Definitions by scope name, then by name. Indices into `definitions_`.
    llvm::StringMap<llvm::StringMap<int32_t>> scopes;

    // Added to a 1-based SemIR line number to get the 0-based document line of
    // the corresponding source text. `nullopt` if we couldn't find where this
    // block's input file was written, which happens when it came from an
    // `// INCLUDE-FILE` rather than from a `// --- file.carbon` split.
    std::optional<int> source_line_base;
  };

  // A `%name`, or `scope.%name`, written somewhere in the output.
  struct Ref {
    // Position in the document. `refs_` is sorted by these.
    int32_t line;
    int32_t column_begin;
    int32_t column_end;

    // The enclosing `--- file.carbon` section, as an index into `blocks_`.
    int32_t block;

    // The name, without the `%`.
    llvm::StringRef name;

    // The definitions this resolves to, as the range
    // `ref_definitions_[definitions_begin ..][0 .. definitions_size)` of
    // indices into `definitions_`.
    int32_t definitions_begin;
    int32_t definitions_size;
  };

  // A reference found while reading, kept until the whole block has been read
  // so that it can be resolved against names defined later in the block.
  struct PendingRef {
    Ref ref;

    // The scope written before the `%`, or empty if the name was unqualified.
    llvm::StringRef explicit_scope;

    // The scope of the line the name is written on, used to resolve a name
    // that has no `explicit_scope`.
    llvm::StringRef context_scope;

    // Whether the name sits in a type annotation, which is written in the
    // `constants` scope rather than in `context_scope`.
    bool in_type;

    // The specific-value row this name labels, as an index into
    // `specific_values_`, or -1. Set for the `%name` of a `%name => value` row
    // of a `specific` block, which names an instruction of the generic.
    int32_t specific_value = -1;

    // The specific-value row this name is the value of, as an index into
    // `specific_values_`, or -1. Set when a name makes up the whole of the
    // right side of a `%name => value` row, so that resolving the name also
    // tells us where that row's value is defined.
    int32_t value_of_specific = -1;
  };

  // Reads the SemIR of one `--- file.carbon` section, and appends what it
  // finds to the index. `lines` is that section's content lines, excluding the
  // `--- file.carbon` line itself.
  class BlockReader;

  auto FindRef(clang::clangd::Position position) const -> const Ref*;

  std::vector<Block> blocks_;
  std::vector<Definition> definitions_;
  std::vector<SpecificValue> specific_values_;

  // References, sorted by position, so that a lookup is a binary search.
  std::vector<Ref> refs_;

  // The definitions each `Ref` resolves to; see `Ref::definitions_begin`.
  std::vector<int32_t> ref_definitions_;

  // Holds text that we build rather than point at in the document: the
  // `@Entity.WithSelf` scope name of a `!with Self:` region, and the text of a
  // definition that spans more than one line, which has to have each line's
  // `// CHECK:STDOUT:` prefix removed before they can be joined.
  llvm::BumpPtrAllocator allocator_;
  llvm::StringSaver strings_ = llvm::StringSaver(allocator_);
};

}  // namespace Carbon::LanguageServer

#endif  // CARBON_TOOLCHAIN_LANGUAGE_SERVER_SEM_IR_TEXT_H_
