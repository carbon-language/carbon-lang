// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_STRING_HELPERS_H_
#define CARBON_COMMON_STRING_HELPERS_H_

#include <optional>
#include <string>

#include "common/error.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"

namespace Carbon {

// Note llvm StringExtras has significant functionality which is intended to be
// complementary to this.

// Unescapes Carbon escape sequences in the source string. Returns std::nullopt
// on bad input. `is_block_string` enables escaping unique to block string
// literals, such as \<newline>.
auto UnescapeStringLiteral(llvm::StringRef source, int hashtag_num = 0,
                           bool is_block_string = false)
    -> std::optional<std::string>;

// Parses a block string literal in `source`.
auto ParseBlockStringLiteral(llvm::StringRef source, int hashtag_num = 0)
    -> ErrorOr<std::string>;

// Returns true if the pointer is in the string ref (including equality with
// `ref.end()`). This should be used instead of `<=` comparisons for
// correctness.
auto StringRefContainsPointer(llvm::StringRef ref, const char* ptr) -> bool;

// Converts `tool_path` and each of the `args` into C-strings and returns the
// results. This is intended for use with APIs that expect `argv`-like command
// line argument lists.
//
// Accepts a `cstr_arg_storage` that will provide the underlying storage for
// the C-strings, and returns a small vector of the C-string pointers. The
// returned small vector uses a large small size to allow most common command
// lines to avoid extra allocations and growth passes.
auto BuildCStrArgs(llvm::StringRef tool_path,
                   llvm::ArrayRef<llvm::StringRef> args,
                   llvm::BumpPtrAllocator& alloc)
    -> llvm::SmallVector<const char*, 64>;

// An overload of `BuildCStrArgs` with the same core behavior as the above, but
// with an extra series of `prefix_args` that are placed between the `tool_path`
// and the `args` in the resulting list.
//
// Unlike the tool path and the main `args`, the `prefix_args` are accepted as
// an array of `std::string`s and those string object's `c_str()` method is used
// to get the underlying C-strings to include in the result. This is because
// callers with prefix arguments regularly need to provide dedicated storage for
// these arguments anyways and we can efficiently reuse that. In contrast, the
// `args` are often pulled from an existing `llvm::StringRef` that may never
// exist as a valid C-string and so we need to rebuild those using the storage.
auto BuildCStrArgs(llvm::StringRef tool_path,
                   llvm::ArrayRef<std::string> prefix_args,
                   llvm::ArrayRef<llvm::StringRef> args,
                   llvm::BumpPtrAllocator& alloc)
    -> llvm::SmallVector<const char*, 64>;

}  // namespace Carbon

#endif  // CARBON_COMMON_STRING_HELPERS_H_
