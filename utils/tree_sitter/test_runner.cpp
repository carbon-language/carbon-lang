// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <tree_sitter/api.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "testing/base/file_helpers.h"

extern "C" {
auto tree_sitter_carbon() -> TSLanguage*;
}

namespace {

// A portion of a test file to parse as a Carbon source file.
struct Split {
  // The name of the split, or empty if the file has no splits.
  std::string_view name;
  // The content of the split.
  std::string_view content;
  // The 0-based line number in the test file at which the content starts.
  int start_line;
};

// Returns the portion of `str` after any leading whitespace.
auto TrimLeft(std::string_view str) -> std::string_view {
  auto pos = str.find_first_not_of(" \t");
  return pos == std::string_view::npos ? std::string_view() : str.substr(pos);
}

// Returns `str` without any trailing whitespace.
auto TrimRight(std::string_view str) -> std::string_view {
  auto pos = str.find_last_not_of(" \t\r");
  return pos == std::string_view::npos ? std::string_view()
                                       : str.substr(0, pos + 1);
}

// Returns the final component of a `/`-separated path.
auto Basename(std::string_view path) -> std::string_view {
  auto pos = path.rfind('/');
  return pos == std::string_view::npos ? path : path.substr(pos + 1);
}

// Divides a test file into splits, following file_test's `// --- <filename>`
// convention. If the file has no splits, the whole file is a single split with
// an empty name. When there are splits, content before the first split (which
// can only be comments) is discarded.
auto GetSplits(std::string_view source) -> std::vector<Split> {
  std::vector<Split> splits;
  Split current = {.name = "", .content = source, .start_line = 0};
  bool has_splits = false;
  size_t current_start = 0;
  int line_number = 0;
  size_t pos = 0;
  while (pos < source.size()) {
    size_t line_end = source.find('\n', pos);
    size_t next =
        line_end == std::string_view::npos ? source.size() : line_end + 1;
    std::string_view line = TrimLeft(source.substr(pos, next - pos));
    if (line.starts_with("// ---")) {
      if (has_splits) {
        current.content = source.substr(current_start, pos - current_start);
        splits.push_back(current);
      }
      has_splits = true;
      line.remove_prefix(std::string_view("// ---").size());
      current = {.name = TrimRight(TrimLeft(line.substr(0, line.find('\n')))),
                 .content = {},
                 .start_line = line_number + 1};
      current_start = next;
    }
    pos = next;
    ++line_number;
  }
  if (has_splits) {
    current.content = source.substr(current_start);
  }
  splits.push_back(current);
  return splits;
}

// Returns whether the given split should be parsed. In normal mode, splits
// that the toolchain expects to fail are skipped, along with non-Carbon splits
// such as C++ headers or `STDIN`.
auto ShouldParseSplit(const Split& split, bool fail_tests) -> bool {
  if (split.name.empty()) {
    return true;
  }
  if (!split.name.ends_with(".carbon")) {
    return false;
  }
  return fail_tests || !Basename(split.name).starts_with("fail_");
}

// Finds the first node in the tree that is an error or is missing.
auto FindFirstError(TSNode node) -> TSNode {
  while (!ts_node_is_error(node) && !ts_node_is_missing(node)) {
    uint32_t count = ts_node_child_count(node);
    bool found_child = false;
    for (uint32_t i = 0; i < count; ++i) {
      TSNode child = ts_node_child(node, i);
      if (ts_node_has_error(child)) {
        node = child;
        found_child = true;
        break;
      }
    }
    if (!found_child) {
      break;
    }
  }
  return node;
}

// Returns the given 0-based line of `content`.
auto GetLine(std::string_view content, uint32_t line) -> std::string_view {
  size_t pos = 0;
  for (uint32_t i = 0; i < line && pos != std::string_view::npos; ++i) {
    pos = content.find('\n', pos);
    if (pos != std::string_view::npos) {
      ++pos;
    }
  }
  if (pos == std::string_view::npos || pos >= content.size()) {
    return {};
  }
  return content.substr(pos, content.find('\n', pos) - pos);
}

}  // namespace

// TODO: use file_test.cpp
auto main(int argc, char** argv) -> int {
  if (argc < 2) {
    std::cerr << "Usage: test_runner <file>...\n";
    return 2;
  }

  auto* parser = ts_parser_new();
  ts_parser_set_language(parser, tree_sitter_carbon());

  // In FAIL_TESTS mode, every file and split is expected to fail to parse.
  // Otherwise, files and splits named `fail_*` are skipped, and everything
  // else is expected to parse successfully.
  bool fail_tests = std::getenv("FAIL_TESTS") != nullptr;
  // When set, print the parse tree for every file.
  bool verbose = std::getenv("VERBOSE") != nullptr;

  std::vector<std::string> incorrect;
  int num_parsed = 0;
  for (int i = 1; i < argc; i++) {
    std::string file_path = argv[i];
    if (!file_path.ends_with(".carbon")) {
      continue;
    }
    if (!fail_tests && Basename(file_path).starts_with("fail_")) {
      continue;
    }
    std::string source = std::move(*Carbon::Testing::ReadFile(file_path));

    for (const auto& split : GetSplits(source)) {
      if (!ShouldParseSplit(split, fail_tests)) {
        continue;
      }
      ++num_parsed;

      std::string name = file_path;
      if (!split.name.empty()) {
        name += " (";
        name += split.name;
        name += ')';
      }

      auto* tree = ts_parser_parse_string(parser, nullptr, split.content.data(),
                                          split.content.size());
      auto root = ts_tree_root_node(tree);
      bool has_error = ts_node_has_error(root);

      if (verbose || (has_error && !fail_tests)) {
        char* node_debug = ts_node_string(root);
        std::cout << name << ":\n" << node_debug << "\n";
        free(node_debug);
      }

      if (has_error && !fail_tests) {
        auto error = FindFirstError(root);
        auto point = ts_node_start_point(error);
        std::cout << file_path << ":" << (split.start_line + point.row + 1)
                  << ":" << (point.column + 1) << ": parse error"
                  << (ts_node_is_missing(error) ? " (missing node)" : "")
                  << "\n"
                  << GetLine(split.content, point.row) << "\n"
                  << std::string(point.column, ' ') << "^\n";
      }

      if (has_error != fail_tests) {
        incorrect.push_back(name);
      }

      ts_tree_delete(tree);
    }
  }
  ts_parser_delete(parser);

  for (const auto& file : incorrect) {
    if (fail_tests) {
      std::cout << "INCORRECTLY PASSING " << file << "\n";
    } else {
      std::cout << "FAILED " << file << "\n";
    }
  }
  if (!incorrect.empty()) {
    if (fail_tests) {
      std::cout << incorrect.size() << " of " << num_parsed
                << " tests incorrectly passing.\n";
    } else {
      std::cout << incorrect.size() << " of " << num_parsed
                << " tests failing.\n";
    }
    return 1;
  }
  std::cout << "All " << num_parsed << " tests passed.\n";
  return 0;
}
