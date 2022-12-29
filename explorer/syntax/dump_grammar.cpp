// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <tree_sitter/api.h>

#include "common/bazel_working_dir.h"
#include "common/check.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"

// Declare the `tree_sitter_json` function, which is
// implemented by the `tree-sitter-json` library.
extern "C" const TSLanguage* tree_sitter_carbon();

// Prints a row and column in source.
auto operator<<(llvm::raw_ostream& out, TSPoint point) -> llvm::raw_ostream& {
  out << point.row + 1 << ":" << point.column + 1;
  return out;
}

// Prints the full parse tree.
auto PrintTree(TSNode root_node) -> void {
  TSTreeCursor cursor = ts_tree_cursor_new(root_node);
  auto delete_cursor =
      llvm::make_scope_exit([&]() { ts_tree_cursor_delete(&cursor); });

  int depth = 0;
  for (;;) {
    TSNode node = ts_tree_cursor_current_node(&cursor);
    const char* type = ts_node_type(node);
    TSPoint start = ts_node_start_point(node);
    TSPoint end = ts_node_end_point(node);
    llvm::outs().indent(2 * depth);
    llvm::outs() << start << " - " << end << ": " << type << "\n";

    if (ts_tree_cursor_goto_first_child(&cursor)) {
      // Went down a level.
      ++depth;
    } else {
      bool no_parent = false;
      while (!ts_tree_cursor_goto_next_sibling(&cursor)) {
        if (ts_tree_cursor_goto_parent(&cursor)) {
          --depth;
        } else {
          no_parent = true;
          break;
        }
      }
      if (no_parent) {
        break;
      }
    }
  }
}

auto main(int argc, char** argv) -> int {
  // Standard binary setup.
  Carbon::SetWorkingDirForBazel();
  llvm::setBugReportMsg(
      "Please report issues to "
      "https://github.com/carbon-language/carbon-lang/issues and include the "
      "crash backtrace.\n");
  llvm::InitLLVM init_llvm(argc, argv);

  // Read the file.
  CARBON_CHECK(argc == 2) << "Syntax: dump_grammar file";
  namespace fs = llvm::sys::fs;
  llvm::Expected<fs::file_t> fd = fs::openNativeFileForRead(argv[1]);
  if (auto err = fd.takeError()) {
    llvm::errs() << "Error opening `" << argv[1]
                 << "`: " << llvm::toString(std::move(err)) << "\n";
    return 1;
  }
  auto close_fd = llvm::make_scope_exit([&] { fs::closeFile(*fd); });
  llvm::SmallString<0> buffer;
  if (llvm::Error err = fs::readNativeFileToEOF(*fd, buffer)) {
    llvm::errs() << "Error reading `" << argv[1]
                 << "`: " << llvm::toString(std::move(err)) << "\n";
    return 1;
  }

  // Load Carbon's parser.
  TSParser* parser = ts_parser_new();
  auto delete_parser =
      llvm::make_scope_exit([&]() { ts_parser_delete(parser); });
  ts_parser_set_language(parser, tree_sitter_carbon());

  // Build the syntax tree.
  TSTree* tree =
      ts_parser_parse_string(parser, nullptr, buffer.c_str(), buffer.size());
  auto delete_tree = llvm::make_scope_exit([&]() { ts_tree_delete(tree); });

  // Print the syntax tree.
  TSNode root_node = ts_tree_root_node(tree);
  PrintTree(root_node);

  return 0;
}
