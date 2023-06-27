// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <tree_sitter/api.h>
#include <tree_sitter/parser.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

extern "C" {
TSLanguage* tree_sitter_carbon();
}

// Reads a file to string.
static auto ReadFile(std::filesystem::path path) -> std::string {
  std::ifstream file(path);
  std::stringstream buffer;
  buffer << file.rdbuf();
  file.close();
  return buffer.str();
}

auto main(int argc, char** argv) -> int {
  if (argc != 2) {
    std::cerr << "Usage: treesitter_carbon_tester <file>\n";
    return 2;
  }

  std::string str = ReadFile(std::string(argv[1]));

  auto* parser = ts_parser_new();
  ts_parser_set_language(parser, tree_sitter_carbon());

  auto* tree = ts_parser_parse_string(parser, nullptr, str.c_str(), str.size());

  auto root = ts_tree_root_node(tree);
  char* node_debug = ts_node_string(root);

  std::cout << node_debug;
  auto has_error = ts_node_has_error(root);
  free(node_debug);
  ts_tree_delete(tree);
  ts_parser_delete(parser);
  if (has_error) {
    return 1;
  }
}
