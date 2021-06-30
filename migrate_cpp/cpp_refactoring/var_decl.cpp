// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "migrate_cpp/cpp_refactoring/var_decl.h"

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

namespace cam = ::clang::ast_matchers;

namespace Carbon {

VarDecl::VarDecl(std::map<std::string, Replacements>& in_replacements,
                 cam::MatchFinder* finder)
    : Matcher(in_replacements) {
  // Skip range-based for loops in this var-inserter.
  finder->addMatcher(cam::varDecl(cam::unless(cam::hasParent(cam::declStmt(
                                      cam::hasParent(cam::cxxForRangeStmt())))))
                         .bind(Label),
                     this);
}

// Helper function for printing TypeLocClass. Useful for debugging.
LLVM_ATTRIBUTE_UNUSED
static auto TypeLocClassToString(clang::TypeLoc::TypeLocClass c)
    -> std::string {
  switch (c) {
    // Mirrors the definition in clang/AST/TypeLoc.h in order to print names.
#define ABSTRACT_TYPE(Class, Base)
#define TYPE(Class, Base)     \
  case clang::TypeLoc::Class: \
    return #Class;
#include "clang/AST/TypeNodes.inc"
    case clang::TypeLoc::Qualified:
      return "Qualified";
  }
}

// Returns a string for the type.
static auto GetTypeStr(const clang::VarDecl* decl,
                       const clang::SourceManager& sm,
                       const clang::LangOptions& lang_opts) -> std::string {
  // Built a vector of class information, because we'll be traversing reverse
  // order to construct the final type.
  auto type_loc = decl->getTypeSourceInfo()->getTypeLoc();
  std::vector<std::pair<clang::TypeLoc::TypeLocClass, std::string>> segments;
  while (!type_loc.isNull()) {
    std::string text;
    auto qualifiers = type_loc.getType().getLocalQualifiers();
    std::string qual_str;
    if (!qualifiers.empty()) {
      qual_str = qualifiers.getAsString();
    }
    auto range =
        clang::CharSourceRange::getTokenRange(type_loc.getLocalSourceRange());
    std::string range_str =
        clang::Lexer::getSourceText(range, sm, lang_opts).str();

    // Make a list of segments with their TypeLocClass for reconstruction of the
    // string. Locally, we will have a qualifier (such as `const`) and a type
    // string (such as `int`) which is also used.
    auto type_loc_class = type_loc.getTypeLocClass();
    if (qual_str.empty()) {
      segments.push_back({type_loc_class, range_str});
    } else if (range_str.empty()) {
      segments.push_back({type_loc_class, qual_str});
    } else {
      segments.push_back({type_loc_class, qual_str + " " + range_str});
    }

    type_loc = type_loc.getNextTypeLoc();
  }

  // Construct the final type based on the class of each step. This reverses to
  // start from the "inside" of the type and go "out" when constructing
  // type_str.
  std::string type_str;
  auto prev_class = clang::TypeLoc::Auto;  // Placeholder class, used in loop.
  for (const auto& [type_loc_class, text] : llvm::reverse(segments)) {
    switch (type_loc_class) {
      case clang::TypeLoc::Elaborated:
        type_str.insert(0, text);
        break;
      case clang::TypeLoc::Qualified:
        if (prev_class == clang::TypeLoc::Pointer) {
          type_str += " " + text;
        } else {
          if (!type_str.empty()) {
            type_str.insert(0, " ");
          }
          type_str.insert(0, text);
        }
        break;
      default:
        type_str += text;
        break;
    }
    prev_class = type_loc_class;
  }
  return type_str;
}

void VarDecl::run(const cam::MatchFinder::MatchResult& result) {
  const auto* decl = result.Nodes.getNodeAs<clang::VarDecl>(Label);
  if (!decl) {
    llvm::report_fatal_error(std::string("getNodeAs failed for ") + Label);
  }

  if (decl->getTypeSourceInfo() == nullptr) {
    // TODO: Need to understand what's happening in this case. Not sure if we
    // need to address it.
    return;
  }

  auto& sm = *(result.SourceManager);
  auto lang_opts = result.Context->getLangOpts();

  std::string after;
  if (decl->getType().isConstQualified()) {
    after = "let ";
  } else if (result.Nodes.getNodeAs<clang::ParmVarDecl>(Label) == nullptr) {
    // Start the replacement with "var" unless it's a parameter.
    after = "var ";
  }
  // Add "identifier: type" to the replacement.
  after += decl->getNameAsString() + ": " + GetTypeStr(decl, sm, lang_opts);

  // This decides the range to replace. Normally the entire decl is replaced,
  // but for code like `int i, j` we need to detect the comma between the
  // declared names. That case currently results in `var i: int, var j: int`.
  // If there's a comma, this range will be non-empty.
  auto type_loc = decl->getTypeSourceInfo()->getTypeLoc();
  auto after_type_loc =
      clang::Lexer::getLocForEndOfToken(type_loc.getEndLoc(), 0, sm, lang_opts);
  auto comma_source_text = clang::Lexer::getSourceText(
      clang::CharSourceRange::getCharRange(after_type_loc, decl->getLocation()),
      sm, lang_opts);
  bool has_comma = !comma_source_text.trim().empty();
  clang::CharSourceRange replace_range = clang::CharSourceRange::getTokenRange(
      has_comma ? decl->getLocation() : decl->getBeginLoc(), decl->getEndLoc());

  AddReplacement(sm, replace_range, after);
}

}  // namespace Carbon
