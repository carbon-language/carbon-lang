// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "migrate_cpp/rewriter.h"

#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"

namespace Carbon {

static constexpr const char CppPlaceholder[] = "__cpp__{ ... }";

auto OutputWriter::Write(clang::SourceLocation loc,
                         const OutputSegment& segment) const -> bool {
  return std::visit(
      [&](auto& content) {
        using Type = std::decay_t<decltype(content)>;
        auto [begin, end] = bounds;

        if constexpr (std::is_same_v<Type, std::string>) {
          auto begin_offset = source_manager.getDecomposedLoc(loc).second;
          // Append the string replacement if the node being replaced falls
          // within `bounds`.
          if (begin <= begin_offset && begin_offset < end) {
            output.append(content);
          }
        } else if constexpr (std::is_same_v<Type, clang::DynTypedNode> ||
                             std::is_same_v<Type, clang::TypeLoc>) {
          auto content_loc = content.getSourceRange().getBegin();
          auto begin_offset =
              source_manager.getDecomposedLoc(content_loc).second;
          // If the node we're considering a replacement for is already beyond
          // the region for which we want to make a replacement, exit early
          // declaring that we have completed replacements (by returning false).
          // Otherwise proceed. Note that we do not exit early or skip anything
          // if the node comes before the relevant region. This is because many
          // nodes in Clang's AST have a starting source location but a
          // meaningless end location, and while the start of the segment may
          // not be in the range, as we recurse, sub-segments may indeed end up
          // being printed.
          if (begin_offset >= end) {
            return false;
          }

          if (auto iter = map.find(content); iter == map.end()) {
            output.append(CppPlaceholder);
          } else {
            for (const auto& output_segment : iter->second) {
              if (!Write(content.getSourceRange().getBegin(), output_segment)) {
                return false;
              }
            }
          }
        } else {
          static_assert(std::is_void_v<Type>,
                        "Failed to handle a case in the `std::variant`.");
        }
        return true;
      },
      segment.content_);
}

auto MigrationConsumer::HandleTranslationUnit(clang::ASTContext& context)
    -> void {
  RewriteBuilder rewriter(context, segment_map_);
  rewriter.TraverseAST(context);

  auto translation_unit_node =
      clang::DynTypedNode::create(*context.getTranslationUnitDecl());
  auto iter = segment_map_.find(translation_unit_node);

  if (iter == segment_map_.end()) {
    result_.append(CppPlaceholder);
  } else {
    OutputWriter w{
        .map = segment_map_,
        .bounds = output_range_,
        .source_manager = context.getSourceManager(),
        .output = result_,
    };

    for (const auto& output_segment : iter->second) {
      w.Write(translation_unit_node.getSourceRange().getBegin(),
              output_segment);
    }
  }
}

auto RewriteBuilder::TextFor(clang::SourceLocation begin,
                             clang::SourceLocation end) const
    -> llvm::StringRef {
  auto range = clang::CharSourceRange::getCharRange(begin, end);
  return clang::Lexer::getSourceText(range, context_.getSourceManager(),
                                     context_.getLangOpts());
}

auto RewriteBuilder::TextForTokenAt(clang::SourceLocation loc) const
    -> llvm::StringRef {
  auto& source_manager = context_.getSourceManager();
  auto [file_id, offset] = source_manager.getDecomposedLoc(loc);
  llvm::StringRef file = source_manager.getBufferData(file_id);
  clang::Lexer lexer(source_manager.getLocForStartOfFile(file_id),
                     context_.getLangOpts(), file.begin(), file.data() + offset,
                     file.end());
  clang::Token token;
  lexer.LexFromRawLexer(token);
  return TextFor(loc, loc.getLocWithOffset(token.getLength()));
}

// TODO: The output written in this member function needs to be
// architecture-dependent. Moreover, even if the output is correct in the sense
// that the types match and are interoperable between Carbon and C++, they may
// not be semantically correct: If the C++ code specifies the type `long`, and
// on the platform for which the migration is occurring `long` has 64-bits, we
// may not want to use `i64` as the replacement: The C++ code may be intended to
// operate in environments where `long` is only 32-bits wide. We need to develop
// a strategy for determining builtin-type replacements that addresses these
// issues.
auto RewriteBuilder::VisitBuiltinTypeLoc(clang::BuiltinTypeLoc type_loc)
    -> bool {
  llvm::StringRef content;
  switch (type_loc.getTypePtr()->getKind()) {
    case clang::BuiltinType::Bool:
      content = "bool";
      break;
    case clang::BuiltinType::Char_U:
      content = "char";
      break;
    case clang::BuiltinType::UChar:
      content = "u8";
      break;
    case clang::BuiltinType::UShort:
      content = "u16";
      break;
    case clang::BuiltinType::UInt:
      content = "u32";
      break;
    case clang::BuiltinType::ULong:
      content = "u64";
      break;
    case clang::BuiltinType::ULongLong:
      content = "u64";
      break;
    case clang::BuiltinType::UInt128:
      content = "u128";
      break;
    case clang::BuiltinType::Char_S:
      content = "char";
      break;
    case clang::BuiltinType::SChar:
      content = "i8";
      break;
    case clang::BuiltinType::Short:
      content = "i16";
      break;
    case clang::BuiltinType::Int:
      content = "i32";
      break;
    case clang::BuiltinType::Long:
      content = "i64";
      break;
    case clang::BuiltinType::LongLong:
      content = "i64";
      break;
    case clang::BuiltinType::Int128:
      content = "i128";
      break;
    case clang::BuiltinType::Float:
      content = "f32";
      break;
    case clang::BuiltinType::Double:
      content = "f64";
      break;
    case clang::BuiltinType::Void:
      content = "()";
      break;
    default:
      // In this case we do not know what the output should be so we do not
      // write any.
      return true;
  }
  SetReplacement(type_loc, OutputSegment(content));
  return true;
}

auto RewriteBuilder::VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr* expr)
    -> bool {
  SetReplacement(expr, OutputSegment(expr->getValue() ? "true" : "false"));
  return true;
}

auto RewriteBuilder::VisitDeclRefExpr(clang::DeclRefExpr* expr) -> bool {
  SetReplacement(expr, OutputSegment(TextForTokenAt(expr->getBeginLoc())));
  return true;
}

auto RewriteBuilder::VisitDeclStmt(clang::DeclStmt* stmt) -> bool {
  std::vector<OutputSegment> segments;
  for (clang::Decl* decl : stmt->decls()) {
    segments.push_back(OutputSegment(decl));
    segments.push_back(OutputSegment(";\n"));
  }
  SetReplacement(stmt, std::move(segments));
  return true;
}

auto RewriteBuilder::VisitImplicitCastExpr(clang::ImplicitCastExpr* expr)
    -> bool {
  SetReplacement(expr, OutputSegment(expr->getSubExpr()));
  return true;
}

auto RewriteBuilder::VisitIntegerLiteral(clang::IntegerLiteral* expr) -> bool {
  // TODO: Replace suffixes.
  std::string text(TextForTokenAt(expr->getBeginLoc()));
  for (char& c : text) {
    // Carbon uses underscores for digit separators whereas C++ uses single
    // quotation marks. Convert all `'` to `_`.
    if (c == '\'') {
      c = '_';
    }
  }
  SetReplacement(expr, {OutputSegment(std::move(text))});
  return true;
}

auto RewriteBuilder::VisitParmVarDecl(clang::ParmVarDecl* decl) -> bool {
  llvm::StringRef name = decl->getName();
  std::vector<OutputSegment> segments = {
      OutputSegment(llvm::formatv("{0}: ", name.empty() ? "_" : name.str())),
      OutputSegment(decl->getTypeSourceInfo()->getTypeLoc()),
  };

  if (clang::Expr* init = decl->getInit()) {
    segments.push_back(OutputSegment(" = "));
    segments.push_back(OutputSegment(init));
  }

  SetReplacement(decl, std::move(segments));
  return true;
}

auto RewriteBuilder::VisitPointerTypeLoc(clang::PointerTypeLoc type_loc)
    -> bool {
  SetReplacement(type_loc,
                 {OutputSegment(type_loc.getPointeeLoc()), OutputSegment("*")});
  return true;
}

auto RewriteBuilder::VisitReturnStmt(clang::ReturnStmt* stmt) -> bool {
  SetReplacement(
      stmt, {OutputSegment("return "), OutputSegment(stmt->getRetValue())});
  return true;
}

auto RewriteBuilder::VisitTranslationUnitDecl(clang::TranslationUnitDecl* decl)
    -> bool {
  std::vector<OutputSegment> segments;

  // Clang starts each translation unit with some initial `TypeDefDecl`s that
  // are not part of the written text. We want to skip past these initial
  // declarations, which we do by ignoring any node of type `TypeDefDecl` which
  // has an invalid source location.
  auto iter = decl->decls_begin();
  while (iter != decl->decls_end() && llvm::isa<clang::TypedefDecl>(*iter) &&
         (*iter)->getLocation().isInvalid()) {
    ++iter;
  }

  for (; iter != decl->decls_end(); ++iter) {
    clang::Decl* d = *iter;
    segments.push_back(OutputSegment(d));

    // Function definitions do not need semicolons.
    bool needs_semicolon = !(llvm::isa<clang::FunctionDecl>(d) &&
                             llvm::cast<clang::FunctionDecl>(d)->hasBody());
    segments.push_back(OutputSegment(needs_semicolon ? ";\n" : "\n"));
  }

  SetReplacement(decl, std::move(segments));
  return true;
}

auto RewriteBuilder::VisitUnaryOperator(clang::UnaryOperator* expr) -> bool {
  switch (expr->getOpcode()) {
    case clang::UO_AddrOf:
      SetReplacement(expr,
                     {OutputSegment("&"), OutputSegment(expr->getSubExpr())});
      break;

    default:
      // TODO: Finish implementing cases.
      break;
  }
  return true;
}

auto RewriteBuilder::TraverseFunctionDecl(clang::FunctionDecl* decl) -> bool {
  clang::TypeLoc return_type_loc = decl->getFunctionTypeLoc().getReturnLoc();
  if (!TraverseTypeLoc(return_type_loc)) {
    return false;
  }

  std::vector<OutputSegment> segments;
  segments.push_back(
      OutputSegment(llvm::formatv("fn {0}(", decl->getNameAsString())));

  size_t i = 0;
  for (; i + 1 < decl->getNumParams(); ++i) {
    clang::ParmVarDecl* param = decl->getParamDecl(i);
    if (!TraverseDecl(param)) {
      return false;
    }
    segments.push_back(OutputSegment(param));
    segments.push_back(OutputSegment(", "));
  }

  if (i + 1 == decl->getNumParams()) {
    clang::ParmVarDecl* param = decl->getParamDecl(i);
    if (!TraverseDecl(param)) {
      return false;
    }
    segments.push_back(OutputSegment(param));
  }

  segments.push_back(OutputSegment(") -> "));
  segments.push_back(OutputSegment(return_type_loc));

  if (decl->hasBody()) {
    segments.push_back(OutputSegment(" {\n"));
    auto* stmts = llvm::dyn_cast<clang::CompoundStmt>(decl->getBody());
    for (clang::Stmt* stmt : stmts->body()) {
      if (!TraverseStmt(stmt)) {
        return false;
      }
      segments.push_back(OutputSegment(stmt));
      segments.push_back(OutputSegment(";\n"));
    }
    segments.push_back(OutputSegment("}"));
  }

  SetReplacement(decl, std::move(segments));
  return true;
}

auto RewriteBuilder::TraverseVarDecl(clang::VarDecl* decl) -> bool {
  clang::TypeLoc loc = decl->getTypeSourceInfo()->getTypeLoc();
  if (!TraverseTypeLoc(loc)) {
    return false;
  }

  // TODO: Check storage class. Determine what happens for static local
  // variables.
  bool is_const = decl->getType().isConstQualified();
  std::vector<OutputSegment> segments = {
      OutputSegment(llvm::formatv("{0} {1}: ", is_const ? "let" : "var",
                                  decl->getNameAsString())),
      OutputSegment(decl->getTypeSourceInfo()->getTypeLoc()),
  };

  if (clang::Expr* init = decl->getInit()) {
    if (!TraverseStmt(init)) {
      return false;
    }

    segments.push_back(OutputSegment(" = "));
    segments.push_back(OutputSegment(init));
  }

  SetReplacement(decl, std::move(segments));
  return true;
}

}  // namespace Carbon
