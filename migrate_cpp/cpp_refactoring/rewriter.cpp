// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "migrate_cpp/cpp_refactoring/rewriter.h"

#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/Twine.h"

namespace Carbon {

auto OutputWriter::Write(clang::SourceLocation loc,
                         const OutputSegment& segment) const -> bool {
  return std::visit(
      [&](auto& content) {
        using type = std::decay_t<decltype(content)>;
        auto [begin, end] = bounds;

        if constexpr (std::is_same_v<type, std::string>) {
          auto begin_offset = source_manager.getDecomposedLoc(loc).second;
          if (begin <= begin_offset && begin_offset < end) {
            output.append(content);
          }
        } else if constexpr (std::is_same_v<type, clang::DynTypedNode>) {
          auto content_loc = content.getSourceRange().getBegin();
          auto begin_offset =
              source_manager.getDecomposedLoc(content_loc).second;
          if (begin_offset >= end) {
            return false;
          }

          if (auto iter = map.find(content); iter == map.end()) {
            output.append("__cpp__{ ... }");
          } else {
            for (const auto& output_segment : iter->second) {
              if (!Write(content.getSourceRange().getBegin(), output_segment)) {
                return false;
              }
            }
          }
        } else if constexpr (std::is_same_v<type, clang::TypeLoc>) {
          auto content_loc = content.getSourceRange().getBegin();
          auto begin_offset =
              source_manager.getDecomposedLoc(content_loc).second;
          if (begin_offset >= end) {
            return false;
          }

          if (auto iter = map.find(content); iter == map.end()) {
            output.append("__cpp__");
          } else {
            for (const auto& output_segment : iter->second) {
              if (!Write(content.getSourceRange().getBegin(), output_segment)) {
                return false;
              }
            }
          }
        } else {
          static_assert(std::is_void_v<type>);
        }
        return true;
      },
      segment.content);
}

auto MigrationConsumer::HandleTranslationUnit(clang::ASTContext& context)
    -> void {
  RewriteBuilder rewriter(context, segment_map);
  rewriter.TraverseAST(context);

  auto translation_unit_node =
      clang::DynTypedNode::create(*context.getTranslationUnitDecl());
  auto iter = segment_map.find(translation_unit_node);

  if (iter == segment_map.end()) {
    result.append("__cpp__{}");
  } else {
    OutputWriter w{
        .map = segment_map,
        .bounds = output_range,
        .source_manager = context.getSourceManager(),
        .output = result,
    };

    for (const auto& output_segment : iter->second) {
      w.Write(translation_unit_node.getSourceRange().getBegin(),
              output_segment);
    }
  }
}

auto RewriteBuilder::TextFor(clang::SourceLocation begin,
                             clang::SourceLocation end) const
    -> std::string_view {
  auto range = clang::CharSourceRange::getCharRange(begin, end);
  llvm::StringRef s = clang::Lexer::getSourceText(
      range, context.getSourceManager(), context.getLangOpts());
  return std::string_view(s.data(), s.size());
}

auto RewriteBuilder::TextForTokenAt(clang::SourceLocation loc) const
    -> std::string_view {
  auto& source_manager = context.getSourceManager();
  auto [file_id, offset] = source_manager.getDecomposedLoc(loc);
  llvm::StringRef file = source_manager.getBufferData(file_id);
  clang::Lexer lexer(source_manager.getLocForStartOfFile(file_id),
                     context.getLangOpts(), file.begin(), file.data() + offset,
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
  std::string_view content;
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
    default:
      // In this case we do not know what the output should be so we do not
      // write any.
      return true;
  }
  Write(type_loc, OutputSegment::Text(content));
  return true;
}

auto RewriteBuilder::VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr* expr)
    -> bool {
  Write(expr, OutputSegment::Text(expr->getValue() ? "true" : "false"));
  return true;
}

auto RewriteBuilder::VisitDeclRefExpr(clang::DeclRefExpr* expr) -> bool {
  Write(expr, OutputSegment::Text(TextForTokenAt(expr->getBeginLoc())));
  return true;
}

auto RewriteBuilder::VisitDeclStmt(clang::DeclStmt* stmt) -> bool {
  std::vector<OutputSegment> segments;
  for (clang::Decl* decl : stmt->decls()) {
    segments.push_back(OutputSegment::Node(decl));
    segments.push_back(OutputSegment::Text(";\n"));
  }
  Write(stmt, std::move(segments));
  return true;
}

auto RewriteBuilder::VisitIntegerLiteral(clang::IntegerLiteral* expr) -> bool {
  // TODO: Replace suffixes.
  std::string text(TextForTokenAt(expr->getBeginLoc()));
  for (char& c : text) {
    if (c == '\'') {
      c = '_';
    }
  }
  Write(expr, {OutputSegment::Text(std::move(text))});
  return true;
}

auto RewriteBuilder::VisitPointerTypeLoc(clang::PointerTypeLoc type_loc)
    -> bool {
  Write(type_loc, {OutputSegment::TypeLoc(type_loc.getPointeeLoc()),
                   OutputSegment::Text("*")});
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
    segments.push_back(OutputSegment::Node(d));
    segments.push_back(OutputSegment::Text(";\n"));
  }

  Write(decl, std::move(segments));
  return true;
}

auto RewriteBuilder::VisitUnaryOperator(clang::UnaryOperator* expr) -> bool {
  switch (expr->getOpcode()) {
    case clang::UO_AddrOf:
      Write(expr, {OutputSegment::Text("&"),
                   OutputSegment::Node(expr->getSubExpr())});
      break;

    default:
      // TODO: Finish implementing cases.
      break;
  }
  return true;
}

auto RewriteBuilder::VisitVarDecl(clang::VarDecl* decl) -> bool {
  // TODO: Check storage class. Determine what happens for static local
  // variables.

  bool is_const = decl->getType().isConstQualified();
  std::vector<OutputSegment> segments = {
      OutputSegment::Text((llvm::Twine(is_const ? "let " : "var ") +
                           decl->getNameAsString() + ": ")
                              .str()),
      OutputSegment::TypeLoc(decl->getTypeSourceInfo()->getTypeLoc()),
  };

  if (clang::Expr* init = decl->getInit()) {
    segments.push_back(OutputSegment::Text(" = "));
    segments.push_back(OutputSegment::Node(init));
  }

  Write(decl, std::move(segments));
  return true;
}

}  // namespace Carbon
