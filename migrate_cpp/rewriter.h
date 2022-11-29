// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_MIGRATE_CPP_REWRITER_H_
#define CARBON_MIGRATE_CPP_REWRITER_H_

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "llvm/ADT/DenseMap.h"
#include "migrate_cpp/output_segment.h"

namespace Carbon {
namespace Internal {

struct Empty {
  friend auto operator==(Empty /*unused*/, Empty /*unused*/) -> bool {
    return true;
  }
};
struct Tombstone {
  friend auto operator==(Tombstone /*unused*/, Tombstone /*unused*/) -> bool {
    return true;
  }
};

// Type alias for the variant representing any of the values that can be
// written with OutputWriter.
using KeyType =
    std::variant<clang::DynTypedNode, clang::TypeLoc, Empty, Tombstone>;

// `KeyInfo` is used as a template argument to `llvm::DenseMap` to specify how
// to equality-compare and hash `KeyType`.
struct KeyInfo {
  static auto isEqual(const KeyType& lhs, const KeyType& rhs) -> bool {
    return lhs == rhs;
  }
  static auto getHashValue(const KeyType& x) -> unsigned {
    return std::visit(
        [](auto x) -> unsigned {
          using Type = std::decay_t<decltype(x)>;
          if constexpr (std::is_same_v<Type, clang::DynTypedNode>) {
            return clang::DynTypedNode::DenseMapInfo::getHashValue(x);
          } else if constexpr (std::is_same_v<Type, clang::TypeLoc>) {
            // TODO: Improve this.
            return reinterpret_cast<uintptr_t>(x.getTypePtr());
          } else {
            return 0;
          }
        },
        x);
  }

  static auto getEmptyKey() -> KeyType { return Empty{}; }
  static auto getTombstoneKey() -> KeyType { return Tombstone{}; }
};

}  // namespace Internal

// `OutputWriter` is responsible for traversing the tree of `OutputSegment`s
// and writing the correct data to its member `output`.
struct OutputWriter {
  using SegmentMapType =
      llvm::DenseMap<Internal::KeyType, std::vector<OutputSegment>,
                     Internal::KeyInfo>;

  auto Write(clang::SourceLocation loc, const OutputSegment& segment) const
      -> bool;

  const SegmentMapType& map;

  // Bounds represent the offsets into the primary file (multi-file refactorings
  // are not yet supported) that should be output. While primarily this is a
  // mechanism to make testing more robust, it can also be used to make local
  // changes to sections of C++ code.
  std::pair<size_t, size_t> bounds;

  clang::SourceManager& source_manager;
  std::string& output;
};

// `RewriteBuilder` is a recursive AST visitor. For each node, it computes and
// stores a sequence of `OutputSegment`s describing how this node should be
// replaced.
class RewriteBuilder : public clang::RecursiveASTVisitor<RewriteBuilder> {
 public:
  using SegmentMapType = typename OutputWriter::SegmentMapType;

  // Constructs a `RewriteBuilder` which can read the AST from `context` and
  // will write results into `segments`.
  explicit RewriteBuilder(clang::ASTContext& context, SegmentMapType& segments)
      : context_(context), segments_(segments) {}

  // By default, traverse children nodes before their parent. Called by the CRTP
  // base class to determine traversal order.
  auto shouldTraversePostOrder() const -> bool { return true; }

  // Visitor member functions, defining how each node should be processed.
  auto VisitBuiltinTypeLoc(clang::BuiltinTypeLoc type_loc) -> bool;
  auto VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr* expr) -> bool;
  auto VisitDeclRefExpr(clang::DeclRefExpr* expr) -> bool;
  auto VisitDeclStmt(clang::DeclStmt* stmt) -> bool;
  auto VisitImplicitCastExpr(clang::ImplicitCastExpr* expr) -> bool;
  auto VisitIntegerLiteral(clang::IntegerLiteral* expr) -> bool;
  auto VisitParmVarDecl(clang::ParmVarDecl* decl) -> bool;
  auto VisitPointerTypeLoc(clang::PointerTypeLoc type_loc) -> bool;
  auto VisitReturnStmt(clang::ReturnStmt* stmt) -> bool;
  auto VisitTranslationUnitDecl(clang::TranslationUnitDecl* decl) -> bool;
  auto VisitUnaryOperator(clang::UnaryOperator* expr) -> bool;

  auto TraverseFunctionDecl(clang::FunctionDecl* decl) -> bool;
  auto TraverseVarDecl(clang::VarDecl* decl) -> bool;

  auto segments() const -> const SegmentMapType& { return segments_; }
  auto segments() -> SegmentMapType& { return segments_; }

 private:
  // Associates `output_segments` in the output map `this->segments()` with the
  // key `node`, so as to declare that, when output is being written, `node`
  // should be replaced with the sequence of outputs described by
  // `output_segments`.
  auto SetReplacement(clang::DynTypedNode node,
                      std::vector<OutputSegment> output_segments) -> void {
    segments_.try_emplace(node, std::move(output_segments));
  }

  auto SetReplacement(clang::TypeLoc node,
                      std::vector<OutputSegment> output_segments) -> void {
    segments_.try_emplace(node, std::move(output_segments));
  }

  template <typename T>
  auto SetReplacement(const T* node, std::vector<OutputSegment> output_segments)
      -> void {
    segments_.try_emplace(clang::DynTypedNode::create(*node),
                          std::move(output_segments));
  }

  // Invokes the overload of `SetReplacement` defined above. Equivalent to
  // `this->SetReplacement(node, std::vector<OutputSegment>(1, segment))`.
  template <typename T>
  auto SetReplacement(const T* node, OutputSegment segment) -> void {
    std::vector<OutputSegment> node_segments;
    node_segments.push_back(std::move(segment));
    SetReplacement(node, std::move(node_segments));
  }

  auto SetReplacement(clang::TypeLoc type_loc, OutputSegment segment) -> void {
    std::vector<OutputSegment> node_segments;
    node_segments.push_back(std::move(segment));
    SetReplacement(type_loc, std::move(node_segments));
  }

  // Returns a `llvm::StringRef` into the source text corresponding to the
  // half-open interval starting at `begin` (inclusive) and ending at `end`
  // (exclusive).
  auto TextFor(clang::SourceLocation begin, clang::SourceLocation end) const
      -> llvm::StringRef;

  // Returns a `llvm::StringRef` into the source text for the single token
  // located at `loc`.
  auto TextForTokenAt(clang::SourceLocation loc) const -> llvm::StringRef;

  clang::ASTContext& context_;
  SegmentMapType& segments_;
};

// An `ASTConsumer` which, when executed, populates a `std::string` with the
// text of a Carbon source file which is a best approximation of of the
// semantics of the corresponding C++ translation unit defined by the consumed
// AST.
class MigrationConsumer : public clang::ASTConsumer {
 public:
  explicit MigrationConsumer(std::string& result,
                             std::pair<size_t, size_t> output_range)
      : result_(result), output_range_(std::move(output_range)) {}

  auto HandleTranslationUnit(clang::ASTContext& context) -> void override;

 private:
  RewriteBuilder::SegmentMapType segment_map_;
  std::string& result_;
  std::pair<size_t, size_t> output_range_;
};

// An `ASTFrontendAction` which constructs a `MigrationConsumer` and invokes it
// on an AST, populating a `std::string` with the text of a Carbon source file
// which is a best approximation of of the semantics of the corresponding C++
// translation unit defined by the consumed AST.
class MigrationAction : public clang::ASTFrontendAction {
 public:
  // Constructs the `MigrationAction`. The parameter `result` is a reference to
  // the `std::string` where output will be written. Only output corresponding
  // to text at offsets that fall in between `output_range.first` and
  // `output_range.second` will be written.
  explicit MigrationAction(std::string& result,
                           std::pair<size_t, size_t> output_range)
      : result_(result), output_range_(std::move(output_range)) {}

  // Returns a `std::unique_ptr` to a `clang::MigrationConsumer` which populates
  // the output `result`.
  auto CreateASTConsumer(clang::CompilerInstance& /*CI*/,
                         llvm::StringRef /*InFile*/)
      -> std::unique_ptr<clang::ASTConsumer> override {
    return std::make_unique<MigrationConsumer>(result_, output_range_);
  }

 private:
  std::string& result_;
  std::pair<size_t, size_t> output_range_;
};

}  // namespace Carbon

#endif  // CARBON_MIGRATE_CPP_REWRITER_H_
