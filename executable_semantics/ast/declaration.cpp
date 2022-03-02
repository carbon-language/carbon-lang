// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/declaration.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

using llvm::cast;

Declaration::~Declaration() = default;

void Declaration::Print(llvm::raw_ostream& out) const {
  switch (kind()) {
    case DeclarationKind::FunctionDeclaration:
      cast<FunctionDeclaration>(*this).PrintDepth(-1, out);
      break;

    case DeclarationKind::ClassDeclaration: {
      const auto& class_decl = cast<ClassDeclaration>(*this);
      out << "class " << class_decl.name() << " {\n";
      for (Nonnull<Declaration*> m : class_decl.members()) {
        out << *m;
      }
      out << "}\n";
      break;
    }

    case DeclarationKind::ChoiceDeclaration: {
      const auto& choice = cast<ChoiceDeclaration>(*this);
      out << "choice " << choice.name() << " {\n";
      for (Nonnull<const AlternativeSignature*> alt : choice.alternatives()) {
        out << *alt << ";\n";
      }
      out << "}\n";
      break;
    }

    case DeclarationKind::VariableDeclaration: {
      const auto& var = cast<VariableDeclaration>(*this);
      out << "var " << var.binding();
      if (var.has_initializer()) {
        out << " = " << var.initializer();
      }
      out << ";\n";
      break;
    }
  }
}

void GenericBinding::Print(llvm::raw_ostream& out) const {
  out << name() << ":! " << type();
}

void ReturnTerm::Print(llvm::raw_ostream& out) const {
  switch (kind_) {
    case ReturnKind::Omitted:
      return;
    case ReturnKind::Auto:
      out << "-> auto";
      return;
    case ReturnKind::Expression:
      out << "-> " << **type_expression_;
      return;
  }
}

// Look for the `me` parameter in the `deduced_parameters_`
// and put it in the `me_pattern_`.
void FunctionDeclaration::ResolveDeducedAndReceiver(
    const std::vector<Nonnull<AstNode*>>& deduced_params) {
  for (Nonnull<AstNode*> param : deduced_params) {
    switch (param->kind()) {
      case AstNodeKind::GenericBinding:
        deduced_parameters_.push_back(&cast<GenericBinding>(*param));
        break;
      case AstNodeKind::BindingPattern: {
        Nonnull<BindingPattern*> bp = &cast<BindingPattern>(*param);
        if (me_pattern_.has_value() || bp->name() != "me") {
          FATAL_COMPILATION_ERROR(source_loc())
              << "illegal binding pattern in implicit parameter list";
        }
        me_pattern_ = bp;
        break;
      }
      default:
        FATAL_COMPILATION_ERROR(source_loc())
            << "illegal AST node in implicit parameter list";
    }
  }
}

void FunctionDeclaration::PrintDepth(int depth, llvm::raw_ostream& out) const {
  out << "fn " << name_ << " ";
  if (!deduced_parameters_.empty()) {
    out << "[";
    llvm::ListSeparator sep;
    for (Nonnull<const GenericBinding*> deduced : deduced_parameters_) {
      out << sep << *deduced;
    }
    out << "]";
  }
  out << *param_pattern_ << return_term_;
  if (body_) {
    out << " {\n";
    (*body_)->PrintDepth(depth, out);
    out << "\n}\n";
  } else {
    out << ";\n";
  }
}

void AlternativeSignature::Print(llvm::raw_ostream& out) const {
  out << "alt " << name() << " " << signature();
}

}  // namespace Carbon
