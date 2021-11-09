// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/declaration.h"

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
      const ClassDefinition& class_def =
          cast<ClassDeclaration>(*this).definition();
      out << "class " << class_def.name() << " {\n";
      for (Nonnull<Member*> m : class_def.members()) {
        out << *m;
      }
      out << "}\n";
      break;
    }

    case DeclarationKind::ChoiceDeclaration: {
      const auto& choice = cast<ChoiceDeclaration>(*this);
      out << "choice " << choice.name() << " {\n";
      for (Nonnull<const AlternativeSignature*> alt : choice.alternatives()) {
        out << "alt " << alt->name() << " " << alt->signature() << ";\n";
      }
      out << "}\n";
      break;
    }

    case DeclarationKind::VariableDeclaration: {
      const auto& var = cast<VariableDeclaration>(*this);
      out << "var " << var.binding() << " = " << var.initializer() << "\n";
      break;
    }
  }
}

void FunctionDeclaration::PrintDepth(int depth, llvm::raw_ostream& out) const {
  out << "fn " << name_ << " ";
  if (!deduced_parameters_.empty()) {
    out << "[";
    unsigned int i = 0;
    for (Nonnull<const GenericBinding*> deduced : deduced_parameters_) {
      if (i != 0) {
        out << ", ";
      }
      out << deduced->name() << ":! ";
      deduced->type().Print(out);
      ++i;
    }
    out << "]";
  }
  out << *param_pattern_;
  if (!is_omitted_return_type_) {
    out << " -> " << *return_type_;
  }
  if (body_) {
    out << " {\n";
    (*body_)->PrintDepth(depth, out);
    out << "\n}\n";
  } else {
    out << ";\n";
  }
}

}  // namespace Carbon
