// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/declaration.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

using llvm::cast;

Declaration::~Declaration() = default;

void Declaration::Print(llvm::raw_ostream& out) const {
  switch (kind()) {
    case DeclarationKind::InterfaceDeclaration:
    case DeclarationKind::ConstraintDeclaration: {
      const auto& iface_decl = cast<ConstraintTypeDeclaration>(*this);
      PrintID(out);
      out << " {\n";
      for (Nonnull<Declaration*> m : iface_decl.members()) {
        out << *m;
      }
      out << "}\n";
      break;
    }
    case DeclarationKind::ImplDeclaration: {
      const auto& impl_decl = cast<ImplDeclaration>(*this);
      PrintID(out);
      out << " {\n";
      for (Nonnull<Declaration*> m : impl_decl.members()) {
        out << *m;
      }
      out << "}\n";
      break;
    }
    case DeclarationKind::MatchFirstDeclaration: {
      const auto& match_first_decl = cast<MatchFirstDeclaration>(*this);
      PrintID(out);
      out << " {\n";
      for (Nonnull<const ImplDeclaration*> m : match_first_decl.impls()) {
        out << *m;
      }
      out << "}\n";
      break;
    }
    case DeclarationKind::FunctionDeclaration:
      cast<FunctionDeclaration>(*this).PrintDepth(-1, out);
      break;
    case DeclarationKind::DestructorDeclaration:
      cast<DestructorDeclaration>(*this).PrintDepth(-1, out);
      break;
    case DeclarationKind::ClassDeclaration: {
      const auto& class_decl = cast<ClassDeclaration>(*this);
      PrintID(out);
      if (class_decl.type_params().has_value()) {
        out << **class_decl.type_params();
      }
      out << " {\n";
      for (Nonnull<Declaration*> m : class_decl.members()) {
        out << *m;
      }
      out << "}\n";
      break;
    }
    case DeclarationKind::MixinDeclaration: {
      const auto& mixin_decl = cast<MixinDeclaration>(*this);
      PrintID(out);
      out << "{\n";
      for (Nonnull<Declaration*> m : mixin_decl.members()) {
        out << *m;
      }
      out << "}\n";
      break;
    }
    case DeclarationKind::MixDeclaration: {
      const auto& mix_decl = cast<MixDeclaration>(*this);
      PrintID(out);
      out << mix_decl.mixin() << ";";
      break;
    }
    case DeclarationKind::ChoiceDeclaration: {
      const auto& choice = cast<ChoiceDeclaration>(*this);
      PrintID(out);
      out << " {\n";
      for (Nonnull<const AlternativeSignature*> alt : choice.alternatives()) {
        out << *alt << ";\n";
      }
      out << "}\n";
      break;
    }

    case DeclarationKind::VariableDeclaration: {
      const auto& var = cast<VariableDeclaration>(*this);
      PrintID(out);
      if (var.has_initializer()) {
        out << " = " << var.initializer();
      }
      out << ";\n";
      break;
    }

    case DeclarationKind::InterfaceExtendsDeclaration:
    case DeclarationKind::InterfaceImplDeclaration:
    case DeclarationKind::AssociatedConstantDeclaration: {
      PrintID(out);
      out << ";\n";
      break;
    }

    case DeclarationKind::SelfDeclaration: {
      out << "Self";
      break;
    }

    case DeclarationKind::AliasDeclaration: {
      const auto& alias = cast<AliasDeclaration>(*this);
      PrintID(out);
      out << " = " << alias.target() << ";\n";
      break;
    }
  }
}

void Declaration::PrintID(llvm::raw_ostream& out) const {
  switch (kind()) {
    case DeclarationKind::InterfaceDeclaration: {
      const auto& iface_decl = cast<InterfaceDeclaration>(*this);
      out << "interface " << iface_decl.name();
      break;
    }
    case DeclarationKind::ConstraintDeclaration: {
      const auto& constraint_decl = cast<ConstraintDeclaration>(*this);
      out << "constraint " << constraint_decl.name();
      break;
    }
    case DeclarationKind::ImplDeclaration: {
      const auto& impl_decl = cast<ImplDeclaration>(*this);
      switch (impl_decl.kind()) {
        case ImplKind::InternalImpl:
          break;
        case ImplKind::ExternalImpl:
          out << "external ";
          break;
      }
      out << "impl " << *impl_decl.impl_type() << " as "
          << impl_decl.interface();
      break;
    }
    case DeclarationKind::MatchFirstDeclaration:
      out << "match_first";
      break;
    case DeclarationKind::FunctionDeclaration:
      out << "fn " << cast<FunctionDeclaration>(*this).name();
      break;
    case DeclarationKind::DestructorDeclaration:
      out << cast<DestructorDeclaration>(*this).name();
      break;
    case DeclarationKind::ClassDeclaration: {
      const auto& class_decl = cast<ClassDeclaration>(*this);
      out << "class " << class_decl.name();
      break;
    }
    case DeclarationKind::MixinDeclaration: {
      const auto& mixin_decl = cast<MixinDeclaration>(*this);
      out << "__mixin " << mixin_decl.name();
      if (mixin_decl.self()->type().kind() != ExpressionKind::TypeTypeLiteral) {
        out << " for " << mixin_decl.self()->type();
      }
      break;
    }
    case DeclarationKind::MixDeclaration: {
      out << "__mix ";
      break;
    }
    case DeclarationKind::ChoiceDeclaration: {
      const auto& choice = cast<ChoiceDeclaration>(*this);
      out << "choice " << choice.name();
      break;
    }

    case DeclarationKind::VariableDeclaration: {
      const auto& var = cast<VariableDeclaration>(*this);
      out << "var " << var.binding();
      break;
    }

    case DeclarationKind::InterfaceExtendsDeclaration: {
      const auto& extends = cast<InterfaceExtendsDeclaration>(*this);
      out << "extends " << *extends.base();
      break;
    }

    case DeclarationKind::InterfaceImplDeclaration: {
      const auto& impl = cast<InterfaceImplDeclaration>(*this);
      out << "impl " << *impl.impl_type() << " as " << *impl.constraint();
      break;
    }

    case DeclarationKind::AssociatedConstantDeclaration: {
      const auto& let = cast<AssociatedConstantDeclaration>(*this);
      out << "let " << let.binding();
      break;
    }

    case DeclarationKind::SelfDeclaration: {
      out << "Self";
      break;
    }

    case DeclarationKind::AliasDeclaration: {
      const auto& alias = cast<AliasDeclaration>(*this);
      out << "alias " << alias.name();
      break;
    }
  }
}

auto GetName(const Declaration& declaration)
    -> std::optional<std::string_view> {
  switch (declaration.kind()) {
    case DeclarationKind::FunctionDeclaration:
      return cast<FunctionDeclaration>(declaration).name();
    case DeclarationKind::DestructorDeclaration:
      return cast<DestructorDeclaration>(declaration).name();
    case DeclarationKind::ClassDeclaration:
      return cast<ClassDeclaration>(declaration).name();
    case DeclarationKind::MixinDeclaration: {
      return cast<MixinDeclaration>(declaration).name();
    }
    case DeclarationKind::MixDeclaration: {
      return std::nullopt;
    }
    case DeclarationKind::ChoiceDeclaration:
      return cast<ChoiceDeclaration>(declaration).name();
    case DeclarationKind::InterfaceDeclaration:
    case DeclarationKind::ConstraintDeclaration:
      return cast<ConstraintTypeDeclaration>(declaration).name();
    case DeclarationKind::VariableDeclaration:
      return cast<VariableDeclaration>(declaration).binding().name();
    case DeclarationKind::AssociatedConstantDeclaration:
      return cast<AssociatedConstantDeclaration>(declaration).binding().name();
    case DeclarationKind::InterfaceExtendsDeclaration:
    case DeclarationKind::InterfaceImplDeclaration:
    case DeclarationKind::ImplDeclaration:
    case DeclarationKind::MatchFirstDeclaration:
      return std::nullopt;
    case DeclarationKind::SelfDeclaration:
      return SelfDeclaration::name();
    case DeclarationKind::AliasDeclaration: {
      return cast<AliasDeclaration>(declaration).name();
    }
  }
}

void GenericBinding::Print(llvm::raw_ostream& out) const {
  out << name() << ":! " << type();
}

void GenericBinding::PrintID(llvm::raw_ostream& out) const { out << name(); }

void ReturnTerm::Print(llvm::raw_ostream& out) const {
  switch (kind_) {
    case ReturnKind::Omitted:
      return;
    case ReturnKind::Auto:
      out << "-> auto";
      return;
    case ReturnKind::Expression:
      CARBON_CHECK(type_expression_.has_value());
      out << "-> " << **type_expression_;
      return;
  }
}

namespace {

// The deduced parameters of a function declaration.
struct DeducedParameters {
  // The `self` parameter, if any.
  std::optional<Nonnull<Pattern*>> self_pattern;

  // All other deduced parameters.
  std::vector<Nonnull<GenericBinding*>> resolved_params;
};

// Split the `self` pattern (if any) out of `deduced_params`.
auto SplitDeducedParameters(
    SourceLocation source_loc,
    const std::vector<Nonnull<AstNode*>>& deduced_params)
    -> ErrorOr<DeducedParameters> {
  DeducedParameters result;
  for (Nonnull<AstNode*> param : deduced_params) {
    switch (param->kind()) {
      case AstNodeKind::GenericBinding:
        result.resolved_params.push_back(&cast<GenericBinding>(*param));
        break;
      case AstNodeKind::BindingPattern: {
        Nonnull<BindingPattern*> binding = &cast<BindingPattern>(*param);
        if (binding->name() != "self") {
          return ProgramError(source_loc)
                 << "illegal binding pattern in implicit parameter list";
        }
        if (result.self_pattern.has_value()) {
          return ProgramError(source_loc)
                 << "parameter list cannot contain more than one `self` "
                    "parameter";
        }
        result.self_pattern = binding;
        break;
      }
      case AstNodeKind::AddrPattern: {
        Nonnull<AddrPattern*> addr_pattern = &cast<AddrPattern>(*param);
        Nonnull<BindingPattern*> binding =
            &cast<BindingPattern>(addr_pattern->binding());
        if (binding->name() != "self") {
          return ProgramError(source_loc)
                 << "illegal binding pattern in implicit parameter list";
        }
        if (result.self_pattern.has_value()) {
          return ProgramError(source_loc)
                 << "parameter list cannot contain more than one `self` "
                    "parameter";
        }
        result.self_pattern = addr_pattern;
        break;
      }
      default:
        return ProgramError(source_loc)
               << "illegal AST node in implicit parameter list";
    }
  }
  return result;
}
}  // namespace

auto DestructorDeclaration::CreateDestructor(
    Nonnull<Arena*> arena, SourceLocation source_loc,
    std::vector<Nonnull<AstNode*>> deduced_params,
    Nonnull<TuplePattern*> param_pattern, ReturnTerm return_term,
    std::optional<Nonnull<Block*>> body)
    -> ErrorOr<Nonnull<DestructorDeclaration*>> {
  DeducedParameters split_params;
  CARBON_ASSIGN_OR_RETURN(split_params,
                          SplitDeducedParameters(source_loc, deduced_params));
  return arena->New<DestructorDeclaration>(
      source_loc, std::move(split_params.resolved_params),
      split_params.self_pattern, param_pattern, return_term, body);
}

auto FunctionDeclaration::Create(Nonnull<Arena*> arena,
                                 SourceLocation source_loc, std::string name,
                                 std::vector<Nonnull<AstNode*>> deduced_params,
                                 Nonnull<TuplePattern*> param_pattern,
                                 ReturnTerm return_term,
                                 std::optional<Nonnull<Block*>> body,
                                 VirtualOverride virt_override)
    -> ErrorOr<Nonnull<FunctionDeclaration*>> {
  DeducedParameters split_params;
  CARBON_ASSIGN_OR_RETURN(split_params,
                          SplitDeducedParameters(source_loc, deduced_params));
  return arena->New<FunctionDeclaration>(
      source_loc, name, std::move(split_params.resolved_params),
      split_params.self_pattern, param_pattern, return_term, body,
      virt_override);
}

void CallableDeclaration::PrintDepth(int depth, llvm::raw_ostream& out) const {
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

auto ImplDeclaration::Create(Nonnull<Arena*> arena, SourceLocation source_loc,
                             ImplKind kind, Nonnull<Expression*> impl_type,
                             Nonnull<Expression*> interface,
                             std::vector<Nonnull<AstNode*>> deduced_params,
                             std::vector<Nonnull<Declaration*>> members)
    -> ErrorOr<Nonnull<ImplDeclaration*>> {
  std::vector<Nonnull<GenericBinding*>> resolved_params;
  for (Nonnull<AstNode*> param : deduced_params) {
    switch (param->kind()) {
      case AstNodeKind::GenericBinding:
        resolved_params.push_back(&cast<GenericBinding>(*param));
        break;
      default:
        return ProgramError(source_loc)
               << "illegal AST node in implicit parameter list of impl";
    }
  }
  Nonnull<SelfDeclaration*> self_decl =
      arena->New<SelfDeclaration>(impl_type->source_loc());
  return arena->New<ImplDeclaration>(source_loc, kind, impl_type, self_decl,
                                     interface, resolved_params, members);
}

void AlternativeSignature::Print(llvm::raw_ostream& out) const {
  out << "alt " << name() << " " << signature();
}

void AlternativeSignature::PrintID(llvm::raw_ostream& out) const {
  out << name();
}

}  // namespace Carbon
