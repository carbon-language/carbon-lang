// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/declaration.h"

#include "explorer/ast/value.h"
#include "explorer/base/print_as_id.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

using llvm::cast;

Declaration::~Declaration() = default;

void Declaration::Print(llvm::raw_ostream& out) const { PrintIndent(0, out); }
void Declaration::PrintIndent(int indent_num_spaces,
                              llvm::raw_ostream& out) const {
  if (kind() != DeclarationKind::FunctionDeclaration &&
      kind() != DeclarationKind::DestructorDeclaration) {
    out.indent(indent_num_spaces);
  }

  switch (kind()) {
    case DeclarationKind::NamespaceDeclaration:
      out << PrintAsID(*this) << ";";
      break;
    case DeclarationKind::InterfaceDeclaration:
    case DeclarationKind::ConstraintDeclaration: {
      const auto& iface_decl = cast<ConstraintTypeDeclaration>(*this);
      out << PrintAsID(*this);
      out << " {\n";
      for (Nonnull<Declaration*> m : iface_decl.members()) {
        out.indent(indent_num_spaces + 2) << *m << "\n";
      }
      out.indent(indent_num_spaces) << "}";
      break;
    }
    case DeclarationKind::ImplDeclaration: {
      const auto& impl_decl = cast<ImplDeclaration>(*this);
      out << PrintAsID(impl_decl) << " {\n";
      for (Nonnull<Declaration*> m : impl_decl.members()) {
        m->PrintIndent(indent_num_spaces + 2, out);
        out << "\n";
      }
      out.indent(indent_num_spaces) << "}";
      break;
    }
    case DeclarationKind::MatchFirstDeclaration: {
      const auto& match_first_decl = cast<MatchFirstDeclaration>(*this);
      out << PrintAsID(match_first_decl) << " {\n";
      for (Nonnull<const ImplDeclaration*> m :
           match_first_decl.impl_declarations()) {
        m->PrintIndent(indent_num_spaces + 2, out);
        out << "\n";
      }
      out.indent(indent_num_spaces) << "}";
      break;
    }
    case DeclarationKind::FunctionDeclaration:
      cast<FunctionDeclaration>(*this).PrintIndent(indent_num_spaces, out);
      break;
    case DeclarationKind::DestructorDeclaration:
      cast<DestructorDeclaration>(*this).PrintIndent(indent_num_spaces, out);
      break;
    case DeclarationKind::ClassDeclaration: {
      const auto& class_decl = cast<ClassDeclaration>(*this);
      out << PrintAsID(class_decl);
      if (class_decl.type_params().has_value()) {
        out << **class_decl.type_params();
      }
      out << " {\n";
      for (Nonnull<Declaration*> m : class_decl.members()) {
        m->PrintIndent(indent_num_spaces + 2, out);
        out << "\n";
      }
      out.indent(indent_num_spaces) << "}";
      break;
    }
    case DeclarationKind::MixinDeclaration: {
      const auto& mixin_decl = cast<MixinDeclaration>(*this);
      out << PrintAsID(mixin_decl) << "{\n";
      for (Nonnull<Declaration*> m : mixin_decl.members()) {
        m->PrintIndent(indent_num_spaces + 2, out);
        out << "\n";
      }
      out.indent(indent_num_spaces) << "}";
      break;
    }
    case DeclarationKind::MixDeclaration: {
      const auto& mix_decl = cast<MixDeclaration>(*this);
      out << PrintAsID(mix_decl);
      out << mix_decl.mixin() << ";";
      break;
    }
    case DeclarationKind::ChoiceDeclaration: {
      const auto& choice = cast<ChoiceDeclaration>(*this);
      out << PrintAsID(choice) << " {\n";
      for (Nonnull<const AlternativeSignature*> alt : choice.alternatives()) {
        out.indent(indent_num_spaces + 2) << *alt << ";\n";
      }
      out.indent(indent_num_spaces) << "}";
      break;
    }

    case DeclarationKind::VariableDeclaration: {
      const auto& var = cast<VariableDeclaration>(*this);
      out << PrintAsID(var);
      if (var.has_initializer()) {
        out << " = " << var.initializer();
      }
      out << ";";
      break;
    }

    case DeclarationKind::InterfaceExtendDeclaration:
    case DeclarationKind::InterfaceRequireDeclaration:
    case DeclarationKind::AssociatedConstantDeclaration: {
      out << PrintAsID(*this) << ";";
      break;
    }

    case DeclarationKind::SelfDeclaration: {
      out << "Self";
      break;
    }

    case DeclarationKind::AliasDeclaration: {
      const auto& alias = cast<AliasDeclaration>(*this);
      out << PrintAsID(alias) << " = " << alias.target() << ";";
      break;
    }

    case DeclarationKind::ExtendBaseDeclaration: {
      out << PrintAsID(*this) << ";";
      break;
    }
  }
}

void Declaration::PrintID(llvm::raw_ostream& out) const {
  switch (kind()) {
    case DeclarationKind::NamespaceDeclaration:
      out << "namespace " << cast<NamespaceDeclaration>(*this).name();
      break;
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
          out << "extend ";
          break;
        case ImplKind::ExternalImpl:
          break;
      }
      out << "impl ";
      if (!impl_decl.deduced_parameters().empty()) {
        out << "forall [";
        llvm::ListSeparator sep;
        for (const auto* param : impl_decl.deduced_parameters()) {
          out << sep << *param;
        }
        out << "] ";
      }
      if (impl_decl.kind() != ImplKind::InternalImpl) {
        out << *impl_decl.impl_type() << " ";
      }
      out << "as " << impl_decl.interface();
      break;
    }
    case DeclarationKind::MatchFirstDeclaration:
      out << "match_first";
      break;
    case DeclarationKind::FunctionDeclaration:
      out << "fn " << cast<FunctionDeclaration>(*this).name();
      break;
    case DeclarationKind::DestructorDeclaration:
      out << *GetName(*this);
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

    case DeclarationKind::InterfaceExtendDeclaration: {
      const auto& extend = cast<InterfaceExtendDeclaration>(*this);
      out << "extend " << *extend.base();
      break;
    }

    case DeclarationKind::InterfaceRequireDeclaration: {
      const auto& impl = cast<InterfaceRequireDeclaration>(*this);
      out << "require " << *impl.impl_type() << " impls " << *impl.constraint();
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

    case DeclarationKind::ExtendBaseDeclaration: {
      const auto& extend = cast<ExtendBaseDeclaration>(*this);
      out << "extend base: " << *extend.base_class();
      break;
    }
  }
}

void DeclaredName::Print(llvm::raw_ostream& out) const {
  for (const auto& [loc, name] : qualifiers()) {
    out << name << ".";
  }
  out << inner_name();
}

auto GetName(const Declaration& declaration)
    -> std::optional<std::string_view> {
  switch (declaration.kind()) {
    case DeclarationKind::NamespaceDeclaration:
      return cast<NamespaceDeclaration>(declaration).name().inner_name();
    case DeclarationKind::FunctionDeclaration:
      return cast<FunctionDeclaration>(declaration).name().inner_name();
    case DeclarationKind::DestructorDeclaration:
      return "destructor";
    case DeclarationKind::ClassDeclaration:
      return cast<ClassDeclaration>(declaration).name().inner_name();
    case DeclarationKind::MixinDeclaration: {
      return cast<MixinDeclaration>(declaration).name().inner_name();
    }
    case DeclarationKind::MixDeclaration: {
      return std::nullopt;
    }
    case DeclarationKind::ChoiceDeclaration:
      return cast<ChoiceDeclaration>(declaration).name().inner_name();
    case DeclarationKind::InterfaceDeclaration:
    case DeclarationKind::ConstraintDeclaration:
      return cast<ConstraintTypeDeclaration>(declaration).name().inner_name();
    case DeclarationKind::VariableDeclaration:
      return cast<VariableDeclaration>(declaration).binding().name();
    case DeclarationKind::AssociatedConstantDeclaration:
      return cast<AssociatedConstantDeclaration>(declaration).binding().name();
    case DeclarationKind::InterfaceExtendDeclaration:
    case DeclarationKind::InterfaceRequireDeclaration:
    case DeclarationKind::ImplDeclaration:
    case DeclarationKind::MatchFirstDeclaration:
      return std::nullopt;
    case DeclarationKind::SelfDeclaration:
      return SelfDeclaration::name();
    case DeclarationKind::AliasDeclaration: {
      return cast<AliasDeclaration>(declaration).name().inner_name();
    }
    case DeclarationKind::ExtendBaseDeclaration: {
      return "extend base";
    }
  }
}

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
    std::optional<Nonnull<Block*>> body, VirtualOverride virt_override)
    -> ErrorOr<Nonnull<DestructorDeclaration*>> {
  DeducedParameters split_params;
  CARBON_ASSIGN_OR_RETURN(split_params,
                          SplitDeducedParameters(source_loc, deduced_params));
  return arena->New<DestructorDeclaration>(
      source_loc, std::move(split_params.resolved_params),
      split_params.self_pattern, param_pattern, return_term, body,
      virt_override);
}

auto FunctionDeclaration::Create(Nonnull<Arena*> arena,
                                 SourceLocation source_loc, DeclaredName name,
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
      source_loc, std::move(name), std::move(split_params.resolved_params),
      split_params.self_pattern, param_pattern, return_term, body,
      virt_override);
}

void CallableDeclaration::PrintIndent(int indent_num_spaces,
                                      llvm::raw_ostream& out) const {
  auto name = GetName(*this);
  CARBON_CHECK(name) << "Unexpected missing name for `" << *this << "`.";
  out.indent(indent_num_spaces) << "fn " << *name << " ";
  if (!deduced_parameters_.empty() || self_pattern_) {
    out << "[";
    llvm::ListSeparator sep;
    for (Nonnull<const GenericBinding*> deduced : deduced_parameters_) {
      out << sep << *deduced;
    }
    if (self_pattern_) {
      out << sep << **self_pattern_;
    }
    out << "]";
  }
  out << *param_pattern_;
  if (!return_term_.is_omitted()) {
    out << " " << return_term_;
  }
  if (body_) {
    out << "\n";
    (*body_)->PrintIndent(indent_num_spaces, out);
  } else {
    out << ";";
  }
}

ClassDeclaration::ClassDeclaration(CloneContext& context,
                                   const ClassDeclaration& other)
    : Declaration(context, other),
      name_(other.name_),
      extensibility_(other.extensibility_),
      self_decl_(context.Clone(other.self_decl_)),
      type_params_(context.Clone(other.type_params_)),
      members_(context.Clone(other.members_)),
      base_type_(context.Clone(other.base_type_)) {}

ExtendBaseDeclaration::ExtendBaseDeclaration(CloneContext& context,
                                             const ExtendBaseDeclaration& other)
    : Declaration(context, other),
      base_class_(context.Clone(other.base_class_)) {}

ConstraintTypeDeclaration::ConstraintTypeDeclaration(
    CloneContext& context, const ConstraintTypeDeclaration& other)
    : Declaration(context, other),
      name_(other.name_),
      params_(context.Clone(other.params_)),
      self_type_(context.Clone(other.self_type_)),
      self_(context.Clone(other.self_)),
      members_(context.Clone(other.members_)),
      constraint_type_(context.Clone(other.constraint_type_)) {}

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

ImplDeclaration::ImplDeclaration(CloneContext& context,
                                 const ImplDeclaration& other)
    : Declaration(context, other),
      kind_(other.kind_),
      deduced_parameters_(context.Clone(other.deduced_parameters_)),
      impl_type_(context.Clone(other.impl_type_)),
      self_decl_(context.Clone(other.self_decl_)),
      interface_(context.Clone(other.interface_)),
      constraint_type_(context.Clone(other.constraint_type_)),
      members_(context.Clone(other.members_)),
      impl_bindings_(context.Remap(other.impl_bindings_)),
      match_first_(context.Remap(other.match_first_)) {}

void AlternativeSignature::Print(llvm::raw_ostream& out) const {
  out << "alt " << name();
  if (auto params = parameters()) {
    out << **params;
  }
}

void AlternativeSignature::PrintID(llvm::raw_ostream& out) const {
  out << name();
}

auto ChoiceDeclaration::FindAlternative(std::string_view name) const
    -> std::optional<const AlternativeSignature*> {
  for (const auto* alt : alternatives()) {
    if (alt->name() == name) {
      return alt;
    }
  }
  return std::nullopt;
}

MixDeclaration::MixDeclaration(CloneContext& context,
                               const MixDeclaration& other)
    : Declaration(context, other),
      mixin_(context.Clone(other.mixin_)),
      mixin_value_(context.Clone(other.mixin_value_)) {}

}  // namespace Carbon
