// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/value.h"

#include <algorithm>
#include <optional>
#include <string_view>

#include "common/check.h"
#include "common/error.h"
#include "explorer/ast/declaration.h"
#include "explorer/ast/element.h"
#include "explorer/common/arena.h"
#include "explorer/common/error_builders.h"
#include "explorer/interpreter/action.h"
#include "explorer/interpreter/element_path.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

namespace Carbon {

using llvm::cast;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::isa;

auto StructValue::FindField(std::string_view name) const
    -> std::optional<Nonnull<const Value*>> {
  for (const NamedValue& element : elements_) {
    if (element.name == name) {
      return element.value;
    }
  }
  return std::nullopt;
}

NominalClassValue::NominalClassValue(
    Nonnull<const Value*> type, Nonnull<const Value*> inits,
    std::optional<Nonnull<const NominalClassValue*>> base,
    Nonnull<const NominalClassValue** const> class_value_ptr)
    : Value(Kind::NominalClassValue),
      type_(type),
      inits_(inits),
      base_(base),
      class_value_ptr_(class_value_ptr) {
  // Update ancestors's class value to point to latest child.
  *class_value_ptr_ = this;
}

static auto FindClassField(Nonnull<const NominalClassValue*> object,
                           std::string_view name)
    -> std::optional<Nonnull<const Value*>> {
  if (auto field = cast<StructValue>(object->inits()).FindField(name)) {
    return field;
  }
  if (object->base().has_value()) {
    return FindClassField(object->base().value(), name);
  }
  return std::nullopt;
}

static auto GetBaseElement(Nonnull<const NominalClassValue*> class_value,
                           SourceLocation source_loc)
    -> ErrorOr<Nonnull<const Value*>> {
  const auto base = cast<NominalClassValue>(class_value)->base();
  if (!base.has_value()) {
    return ProgramError(source_loc)
           << "Non-existent base class for " << *class_value;
  }
  return base.value();
}

static auto GetPositionalElement(Nonnull<const TupleValue*> tuple,
                                 const ElementPath::Component& path_comp,
                                 SourceLocation source_loc)
    -> ErrorOr<Nonnull<const Value*>> {
  CARBON_CHECK(path_comp.element()->kind() == ElementKind::PositionalElement)
      << "Invalid non-tuple member";
  const auto* tuple_element = cast<PositionalElement>(path_comp.element());
  const size_t index = tuple_element->index();
  if (index < 0 || index >= tuple->elements().size()) {
    return ProgramError(source_loc)
           << "index " << index << " out of range for " << *tuple;
  }
  return tuple->elements()[index];
}

static auto GetNamedElement(Nonnull<Arena*> arena, Nonnull<const Value*> v,
                            const ElementPath::Component& field,
                            SourceLocation source_loc,
                            Nonnull<const Value*> me_value)
    -> ErrorOr<Nonnull<const Value*>> {
  CARBON_CHECK(field.element()->kind() == ElementKind::NamedElement)
      << "Invalid element, expecting NamedElement";
  const auto* member = cast<NamedElement>(field.element());
  const auto f = member->name();
  if (field.witness().has_value()) {
    const auto* witness = cast<Witness>(*field.witness());

    // Associated constants.
    if (const auto* assoc_const =
            dyn_cast_or_null<AssociatedConstantDeclaration>(
                member->declaration().value_or(nullptr))) {
      CARBON_CHECK(field.interface()) << "have witness but no interface";
      // TODO: Use witness to find the value of the constant.
      return arena->New<AssociatedConstant>(v, *field.interface(), assoc_const,
                                            witness);
    }

    // Associated functions.
    if (const auto* impl_witness = dyn_cast<ImplWitness>(witness)) {
      if (std::optional<Nonnull<const Declaration*>> mem_decl =
              FindMember(f, impl_witness->declaration().members());
          mem_decl.has_value()) {
        const auto& fun_decl = cast<FunctionDeclaration>(**mem_decl);
        if (fun_decl.is_method()) {
          return arena->New<BoundMethodValue>(&fun_decl, me_value,
                                              &impl_witness->bindings());
        } else {
          // Class function.
          const auto* fun = cast<FunctionValue>(*fun_decl.constant_value());
          return arena->New<FunctionValue>(&fun->declaration(),
                                           &impl_witness->bindings());
        }
      } else {
        return ProgramError(source_loc)
               << "member " << f << " not in " << *witness;
      }
    } else {
      return ProgramError(source_loc)
             << "member lookup for " << f << " in symbolic " << *witness;
    }
  }
  switch (v->kind()) {
    case Value::Kind::StructValue: {
      std::optional<Nonnull<const Value*>> field =
          cast<StructValue>(*v).FindField(f);
      if (field == std::nullopt) {
        return ProgramError(source_loc) << "member " << f << " not in " << *v;
      }
      return *field;
    }
    case Value::Kind::NominalClassValue: {
      const auto& object = cast<NominalClassValue>(*v);
      // Look for a field.
      if (std::optional<Nonnull<const Value*>> field =
              FindClassField(&object, f)) {
        return *field;
      } else {
        // Look for a method in the object's class
        const auto& class_type = cast<NominalClassType>(object.type());
        std::optional<Nonnull<const FunctionValue*>> func =
            FindFunctionWithParents(f, class_type.declaration());
        if (!func) {
          return ProgramError(source_loc) << "member " << f << " not in " << *v
                                          << " or its " << class_type;
        } else if ((*func)->declaration().is_method()) {
          // Found a method. Turn it into a bound method.
          const auto& m = cast<FunctionValue>(**func);
          if (m.declaration().virt_override() == VirtualOverride::None) {
            return arena->New<BoundMethodValue>(&m.declaration(), me_value,
                                                &class_type.bindings());
          }
          // Method is virtual, get child-most class value and perform vtable
          // lookup.
          const auto& last_child_value = **object.class_value_ptr();
          const auto& last_child_type =
              cast<NominalClassType>(last_child_value.type());
          const auto res = last_child_type.vtable().find(f);
          CARBON_CHECK(res != last_child_type.vtable().end());
          const auto [virtual_method, level] = res->second;
          const auto level_diff = last_child_type.hierarchy_level() - level;
          const auto* m_class_value = &last_child_value;
          // Get class value matching the virtual method, and turn it into a
          // bound method.
          for (int i = 0; i < level_diff; ++i) {
            CARBON_CHECK(m_class_value->base())
                << "Error trying to access function class value";
            m_class_value = *m_class_value->base();
          }
          return arena->New<BoundMethodValue>(
              cast<FunctionDeclaration>(virtual_method), m_class_value,
              &class_type.bindings());
        } else {
          // Found a class function
          // TODO: This should not be reachable.
          return arena->New<FunctionValue>(&(*func)->declaration(),
                                           &class_type.bindings());
        }
      }
    }
    case Value::Kind::ChoiceType: {
      const auto& choice = cast<ChoiceType>(*v);
      if (!choice.FindAlternative(f)) {
        return ProgramError(source_loc)
               << "alternative " << f << " not in " << *v;
      }
      return arena->New<AlternativeConstructorValue>(f, choice.name());
    }
    case Value::Kind::NominalClassType: {
      // Access a class function.
      const auto& class_type = cast<NominalClassType>(*v);
      std::optional<Nonnull<const FunctionValue*>> fun =
          FindFunctionWithParents(f, class_type.declaration());
      if (fun == std::nullopt) {
        return ProgramError(source_loc)
               << "class function " << f << " not in " << *v;
      }
      return arena->New<FunctionValue>(&(*fun)->declaration(),
                                       &class_type.bindings());
    }
    default:
      CARBON_FATAL() << "named element access not supported for value " << *v;
  }
}

static auto GetElement(Nonnull<Arena*> arena, Nonnull<const Value*> v,
                       const ElementPath::Component& path_comp,
                       SourceLocation source_loc,
                       Nonnull<const Value*> me_value)
    -> ErrorOr<Nonnull<const Value*>> {
  switch (path_comp.element()->kind()) {
    case ElementKind::NamedElement:
      return GetNamedElement(arena, v, path_comp, source_loc, me_value);
    case ElementKind::PositionalElement: {
      if (const auto* tuple = dyn_cast<TupleValue>(v)) {
        return GetPositionalElement(tuple, path_comp, source_loc);
      } else {
        CARBON_FATAL() << "Invalid value for positional element";
      }
    }
    case ElementKind::BaseElement:
      switch (v->kind()) {
        case Value::Kind::NominalClassValue:
          return GetBaseElement(cast<NominalClassValue>(v), source_loc);
        case Value::Kind::PointerValue: {
          const auto* ptr = cast<PointerValue>(v);
          return arena->New<PointerValue>(
              ptr->address().ElementAddress(path_comp.element()));
        }
        default:
          CARBON_FATAL() << "Invalid value for base element";
      }
  }
}

auto Value::GetElement(Nonnull<Arena*> arena, const ElementPath& path,
                       SourceLocation source_loc,
                       Nonnull<const Value*> me_value) const
    -> ErrorOr<Nonnull<const Value*>> {
  Nonnull<const Value*> value(this);
  for (const ElementPath::Component& field : path.components_) {
    CARBON_ASSIGN_OR_RETURN(
        value, Carbon::GetElement(arena, value, field, source_loc, me_value));
  }
  return value;
}

static auto SetFieldImpl(
    Nonnull<Arena*> arena, Nonnull<const Value*> value,
    std::vector<ElementPath::Component>::const_iterator path_begin,
    std::vector<ElementPath::Component>::const_iterator path_end,
    Nonnull<const Value*> field_value, SourceLocation source_loc)
    -> ErrorOr<Nonnull<const Value*>> {
  if (path_begin == path_end) {
    return field_value;
  }
  switch (value->kind()) {
    case Value::Kind::StructValue: {
      std::vector<NamedValue> elements = cast<StructValue>(*value).elements();
      auto it =
          llvm::find_if(elements, [path_begin](const NamedValue& element) {
            return (*path_begin).IsNamed(element.name);
          });
      if (it == elements.end()) {
        return ProgramError(source_loc)
               << "field " << *path_begin << " not in " << *value;
      }
      CARBON_ASSIGN_OR_RETURN(
          it->value, SetFieldImpl(arena, it->value, path_begin + 1, path_end,
                                  field_value, source_loc));
      return arena->New<StructValue>(elements);
    }
    case Value::Kind::NominalClassValue: {
      const auto& object = cast<NominalClassValue>(*value);
      if (auto inits = SetFieldImpl(arena, &object.inits(), path_begin,
                                    path_end, field_value, source_loc);
          inits.ok()) {
        return arena->New<NominalClassValue>(
            &object.type(), *inits, object.base(), object.class_value_ptr());
      } else if (object.base().has_value()) {
        auto new_base = SetFieldImpl(arena, object.base().value(), path_begin,
                                     path_end, field_value, source_loc);
        if (new_base.ok()) {
          return arena->New<NominalClassValue>(
              &object.type(), &object.inits(),
              cast<NominalClassValue>(*new_base), object.class_value_ptr());
        }
      }
      // Failed to match, show full object content
      return ProgramError(source_loc)
             << "field " << *path_begin << " not in " << *value;
    }
    case Value::Kind::TupleType:
    case Value::Kind::TupleValue: {
      CARBON_CHECK((*path_begin).element()->kind() ==
                   ElementKind::PositionalElement)
          << "Invalid non-positional member for tuple";
      std::vector<Nonnull<const Value*>> elements =
          cast<TupleValueBase>(*value).elements();
      const size_t index =
          cast<PositionalElement>((*path_begin).element())->index();
      if (index < 0 || index >= elements.size()) {
        return ProgramError(source_loc)
               << "index " << index << " out of range in " << *value;
      }
      CARBON_ASSIGN_OR_RETURN(
          elements[index], SetFieldImpl(arena, elements[index], path_begin + 1,
                                        path_end, field_value, source_loc));
      if (isa<TupleType>(value)) {
        return arena->New<TupleType>(elements);
      } else {
        return arena->New<TupleValue>(elements);
      }
    }
    default:
      CARBON_FATAL() << "field access not allowed for value " << *value;
  }
}

auto Value::SetField(Nonnull<Arena*> arena, const ElementPath& path,
                     Nonnull<const Value*> field_value,
                     SourceLocation source_loc) const
    -> ErrorOr<Nonnull<const Value*>> {
  return SetFieldImpl(arena, static_cast<Nonnull<const Value*>>(this),
                      path.components_.begin(), path.components_.end(),
                      field_value, source_loc);
}

static auto PrintNameWithBindings(llvm::raw_ostream& out,
                                  Nonnull<const Declaration*> declaration,
                                  const BindingMap& args) {
  out << GetName(*declaration).value_or("(anonymous)");
  // TODO: Print '()' if declaration is parameterized but no args are provided.
  if (!args.empty()) {
    out << "(";
    llvm::ListSeparator sep;
    for (const auto& [bind, val] : args) {
      out << sep << bind->name() << " = " << *val;
    }
    out << ")";
  }
}

void Value::Print(llvm::raw_ostream& out) const {
  switch (kind()) {
    case Value::Kind::AlternativeConstructorValue: {
      const auto& alt = cast<AlternativeConstructorValue>(*this);
      out << alt.choice_name() << "." << alt.alt_name();
      break;
    }
    case Value::Kind::BindingPlaceholderValue: {
      const auto& placeholder = cast<BindingPlaceholderValue>(*this);
      out << "Placeholder<";
      if (placeholder.value_node().has_value()) {
        out << (*placeholder.value_node());
      } else {
        out << "_";
      }
      out << ">";
      break;
    }
    case Value::Kind::AddrValue: {
      const auto& addr = cast<AddrValue>(*this);
      out << "Addr<" << addr.pattern() << ">";
      break;
    }
    case Value::Kind::AlternativeValue: {
      const auto& alt = cast<AlternativeValue>(*this);
      out << "alt " << alt.choice_name() << "." << alt.alt_name() << " "
          << alt.argument();
      break;
    }
    case Value::Kind::StructValue: {
      const auto& struct_val = cast<StructValue>(*this);
      out << "{";
      llvm::ListSeparator sep;
      for (const NamedValue& element : struct_val.elements()) {
        out << sep << "." << element.name << " = " << *element.value;
      }
      out << "}";
      break;
    }
    case Value::Kind::NominalClassValue: {
      const auto& s = cast<NominalClassValue>(*this);
      out << cast<NominalClassType>(s.type()).declaration().name() << s.inits();
      if (s.base().has_value()) {
        out << " base " << *s.base().value();
      }
      break;
    }
    case Value::Kind::TupleType:
    case Value::Kind::TupleValue: {
      out << "(";
      llvm::ListSeparator sep;
      for (Nonnull<const Value*> element :
           cast<TupleValueBase>(*this).elements()) {
        out << sep << *element;
      }
      out << ")";
      break;
    }
    case Value::Kind::IntValue:
      out << cast<IntValue>(*this).value();
      break;
    case Value::Kind::BoolValue:
      out << (cast<BoolValue>(*this).value() ? "true" : "false");
      break;
    case Value::Kind::DestructorValue: {
      const auto& destructor = cast<DestructorValue>(*this);
      out << "destructor [ ";
      out << destructor.declaration().self_pattern();
      out << " ]";
      break;
    }
    case Value::Kind::FunctionValue: {
      const auto& fun = cast<FunctionValue>(*this);
      out << "fun<" << fun.declaration().name() << ">";
      if (!fun.type_args().empty()) {
        out << "[";
        llvm::ListSeparator sep;
        for (const auto& [ty_var, ty_arg] : fun.type_args()) {
          out << sep << *ty_var << "=" << *ty_arg;
        }
        out << "]";
      }
      if (!fun.witnesses().empty()) {
        out << "{|";
        llvm::ListSeparator sep;
        for (const auto& [impl_bind, witness] : fun.witnesses()) {
          out << sep << *witness;
        }
        out << "|}";
      }
      break;
    }
    case Value::Kind::BoundMethodValue: {
      const auto& method = cast<BoundMethodValue>(*this);
      out << "bound_method<" << method.declaration().name() << ">";
      if (!method.type_args().empty()) {
        out << "[";
        llvm::ListSeparator sep;
        for (const auto& [ty_var, ty_arg] : method.type_args()) {
          out << sep << *ty_var << "=" << *ty_arg;
        }
        out << "]";
      }
      if (!method.witnesses().empty()) {
        out << "{|";
        llvm::ListSeparator sep;
        for (const auto& [impl_bind, witness] : method.witnesses()) {
          out << sep << *witness;
        }
        out << "|}";
      }
      break;
    }
    case Value::Kind::PointerValue:
      out << "ptr<" << cast<PointerValue>(*this).address() << ">";
      break;
    case Value::Kind::LValue:
      out << "lval<" << cast<LValue>(*this).address() << ">";
      break;
    case Value::Kind::BoolType:
      out << "bool";
      break;
    case Value::Kind::IntType:
      out << "i32";
      break;
    case Value::Kind::TypeType:
      out << "type";
      break;
    case Value::Kind::AutoType:
      out << "auto";
      break;
    case Value::Kind::ContinuationType:
      out << "Continuation";
      break;
    case Value::Kind::PointerType:
      out << cast<PointerType>(*this).pointee_type() << "*";
      break;
    case Value::Kind::FunctionType: {
      const auto& fn_type = cast<FunctionType>(*this);
      out << "fn ";
      if (!fn_type.deduced_bindings().empty()) {
        out << "[";
        llvm::ListSeparator sep;
        for (Nonnull<const GenericBinding*> deduced :
             fn_type.deduced_bindings()) {
          out << sep << *deduced;
        }
        out << "]";
      }
      out << fn_type.parameters() << " -> " << fn_type.return_type();
      break;
    }
    case Value::Kind::StructType: {
      out << "{";
      llvm::ListSeparator sep;
      for (const auto& [name, type] : cast<StructType>(*this).fields()) {
        out << sep << "." << name << ": " << *type;
      }
      out << "}";
      break;
    }
    case Value::Kind::UninitializedValue: {
      const auto& uninit = cast<UninitializedValue>(*this);
      out << "Uninit<" << uninit.pattern() << ">";
      break;
    }
    case Value::Kind::NominalClassType: {
      const auto& class_type = cast<NominalClassType>(*this);
      out << "class ";
      PrintNameWithBindings(out, &class_type.declaration(),
                            class_type.type_args());
      if (!class_type.witnesses().empty()) {
        out << " witnesses ";
        llvm::ListSeparator sep;
        for (const auto& [impl_bind, witness] : class_type.witnesses()) {
          out << sep << *witness;
        }
      }
      break;
    }
    case Value::Kind::MixinPseudoType: {
      const auto& mixin_type = cast<MixinPseudoType>(*this);
      out << "mixin ";
      PrintNameWithBindings(out, &mixin_type.declaration(), mixin_type.args());
      if (!mixin_type.witnesses().empty()) {
        out << " witnesses ";
        llvm::ListSeparator sep;
        for (const auto& [impl_bind, witness] : mixin_type.witnesses()) {
          out << sep << *witness;
        }
      }
      // TODO: print the import interface
      break;
    }
    case Value::Kind::InterfaceType: {
      const auto& iface_type = cast<InterfaceType>(*this);
      out << "interface ";
      PrintNameWithBindings(out, &iface_type.declaration(),
                            iface_type.bindings().args());
      break;
    }
    case Value::Kind::NamedConstraintType: {
      const auto& constraint_type = cast<NamedConstraintType>(*this);
      out << "constraint ";
      PrintNameWithBindings(out, &constraint_type.declaration(),
                            constraint_type.bindings().args());
      break;
    }
    case Value::Kind::ConstraintType: {
      const auto& constraint = cast<ConstraintType>(*this);
      llvm::ListSeparator combine(" & ");
      for (const LookupContext& ctx : constraint.lookup_contexts()) {
        out << combine << *ctx.context;
      }
      if (constraint.lookup_contexts().empty()) {
        out << "type";
      }
      out << " where ";
      llvm::ListSeparator sep(" and ");
      for (const RewriteConstraint& rewrite :
           constraint.rewrite_constraints()) {
        out << sep << ".(";
        PrintNameWithBindings(out, &rewrite.constant->interface().declaration(),
                              rewrite.constant->interface().args());
        out << "." << *GetName(rewrite.constant->constant())
            << ") = " << *rewrite.unconverted_replacement;
      }
      for (const ImplConstraint& impl : constraint.impl_constraints()) {
        // TODO: Skip cases where `impl.type` is `.Self` and the interface is
        // in `lookup_contexts()`.
        out << sep << *impl.type << " is " << *impl.interface;
      }
      for (const EqualityConstraint& equality :
           constraint.equality_constraints()) {
        out << sep;
        llvm::ListSeparator equal(" == ");
        for (Nonnull<const Value*> value : equality.values) {
          out << equal << *value;
        }
      }
      break;
    }
    case Value::Kind::ImplWitness: {
      const auto& witness = cast<ImplWitness>(*this);
      out << "witness for impl " << *witness.declaration().impl_type() << " as "
          << witness.declaration().interface();
      break;
    }
    case Value::Kind::BindingWitness: {
      const auto& witness = cast<BindingWitness>(*this);
      out << "witness for " << *witness.binding()->type_var();
      break;
    }
    case Value::Kind::ConstraintWitness: {
      const auto& witness = cast<ConstraintWitness>(*this);
      out << "(";
      llvm::ListSeparator sep;
      for (const auto* elem : witness.witnesses()) {
        out << sep << *elem;
      }
      out << ")";
      break;
    }
    case Value::Kind::ConstraintImplWitness: {
      const auto& witness = cast<ConstraintImplWitness>(*this);
      out << "witness " << witness.index() << " of "
          << *witness.constraint_witness();
      break;
    }
    case Value::Kind::ParameterizedEntityName:
      out << *GetName(cast<ParameterizedEntityName>(*this).declaration());
      break;
    case Value::Kind::MemberName: {
      const auto& member_name = cast<MemberName>(*this);
      if (member_name.base_type().has_value()) {
        out << *member_name.base_type().value();
      }
      if (member_name.base_type().has_value() &&
          member_name.interface().has_value()) {
        out << "(";
      }
      if (member_name.interface().has_value()) {
        out << *member_name.interface().value();
      }
      out << "." << member_name;
      if (member_name.base_type().has_value() &&
          member_name.interface().has_value()) {
        out << ")";
      }
      break;
    }
    case Value::Kind::ChoiceType:
      out << "choice " << cast<ChoiceType>(*this).name();
      break;
    case Value::Kind::VariableType:
      out << cast<VariableType>(*this).binding().name();
      break;
    case Value::Kind::AssociatedConstant: {
      const auto& assoc = cast<AssociatedConstant>(*this);
      out << "(" << assoc.base() << ").(";
      PrintNameWithBindings(out, &assoc.interface().declaration(),
                            assoc.interface().args());
      out << "." << *GetName(assoc.constant()) << ")";
      break;
    }
    case Value::Kind::ContinuationValue: {
      out << cast<ContinuationValue>(*this).stack();
      break;
    }
    case Value::Kind::StringType:
      out << "String";
      break;
    case Value::Kind::StringValue:
      out << "\"";
      out.write_escaped(cast<StringValue>(*this).value());
      out << "\"";
      break;
    case Value::Kind::TypeOfMixinPseudoType:
      out << "typeof("
          << cast<TypeOfMixinPseudoType>(*this)
                 .mixin_type()
                 .declaration()
                 .name()
          << ")";
      break;
    case Value::Kind::TypeOfParameterizedEntityName:
      out << "parameterized entity name "
          << cast<TypeOfParameterizedEntityName>(*this).name();
      break;
    case Value::Kind::TypeOfMemberName: {
      out << "member name " << cast<TypeOfMemberName>(*this).member();
      break;
    }
    case Value::Kind::StaticArrayType: {
      const auto& array_type = cast<StaticArrayType>(*this);
      out << "[" << array_type.element_type() << "; " << array_type.size()
          << "]";
      break;
    }
  }
}

ContinuationValue::StackFragment::~StackFragment() {
  CARBON_CHECK(reversed_todo_.empty())
      << "All StackFragments must be empty before the Carbon program ends.";
}

void ContinuationValue::StackFragment::StoreReversed(
    std::vector<std::unique_ptr<Action>> reversed_todo) {
  CARBON_CHECK(reversed_todo_.empty());
  reversed_todo_ = std::move(reversed_todo);
}

void ContinuationValue::StackFragment::RestoreTo(
    Stack<std::unique_ptr<Action>>& todo) {
  while (!reversed_todo_.empty()) {
    todo.Push(std::move(reversed_todo_.back()));
    reversed_todo_.pop_back();
  }
}

void ContinuationValue::StackFragment::Clear() {
  // We destroy the underlying Actions explicitly to ensure they're
  // destroyed in the correct order.
  for (auto& action : reversed_todo_) {
    action.reset();
  }
  reversed_todo_.clear();
}

void ContinuationValue::StackFragment::Print(llvm::raw_ostream& out) const {
  out << "{";
  llvm::ListSeparator sep(" :: ");
  for (const std::unique_ptr<Action>& action : reversed_todo_) {
    out << sep << *action;
  }
  out << "}";
}

// Check whether two binding maps, which are assumed to have the same keys, are
// equal.
static auto BindingMapEqual(
    const BindingMap& map1, const BindingMap& map2,
    std::optional<Nonnull<const EqualityContext*>> equality_ctx) -> bool {
  CARBON_CHECK(map1.size() == map2.size()) << "maps should have same keys";
  for (const auto& [key, value] : map1) {
    if (!ValueEqual(value, map2.at(key), equality_ctx)) {
      return false;
    }
  }
  return true;
}

auto TypeEqual(Nonnull<const Value*> t1, Nonnull<const Value*> t2,
               std::optional<Nonnull<const EqualityContext*>> equality_ctx)
    -> bool {
  if (t1 == t2) {
    return true;
  }
  if (t1->kind() != t2->kind()) {
    if (IsValueKindDependent(t1) || IsValueKindDependent(t2)) {
      return ValueEqual(t1, t2, equality_ctx);
    }
    return false;
  }
  switch (t1->kind()) {
    case Value::Kind::PointerType:
      return TypeEqual(&cast<PointerType>(*t1).pointee_type(),
                       &cast<PointerType>(*t2).pointee_type(), equality_ctx);
    case Value::Kind::FunctionType: {
      const auto& fn1 = cast<FunctionType>(*t1);
      const auto& fn2 = cast<FunctionType>(*t2);
      return TypeEqual(&fn1.parameters(), &fn2.parameters(), equality_ctx) &&
             TypeEqual(&fn1.return_type(), &fn2.return_type(), equality_ctx);
    }
    case Value::Kind::StructType: {
      const auto& struct1 = cast<StructType>(*t1);
      const auto& struct2 = cast<StructType>(*t2);
      if (struct1.fields().size() != struct2.fields().size()) {
        return false;
      }
      for (size_t i = 0; i < struct1.fields().size(); ++i) {
        if (struct1.fields()[i].name != struct2.fields()[i].name ||
            !TypeEqual(struct1.fields()[i].value, struct2.fields()[i].value,
                       equality_ctx)) {
          return false;
        }
      }
      return true;
    }
    case Value::Kind::NominalClassType: {
      const auto& class1 = cast<NominalClassType>(*t1);
      const auto& class2 = cast<NominalClassType>(*t2);
      return class1.declaration().name() == class2.declaration().name() &&
             BindingMapEqual(class1.bindings().args(), class2.bindings().args(),
                             equality_ctx);
    }
    case Value::Kind::InterfaceType: {
      const auto& iface1 = cast<InterfaceType>(*t1);
      const auto& iface2 = cast<InterfaceType>(*t2);
      return iface1.declaration().name() == iface2.declaration().name() &&
             BindingMapEqual(iface1.bindings().args(), iface2.bindings().args(),
                             equality_ctx);
    }
    case Value::Kind::NamedConstraintType: {
      const auto& constraint1 = cast<NamedConstraintType>(*t1);
      const auto& constraint2 = cast<NamedConstraintType>(*t2);
      return constraint1.declaration().name() ==
                 constraint2.declaration().name() &&
             BindingMapEqual(constraint1.bindings().args(),
                             constraint2.bindings().args(), equality_ctx);
    }
    case Value::Kind::AssociatedConstant:
      // Associated constants are sometimes types.
      return ValueEqual(t1, t2, equality_ctx);
    case Value::Kind::ConstraintType: {
      const auto& constraint1 = cast<ConstraintType>(*t1);
      const auto& constraint2 = cast<ConstraintType>(*t2);
      if (constraint1.impl_constraints().size() !=
              constraint2.impl_constraints().size() ||
          constraint1.equality_constraints().size() !=
              constraint2.equality_constraints().size() ||
          constraint1.lookup_contexts().size() !=
              constraint2.lookup_contexts().size()) {
        return false;
      }
      for (size_t i = 0; i < constraint1.impl_constraints().size(); ++i) {
        const auto& impl1 = constraint1.impl_constraints()[i];
        const auto& impl2 = constraint2.impl_constraints()[i];
        if (!TypeEqual(impl1.type, impl2.type, equality_ctx) ||
            !TypeEqual(impl1.interface, impl2.interface, equality_ctx)) {
          return false;
        }
      }
      for (size_t i = 0; i < constraint1.equality_constraints().size(); ++i) {
        const auto& equality1 = constraint1.equality_constraints()[i];
        const auto& equality2 = constraint2.equality_constraints()[i];
        if (equality1.values.size() != equality2.values.size()) {
          return false;
        }
        for (size_t j = 0; j < equality1.values.size(); ++j) {
          if (!ValueEqual(equality1.values[i], equality2.values[i],
                          equality_ctx)) {
            return false;
          }
        }
      }
      for (size_t i = 0; i < constraint1.lookup_contexts().size(); ++i) {
        const auto& context1 = constraint1.lookup_contexts()[i];
        const auto& context2 = constraint2.lookup_contexts()[i];
        if (!TypeEqual(context1.context, context2.context, equality_ctx)) {
          return false;
        }
      }
      return true;
    }
    case Value::Kind::ChoiceType:
      return cast<ChoiceType>(*t1).name() == cast<ChoiceType>(*t2).name();
    case Value::Kind::TupleType:
    case Value::Kind::TupleValue: {
      const auto& tup1 = cast<TupleValueBase>(*t1);
      const auto& tup2 = cast<TupleValueBase>(*t2);
      if (tup1.elements().size() != tup2.elements().size()) {
        return false;
      }
      for (size_t i = 0; i < tup1.elements().size(); ++i) {
        if (!TypeEqual(tup1.elements()[i], tup2.elements()[i], equality_ctx)) {
          return false;
        }
      }
      return true;
    }
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::ContinuationType:
    case Value::Kind::TypeType:
    case Value::Kind::StringType:
      return true;
    case Value::Kind::VariableType:
      return &cast<VariableType>(*t1).binding() ==
             &cast<VariableType>(*t2).binding();
    case Value::Kind::StaticArrayType: {
      const auto& array1 = cast<StaticArrayType>(*t1);
      const auto& array2 = cast<StaticArrayType>(*t2);
      return TypeEqual(&array1.element_type(), &array2.element_type(),
                       equality_ctx) &&
             array1.size() == array2.size();
    }
    case Value::Kind::IntValue:
    case Value::Kind::BoolValue:
    case Value::Kind::DestructorValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::BoundMethodValue:
    case Value::Kind::StructValue:
    case Value::Kind::NominalClassValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::StringValue:
    case Value::Kind::PointerValue:
    case Value::Kind::LValue:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AddrValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::UninitializedValue:
    case Value::Kind::ParameterizedEntityName:
    case Value::Kind::MemberName:
    case Value::Kind::TypeOfParameterizedEntityName:
    case Value::Kind::TypeOfMemberName:
    case Value::Kind::MixinPseudoType:
    case Value::Kind::TypeOfMixinPseudoType:
      CARBON_FATAL() << "TypeEqual used to compare non-type values\n"
                     << *t1 << "\n"
                     << *t2;
    case Value::Kind::ImplWitness:
    case Value::Kind::BindingWitness:
    case Value::Kind::ConstraintWitness:
    case Value::Kind::ConstraintImplWitness:
      CARBON_FATAL() << "TypeEqual: unexpected Witness";
      break;
    case Value::Kind::AutoType:
      CARBON_FATAL() << "TypeEqual: unexpected AutoType";
      break;
  }
}

// Returns true if the two values are known to be equal and are written in the
// same way at the top level.
auto ValueStructurallyEqual(
    Nonnull<const Value*> v1, Nonnull<const Value*> v2,
    std::optional<Nonnull<const EqualityContext*>> equality_ctx) -> bool {
  if (v1 == v2) {
    return true;
  }
  if (v1->kind() != v2->kind()) {
    return false;
  }
  switch (v1->kind()) {
    case Value::Kind::IntValue:
      return cast<IntValue>(*v1).value() == cast<IntValue>(*v2).value();
    case Value::Kind::BoolValue:
      return cast<BoolValue>(*v1).value() == cast<BoolValue>(*v2).value();
    case Value::Kind::FunctionValue: {
      std::optional<Nonnull<const Statement*>> body1 =
          cast<FunctionValue>(*v1).declaration().body();
      std::optional<Nonnull<const Statement*>> body2 =
          cast<FunctionValue>(*v2).declaration().body();
      return body1.has_value() == body2.has_value() &&
             (!body1.has_value() || *body1 == *body2);
    }
    case Value::Kind::DestructorValue:
      return false;
    case Value::Kind::BoundMethodValue: {
      const auto& m1 = cast<BoundMethodValue>(*v1);
      const auto& m2 = cast<BoundMethodValue>(*v2);
      std::optional<Nonnull<const Statement*>> body1 = m1.declaration().body();
      std::optional<Nonnull<const Statement*>> body2 = m2.declaration().body();
      return ValueEqual(m1.receiver(), m2.receiver(), equality_ctx) &&
             body1.has_value() == body2.has_value() &&
             (!body1.has_value() || *body1 == *body2);
    }
    case Value::Kind::TupleType:
    case Value::Kind::TupleValue: {
      const std::vector<Nonnull<const Value*>>& elements1 =
          cast<TupleValueBase>(*v1).elements();
      const std::vector<Nonnull<const Value*>>& elements2 =
          cast<TupleValueBase>(*v2).elements();
      if (elements1.size() != elements2.size()) {
        return false;
      }
      for (size_t i = 0; i < elements1.size(); ++i) {
        if (!ValueEqual(elements1[i], elements2[i], equality_ctx)) {
          return false;
        }
      }
      return true;
    }
    case Value::Kind::StructValue: {
      const auto& struct_v1 = cast<StructValue>(*v1);
      const auto& struct_v2 = cast<StructValue>(*v2);
      CARBON_CHECK(struct_v1.elements().size() == struct_v2.elements().size());
      for (size_t i = 0; i < struct_v1.elements().size(); ++i) {
        CARBON_CHECK(struct_v1.elements()[i].name ==
                     struct_v2.elements()[i].name);
        if (!ValueEqual(struct_v1.elements()[i].value,
                        struct_v2.elements()[i].value, equality_ctx)) {
          return false;
        }
      }
      return true;
    }
    case Value::Kind::StringValue:
      return cast<StringValue>(*v1).value() == cast<StringValue>(*v2).value();
    case Value::Kind::ParameterizedEntityName: {
      std::optional<std::string_view> name1 =
          GetName(cast<ParameterizedEntityName>(v1)->declaration());
      std::optional<std::string_view> name2 =
          GetName(cast<ParameterizedEntityName>(v2)->declaration());
      CARBON_CHECK(name1.has_value() && name2.has_value())
          << "parameterized name refers to unnamed declaration";
      return *name1 == *name2;
    }
    case Value::Kind::AssociatedConstant: {
      // The witness value is not part of determining value equality.
      const auto& assoc1 = cast<AssociatedConstant>(*v1);
      const auto& assoc2 = cast<AssociatedConstant>(*v2);
      return &assoc1.constant() == &assoc2.constant() &&
             TypeEqual(&assoc1.base(), &assoc2.base(), equality_ctx) &&
             TypeEqual(&assoc1.interface(), &assoc2.interface(), equality_ctx);
    }
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::FunctionType:
    case Value::Kind::PointerType:
    case Value::Kind::AutoType:
    case Value::Kind::StructType:
    case Value::Kind::NominalClassType:
    case Value::Kind::MixinPseudoType:
    case Value::Kind::InterfaceType:
    case Value::Kind::NamedConstraintType:
    case Value::Kind::ConstraintType:
    case Value::Kind::ImplWitness:
    case Value::Kind::BindingWitness:
    case Value::Kind::ConstraintWitness:
    case Value::Kind::ConstraintImplWitness:
    case Value::Kind::ChoiceType:
    case Value::Kind::ContinuationType:
    case Value::Kind::VariableType:
    case Value::Kind::StringType:
    case Value::Kind::TypeOfMixinPseudoType:
    case Value::Kind::TypeOfParameterizedEntityName:
    case Value::Kind::TypeOfMemberName:
    case Value::Kind::StaticArrayType:
      return TypeEqual(v1, v2, equality_ctx);
    case Value::Kind::NominalClassValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AddrValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::PointerValue:
    case Value::Kind::LValue:
    case Value::Kind::UninitializedValue:
    case Value::Kind::MemberName:
      // TODO: support pointer comparisons once we have a clearer distinction
      // between pointers and lvalues.
      CARBON_FATAL() << "ValueEqual does not support this kind of value: "
                     << *v1;
  }
}

// Returns true if the two values are equal and returns false otherwise.
//
// This function implements the `==` operator of Carbon.
auto ValueEqual(Nonnull<const Value*> v1, Nonnull<const Value*> v2,
                std::optional<Nonnull<const EqualityContext*>> equality_ctx)
    -> bool {
  if (v1 == v2) {
    return true;
  }

  // If we're given an equality context, check to see if it knows these values
  // are equal. Only perform the check if one or the other value is an
  // associated constant; otherwise we should be able to do better by looking
  // at the structures of the values.
  if (equality_ctx) {
    if (IsValueKindDependent(v1)) {
      auto visitor = [&](Nonnull<const Value*> maybe_v2) {
        return !ValueStructurallyEqual(v2, maybe_v2, equality_ctx);
      };
      if (!(*equality_ctx)->VisitEqualValues(v1, visitor)) {
        return true;
      }
    }
    if (IsValueKindDependent(v2)) {
      auto visitor = [&](Nonnull<const Value*> maybe_v1) {
        return !ValueStructurallyEqual(v1, maybe_v1, equality_ctx);
      };
      if (!(*equality_ctx)->VisitEqualValues(v2, visitor)) {
        return true;
      }
    }
  }

  return ValueStructurallyEqual(v1, v2, equality_ctx);
}

auto EqualityConstraint::VisitEqualValues(
    Nonnull<const Value*> value,
    llvm::function_ref<bool(Nonnull<const Value*>)> visitor) const -> bool {
  // See if the given value is part of this constraint.
  auto first_equal = llvm::find_if(values, [value](Nonnull<const Value*> val) {
    return ValueEqual(value, val, std::nullopt);
  });
  if (first_equal == values.end()) {
    return true;
  }

  // The value is in this group; pass all non-identical values in the group
  // to the visitor. First visit the values we already compared.
  for (const auto* val : llvm::make_range(values.begin(), first_equal)) {
    if (!visitor(val)) {
      return false;
    }
  }
  // Then visit any remaining non-identical values, skipping the one we already
  // found was identical.
  ++first_equal;
  for (const auto* val : llvm::make_range(first_equal, values.end())) {
    if (!ValueEqual(value, val, std::nullopt) && !visitor(val)) {
      return false;
    }
  }
  return true;
}

auto ConstraintType::VisitEqualValues(
    Nonnull<const Value*> value,
    llvm::function_ref<bool(Nonnull<const Value*>)> visitor) const -> bool {
  for (const auto& eq : equality_constraints()) {
    if (!eq.VisitEqualValues(value, visitor)) {
      return false;
    }
  }
  return true;
}

auto ChoiceType::FindAlternative(std::string_view name) const
    -> std::optional<Nonnull<const Value*>> {
  std::vector<NamedValue> alternatives = declaration_->members();
  for (const NamedValue& alternative : alternatives) {
    if (alternative.name == name) {
      return alternative.value;
    }
  }
  return std::nullopt;
}

auto FindFunction(std::string_view name,
                  llvm::ArrayRef<Nonnull<Declaration*>> members)
    -> std::optional<Nonnull<const FunctionValue*>> {
  for (const auto& member : members) {
    switch (member->kind()) {
      case DeclarationKind::MixDeclaration: {
        const auto& mix_decl = cast<MixDeclaration>(*member);
        Nonnull<const MixinPseudoType*> mixin = &mix_decl.mixin_value();
        const auto res = mixin->FindFunction(name);
        if (res.has_value()) {
          return res;
        }
        break;
      }
      case DeclarationKind::FunctionDeclaration: {
        const auto& fun = cast<CallableDeclaration>(*member);
        if (fun.name() == name) {
          return &cast<FunctionValue>(**fun.constant_value());
        }
        break;
      }
      default:
        break;
    }
  }
  return std::nullopt;
}

// TODO: Find out a way to remove code duplication
auto MixinPseudoType::FindFunction(const std::string_view& name) const
    -> std::optional<Nonnull<const FunctionValue*>> {
  for (const auto& member : declaration().members()) {
    switch (member->kind()) {
      case DeclarationKind::MixDeclaration: {
        const auto& mix_decl = cast<MixDeclaration>(*member);
        Nonnull<const MixinPseudoType*> mixin = &mix_decl.mixin_value();
        const auto res = mixin->FindFunction(name);
        if (res.has_value()) {
          return res;
        }
        break;
      }
      case DeclarationKind::FunctionDeclaration: {
        const auto& fun = cast<CallableDeclaration>(*member);
        if (fun.name() == name) {
          return &cast<FunctionValue>(**fun.constant_value());
        }
        break;
      }
      default:
        break;
    }
  }
  return std::nullopt;
}

auto FindFunctionWithParents(std::string_view name,
                             const ClassDeclaration& class_decl)
    -> std::optional<Nonnull<const FunctionValue*>> {
  if (auto fun = FindFunction(name, class_decl.members()); fun.has_value()) {
    return fun;
  }
  if (const auto base_type = class_decl.base_type(); base_type.has_value()) {
    return FindFunctionWithParents(name, base_type.value()->declaration());
  }
  return std::nullopt;
}

auto FindMember(std::string_view name,
                llvm::ArrayRef<Nonnull<Declaration*>> members)
    -> std::optional<Nonnull<const Declaration*>> {
  for (Nonnull<const Declaration*> member : members) {
    if (std::optional<std::string_view> mem_name = GetName(*member);
        mem_name.has_value()) {
      if (*mem_name == name) {
        return member;
      }
    }
  }
  return std::nullopt;
}

void ImplBinding::Print(llvm::raw_ostream& out) const {
  out << "impl binding " << *type_var_ << " as " << **iface_;
}

void ImplBinding::PrintID(llvm::raw_ostream& out) const {
  out << *type_var_ << " as " << **iface_;
}

auto NominalClassType::InheritsClass(Nonnull<const Value*> other) const
    -> bool {
  const auto* other_class = dyn_cast<NominalClassType>(other);
  if (!other_class) {
    return false;
  }
  std::optional<Nonnull<const NominalClassType*>> ancestor_class = this;
  while (ancestor_class) {
    if (TypeEqual(*ancestor_class, other_class, std::nullopt)) {
      return true;
    }
    ancestor_class = (*ancestor_class)->base();
  }
  return false;
}

}  // namespace Carbon
