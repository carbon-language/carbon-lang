// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/static_scope.h"

#include <optional>

#include "common/ostream.h"
#include "explorer/base/error_builders.h"
#include "explorer/base/print_as_id.h"
#include "llvm/ADT/ScopeExit.h"

namespace Carbon {

auto StaticScope::Add(std::string_view name, ValueNodeView entity,
                      NameStatus status) -> ErrorOr<Success> {
  auto [it, inserted] = declared_names_.insert({name, {entity, status}});
  if (!inserted) {
    if (it->second.entity != entity) {
      return ProgramError(entity.base().source_loc())
             << "Duplicate name `" << name << "` also found at "
             << it->second.entity.base().source_loc();
    }
    if (static_cast<int>(status) > static_cast<int>(it->second.status)) {
      it->second.status = status;
    }
  } else {
    if (trace_stream_->is_enabled()) {
      trace_stream_->Result()
          << "declared `" << name << "` as `" << entity << "` in `"
          << PrintAsID(*this) << "` (" << entity.base().source_loc() << ")\n";
    }
  }
  return Success();
}

template <typename Action>
void StaticScope::PrintCommon(Action action) const {
  if (ast_node_) {
    action(ast_node_.value());
  } else {
    *trace_stream_ << "package";
  }
}

void StaticScope::Print(llvm::raw_ostream& out) const {
  PrintCommon([&out](auto node) { node->Print(out); });
}

void StaticScope::PrintID(llvm::raw_ostream& out) const {
  PrintCommon([&out](auto node) { node->PrintID(out); });
}

void StaticScope::MarkDeclared(std::string_view name) {
  auto it = declared_names_.find(name);
  CARBON_CHECK(it != declared_names_.end()) << name << " not found";
  if (it->second.status == NameStatus::KnownButNotDeclared) {
    it->second.status = NameStatus::DeclaredButNotUsable;
    if (trace_stream_->is_enabled()) {
      trace_stream_->Result()
          << "marked `" << name << "` declared but not usable in `"
          << PrintAsID(*this) << "`\n";
    }
  }
}

void StaticScope::MarkUsable(std::string_view name) {
  auto it = declared_names_.find(name);
  CARBON_CHECK(it != declared_names_.end()) << name << " not found";
  it->second.status = NameStatus::Usable;
  if (trace_stream_->is_enabled()) {
    trace_stream_->Result()
        << "marked `" << name << "` usable in `" << PrintAsID(*this) << "`\n";
  }
}

auto StaticScope::Resolve(std::string_view name,
                          SourceLocation source_loc) const
    -> ErrorOr<ValueNodeView> {
  CARBON_ASSIGN_OR_RETURN(std::optional<ValueNodeView> result,
                          TryResolve(name, source_loc));
  if (!result) {
    return ProgramError(source_loc) << "could not resolve '" << name << "'";
  }
  return *result;
}

auto StaticScope::ResolveHere(std::optional<ValueNodeView> this_scope,
                              std::string_view name, SourceLocation source_loc,
                              bool allow_undeclared) const
    -> ErrorOr<ValueNodeView> {
  CARBON_ASSIGN_OR_RETURN(std::optional<ValueNodeView> result,
                          TryResolveHere(name, source_loc, allow_undeclared));
  if (!result) {
    if (this_scope) {
      return ProgramError(source_loc)
             << "name '" << name << "' has not been declared in "
             << PrintAsID(this_scope->base());
    } else {
      return ProgramError(source_loc)
             << "name '" << name << "' has not been declared in this scope";
    }
  }
  return *result;
}

auto StaticScope::TryResolve(std::string_view name,
                             SourceLocation source_loc) const
    -> ErrorOr<std::optional<ValueNodeView>> {
  for (const StaticScope* scope = this; scope;
       scope = scope->parent_scope_.value_or(nullptr)) {
    CARBON_ASSIGN_OR_RETURN(
        std::optional<ValueNodeView> value,
        scope->TryResolveHere(name, source_loc, /*allow_undeclared=*/false));
    if (value) {
      return value;
    }
  }
  return {std::nullopt};
}

auto StaticScope::TryResolveHere(std::string_view name,
                                 SourceLocation source_loc,
                                 bool allow_undeclared) const
    -> ErrorOr<std::optional<ValueNodeView>> {
  auto it = declared_names_.find(name);
  if (it == declared_names_.end()) {
    return {std::nullopt};
  }

  auto exit_scope_function = llvm::make_scope_exit([&]() {
    if (trace_stream_->is_enabled()) {
      trace_stream_->Result()
          << "resolved `" << name << "` as `" << it->second.entity << "` in `"
          << PrintAsID(*this) << "` (" << source_loc << ")\n";
    }
  });

  if (allow_undeclared || it->second.status == NameStatus::Usable) {
    return {it->second.entity};
  }
  return ProgramError(source_loc)
         << "'" << name
         << (it->second.status == NameStatus::KnownButNotDeclared
                 ? "' has not been declared yet"
                 : "' is not usable until after it has been completely "
                   "declared");
}

auto StaticScope::AddReturnedVar(ValueNodeView returned_var_def_view)
    -> ErrorOr<Success> {
  std::optional<ValueNodeView> resolved_returned_var = ResolveReturned();
  if (resolved_returned_var.has_value()) {
    return ProgramError(returned_var_def_view.base().source_loc())
           << "Duplicate definition of returned var also found at "
           << resolved_returned_var->base().source_loc();
  }
  returned_var_def_view_ = std::move(returned_var_def_view);
  return Success();
}

auto StaticScope::ResolveReturned() const -> std::optional<ValueNodeView> {
  for (const StaticScope* scope = this; scope;
       scope = scope->parent_scope_.value_or(nullptr)) {
    if (scope->returned_var_def_view_.has_value()) {
      return scope->returned_var_def_view_;
    }
  }
  return std::nullopt;
}

}  // namespace Carbon
