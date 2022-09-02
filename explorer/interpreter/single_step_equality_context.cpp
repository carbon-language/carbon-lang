#include "explorer/interpreter/single_step_equality_context.h"

namespace Carbon {

struct TypeChecker::SingleStepEqualityContext : public EqualityContext {
 public:
  SingleStepEqualityContext(Nonnull<const TypeChecker*> type_checker,
                            Nonnull<const ImplScope*> impl_scope)
      : type_checker_(type_checker), impl_scope_(impl_scope) {}

  // Attempt to resolve the witness for the given associated constant in the
  // in-scope `impl`s.
  auto TryResolveWitness(Nonnull<const AssociatedConstant*> assoc,
                         SourceLocation source_loc) const
      -> ErrorOr<Nonnull<const ImplWitness*>> {
    auto* impl_witness = dyn_cast<ImplWitness>(&assoc->witness());
    if (impl_witness) {
      return impl_witness;
    }

    CARBON_ASSIGN_OR_RETURN(
        Nonnull<const Expression*> witness_expr,
        impl_scope_->Resolve(&assoc->interface(), &assoc->base(), source_loc,
                             *type_checker_));
    CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> witness_value,
                            InterpExp(witness_expr, type_checker_->arena_,
                                      type_checker_->trace_stream_));
    impl_witness = dyn_cast<ImplWitness>(witness_value);
    if (impl_witness) {
      return impl_witness;
    }
    return CompilationError(source_loc)
           << "value of associated constant " << *assoc
           << " depends on a generic parameter";
  }

  // Visits the values that are equal to the given value and a single step away
  // according to an equality constraint that is either scope or within a final
  // impl corresponding to an associated constant. Stops and returns `false` if
  // the visitor returns `false`, otherwise returns `true`.
  auto VisitEqualValues(Nonnull<const Value*> value,
                        llvm::function_ref<bool(Nonnull<const Value*>)> visitor)
      const -> bool override {
    if (type_checker_->trace_stream_) {
      **type_checker_->trace_stream_ << "looking for values equal to " << *value
                                     << " in\n"
                                     << *impl_scope_;
    }

    if (!impl_scope_->VisitEqualValues(value, visitor)) {
      return false;
    }

    // Also look up and visit the corresponding impl if this is an associated
    // constant.
    if (auto* assoc = dyn_cast<AssociatedConstant>(value)) {
      // Perform an impl lookup to see if we can resolve this constant.
      // The source location doesn't matter, we're discarding the diagnostics.
      SourceLocation source_loc("", 0);
      ErrorOr<Nonnull<const ImplWitness*>> impl_witness =
          TryResolveWitness(assoc, source_loc);
      if (impl_witness.ok()) {
        // Instantiate the impl to find the concrete constraint it implements.
        Nonnull<const ConstraintType*> constraint =
            (*impl_witness)->declaration().constraint_type();
        BindingMap bindings = (*impl_witness)->type_args();
        bindings[constraint->self_binding()] = &assoc->base();
        constraint = cast<ConstraintType>(
            type_checker_->Substitute(bindings, constraint));

        // Look for the value of this constant within that constraint.
        if (!constraint->VisitEqualValues(value, visitor)) {
          return false;
        }
      } else {
        if (type_checker_->trace_stream_) {
          **type_checker_->trace_stream_
              << "Could not resolve associated constant " << *assoc << ": "
              << impl_witness.error() << "\n";
        }
      }
    }

    return true;
  }

 private:
  Nonnull<const TypeChecker*> type_checker_;
  Nonnull<const ImplScope*> impl_scope_;
};

};  // namespace Carbon