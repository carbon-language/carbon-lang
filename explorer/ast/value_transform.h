// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_VALUE_TRANSFORM_H_
#define CARBON_EXPLORER_AST_VALUE_TRANSFORM_H_

#include "common/error.h"
#include "explorer/ast/expression_category.h"
#include "explorer/ast/value.h"

namespace Carbon {

// Constructs a T instance by direct-list-initialization from the given
// components (which must have been produced by Decompose).
template <typename T, typename... Args>
auto ConstructFromComponents(Args&&... args)
    -> decltype(T{std::declval<Args>()...}) {
  return T{std::forward<Args>(args)...};
}

// Overload of the above to accommodate the case where T is an aggregate and
// has a CRTP base class, in which case the initializer list must start with
// an empty initializer for the base class.
template <typename T, typename... Args>
auto ConstructFromComponents(Args&&... args)
    -> decltype(T{{}, std::declval<Args>()...}) {
  return T{{}, std::forward<Args>(args)...};
}

template <typename T, typename, typename... Args>
constexpr bool IsConstructibleFromComponentsImpl = false;

template <typename T, typename... Args>
constexpr bool IsConstructibleFromComponentsImpl<
    T, decltype(ConstructFromComponents<T>(std::declval<Args>()...)), Args...> =
    true;

// A no-op visitor used to implement `IsRecursivelyTransformable`. The
// `operator()` function returns `true_type` if it's called with arguments that
// can be used to direct-list-initialize `T`, and `false_type` otherwise.
template <typename T>
struct IsRecursivelyTransformableVisitor {
  template <typename... Args>
  auto operator()(Args&&... args) -> std::integral_constant<
      bool, IsConstructibleFromComponentsImpl<T, T, Args...>>;
};

// A type trait that indicates whether `T` is transformable. A transformable
// type provides a function
//
// template<typename F> void Decompose(F f) const;
//
// that takes a callable `f` and passes it an argument list that can be passed
// to the constructor of `T` to create an equivalent value.
template <typename T, typename = std::true_type>
constexpr bool IsRecursivelyTransformable = false;
template <typename T>
// NOLINTNEXTLINE(misc-definitions-in-headers)
constexpr bool IsRecursivelyTransformable<
    T, decltype(std::declval<const T>().Decompose(
           IsRecursivelyTransformableVisitor<T>{}))> = true;

// Unwrapper for the case where there's nothing to unwrap.
class NoOpUnwrapper {
 public:
  template <typename T, typename U>
  auto UnwrapOr(T&& value, const U&) -> T {
    return std::forward<T>(value);
  }

  template <typename T>
  auto Wrap(T&& value) -> T&& {
    return std::forward<T>(value);
  }

  constexpr bool failed() const { return false; }
};

// Helper to temporarily unwrap the ErrorOr around a value, and then put it
// back when we're done with the overall computation.
class ErrorUnwrapper {
 public:
  // Unwrap the `ErrorOr` from the given value, or collect the error and return
  // the given fallback value on failure.
  template <typename T, typename U>
  auto UnwrapOr(ErrorOr<T> value, const U& fallback) -> T {
    if (!value.ok()) {
      status_ = std::move(value).error();
      return fallback;
    }
    return std::move(*value);
  }
  template <typename T, typename U>
  auto UnwrapOr(T&& value, const U&) -> T {
    return std::forward<T>(value);
  }

  // Wrap the given value into `ErrorOr`, returning our collected error if any,
  // or the given value if we succeeded.
  template <typename T>
  auto Wrap(T&& value) -> ErrorOr<T> {
    if (!status_.ok()) {
      Error error = std::move(status_).error();
      status_ = Success();
      return error;
    }
    return std::forward<T>(value);
  }

  bool failed() const { return !status_.ok(); }

 private:
  ErrorOr<Success> status_ = Success();
};

// Base class for transforms of visitable data types.
template <typename Derived, typename ResultUnwrapper>
class TransformBase {
 public:
  explicit TransformBase(Nonnull<Arena*> arena) : arena_(arena) {}

  // Transform the given value, and produce either the transformed value or an
  // error.
  template <typename T>
  auto Transform(const T& v) -> decltype(auto) {
    return unwrapper_.Wrap(TransformOrOriginal(v));
  }

 protected:
  // Transform the given value, or return the original if transformation fails.
  template <typename T>
  auto TransformOrOriginal(const T& v)
      -> decltype(std::declval<ResultUnwrapper>().UnwrapOr(
          std::declval<Derived>()(v), v)) {
    // If we've already failed, don't do any more transformations.
    if (unwrapper_.failed()) {
      return v;
    }
    return unwrapper_.UnwrapOr(static_cast<Derived&>(*this)(v), v);
  }

  // Transformable values are recursively transformed by default.
  template <typename T,
            std::enable_if_t<IsRecursivelyTransformable<T>, void*> = nullptr>
  auto operator()(const T& value) -> T {
    return value.Decompose([&](const auto&... elements) {
      return [&](auto&&... transformed_elements) {
        if (unwrapper_.failed()) {
          return value;
        }
        return ConstructFromComponents<T>(
            decltype(transformed_elements)(transformed_elements)...);
      }(TransformOrOriginal(elements)...);
    });
  }

  // Transformable pointers are recursively transformed and reallocated by
  // default.
  template <typename T,
            std::enable_if_t<IsRecursivelyTransformable<T>, void*> = nullptr>
  auto operator()(Nonnull<const T*> value) -> auto {
    return value->Decompose([&](const auto&... elements) {
      return [&](auto&&... transformed_elements)
                 -> decltype(AllocateTrait<T>::New(
                     arena_,
                     decltype(transformed_elements)(transformed_elements)...)) {
        if (unwrapper_.failed()) {
          return value;
        }
        return AllocateTrait<T>::New(
            arena_, decltype(transformed_elements)(transformed_elements)...);
      }(TransformOrOriginal(elements)...);
    });
  }

  // Fundamental types like `int` are assumed to not need transformation.
  template <typename T>
  auto operator()(const T& v) -> std::enable_if_t<std::is_fundamental_v<T>, T> {
    return v;
  }
  auto operator()(const std::string& str) -> const std::string& { return str; }
  auto operator()(llvm::StringRef str) -> llvm::StringRef { return str; }

  // Transform `optional<T>` by transforming the `T` if it's present.
  template <typename T>
  auto operator()(const std::optional<T>& v) -> std::optional<T> {
    if (!v) {
      return std::nullopt;
    }
    return TransformOrOriginal(*v);
  }

  // Transform `pair<T, U>` by transforming T and U.
  template <typename T, typename U>
  auto operator()(const std::pair<T, U>& pair) -> std::pair<T, U> {
    return std::pair<T, U>{TransformOrOriginal(pair.first),
                           TransformOrOriginal(pair.second)};
  }

  // Transform `vector<T>` by transforming its elements.
  template <typename T>
  auto operator()(const std::vector<T>& vec) -> std::vector<T> {
    std::vector<T> result;
    result.reserve(vec.size());
    for (auto& value : vec) {
      result.push_back(TransformOrOriginal(value));
    }
    return result;
  }

  // Transform `map<T, U>` by transforming its keys and values.
  template <typename T, typename U>
  auto operator()(const std::map<T, U>& map) -> std::map<T, U> {
    std::map<T, U> result;
    for (auto& [key, value] : map) {
      result.insert({TransformOrOriginal(key), TransformOrOriginal(value)});
    }
    return result;
  }

  // Transform `llvm::StringMap<T>` by transforming its keys and values.
  template <typename T>
  auto operator()(const llvm::StringMap<T>& map) -> llvm::StringMap<T> {
    llvm::StringMap<T> result;
    for (const auto& it : map) {
      result.insert(
          {TransformOrOriginal(it.first()), TransformOrOriginal(it.second)});
    }
    return result;
  }

 private:
  Nonnull<Arena*> arena_;
  // Unwrapper for results. Used to remove an ErrorOr<...> wrapper temporarily
  // during recursive transformations and re-apply it when we're done.
  ResultUnwrapper unwrapper_;
};

// Base class for transforms of `Value`s.
template <typename Derived, typename ResultUnwrapper>
class ValueTransform : public TransformBase<Derived, ResultUnwrapper> {
 public:
  using TransformBase<Derived, ResultUnwrapper>::TransformBase;
  using TransformBase<Derived, ResultUnwrapper>::operator();

  // Leave references to AST nodes alone by default.
  // The 'int = 0' parameter avoids this function hiding the `operator()(const
  // T*)` in the base class. We can remove this once we start using a compiler
  // that implements P1787R6.
  template <typename NodeT>
  auto operator()(Nonnull<const NodeT*> node, int /*unused*/ = 0)
      -> std::enable_if_t<std::is_base_of_v<AstNode, NodeT>,
                          Nonnull<const NodeT*>> {
    return node;
  }

  auto operator()(Address addr) -> Address { return addr; }

  auto operator()(ExpressionCategory cat) -> ExpressionCategory { return cat; }

  auto operator()(ValueNodeView value_node) -> ValueNodeView {
    return value_node;
  }

  // For a type that provides a `Visit` function to visit the most-derived
  // object, visit and transform that most-derived object.
  template <typename R, typename T>
  auto TransformDerived(Nonnull<const T*> value) -> R {
    return value->template Visit<R>([&](const auto* derived_value) {
      using DerivedType = std::remove_pointer_t<decltype(derived_value)>;
      static_assert(IsRecursivelyTransformable<DerivedType>);
      return this->TransformOrOriginal(derived_value);
    });
  }

  // For values, dispatch on the value kind and recursively transform.
  auto operator()(Nonnull<const Value*> value) -> Nonnull<const Value*> {
    return TransformDerived<Nonnull<const Value*>>(value);
  }

  // Provide a more precise type from transforming a `Witness`.
  auto operator()(Nonnull<const Witness*> value) -> Nonnull<const Witness*> {
    return llvm::cast<Witness>(
        this->TransformOrOriginal(llvm::cast<Value>(value)));
  }

  // For elements, dispatch on the element kind and recursively transform.
  auto operator()(Nonnull<const Element*> elem) -> Nonnull<const Element*> {
    return TransformDerived<Nonnull<const Element*>>(elem);
  }

  // Preserve vtable during transformation.
  auto operator()(Nonnull<const VTable*> vtable) -> Nonnull<const VTable*> {
    return vtable;
  }

  // Preserve class value ptr during transformation.
  auto operator()(Nonnull<const NominalClassValue**> value_ptr)
      -> Nonnull<const NominalClassValue**> {
    return value_ptr;
  }

  // Preserve constraint kind for intrinsic constraints.
  auto operator()(IntrinsicConstraint::Kind kind) -> IntrinsicConstraint::Kind {
    return kind;
  }
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_VALUE_TRANSFORM_H_
