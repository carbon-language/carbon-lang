// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_VALUE_TRANSFORM_H_
#define CARBON_EXPLORER_INTERPRETER_VALUE_TRANSFORM_H_

#include "explorer/interpreter/value.h"

namespace Carbon {

// A no-op visitor used to implement `IsRecursivelyTransformable`. The
// `operator()` function returns `true_type` if it's called with arguments that
// can be used to construct `T`, and `false_type` otherwise.
template <typename T>
struct IsRecursivelyTransformableVisitor {
  template <typename... Args>
  auto operator()(Args&&... args)
      -> std::integral_constant<bool, std::is_constructible_v<T, Args...>>;
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
constexpr bool IsRecursivelyTransformable<
    T, decltype(std::declval<const T>().Decompose(
           IsRecursivelyTransformableVisitor<T>{}))> = true;

// Base class for transforms of visitable data types.
template <typename Derived>
class TransformBase {
 public:
  TransformBase(Nonnull<Arena*> arena) : arena_(arena) {}

  template <typename T>
  auto Transform(T&& v) -> decltype(auto) {
    return static_cast<Derived&>(*this)(std::forward<T>(v));
  }

  // Transformable values are recursively transformed by default.
  template <typename T,
            std::enable_if_t<IsRecursivelyTransformable<T>, void*> = nullptr>
  auto operator()(const T& value) -> T {
    return value.Decompose([&](auto&&... elements) {
      return T{Transform(decltype(elements)(elements))...};
    });
  }

  // Transformable pointers are recursively transformed and reallocated by
  // default.
  template <typename T,
            std::enable_if_t<IsRecursivelyTransformable<T>, void*> = nullptr>
  auto operator()(Nonnull<const T*> value) -> auto{
    return value->Decompose([&](auto&&... elements) {
      return AllocateTrait<T>::New(arena_,
                                   Transform(decltype(elements)(elements))...);
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
    return Transform(*v);
  }

  // Transform `pair<T, U>` by transforming T and U.
  template <typename T, typename U>
  auto operator()(const std::pair<T, U>& pair) -> std::pair<T, U> {
    return std::pair<T, U>{Transform(pair.first), Transform(pair.second)};
  }

  // Transform `vector<T>` by transforming its elements.
  template <typename T>
  auto operator()(const std::vector<T>& vec) -> std::vector<T> {
    std::vector<T> result;
    result.reserve(vec.size());
    for (auto& value : vec) {
      result.push_back(Transform(value));
    }
    return result;
  }

  // Transform `map<T, U>` by transforming its keys and values.
  template <typename T, typename U>
  auto operator()(const std::map<T, U>& map) -> std::map<T, U> {
    std::map<T, U> result;
    for (auto& [key, value] : map) {
      result.insert({Transform(key), Transform(value)});
    }
    return result;
  }

  // Transform `llvm::StringMap<T>` by transforming its keys and values.
  template <typename T>
  auto operator()(const llvm::StringMap<T>& map) -> llvm::StringMap<T> {
    llvm::StringMap<T> result;
    for (const auto& it : map) {
      result.insert({Transform(it.first()), Transform(it.second)});
    }
    return result;
  }

 private:
  Nonnull<Arena*> arena_;
};

// Base class for transforms of `Value`s.
template <typename Derived>
class ValueTransform : public TransformBase<Derived> {
 public:
  using TransformBase<Derived>::TransformBase;
  using TransformBase<Derived>::operator();

  // Leave references to AST nodes alone by default.
  // The 'int = 0' parameter avoids this function hiding the `operator()(const
  // T*)` in the base class. We can remove this once we start using a compiler
  // that implements P1787R6.
  template <typename NodeT>
  auto operator()(Nonnull<const NodeT*> node, int = 0)
      -> std::enable_if_t<std::is_base_of_v<AstNode, NodeT>,
                          Nonnull<const NodeT*>> {
    return node;
  }

  auto operator()(Nonnull<ContinuationValue::StackFragment*> stack_fragment)
      -> Nonnull<ContinuationValue::StackFragment*> {
    return stack_fragment;
  }

  auto operator()(Address addr) -> Address { return addr; }

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
      return this->Transform(derived_value);
    });
  }

  // For values, dispatch on the value kind and recursively transform.
  auto operator()(Nonnull<const Value*> value) -> Nonnull<const Value*> {
    return TransformDerived<Nonnull<const Value*>>(value);
  }

  // Provide a more precise type from transforming a `Witness`.
  auto operator()(Nonnull<const Witness*> value) -> Nonnull<const Witness*> {
    return llvm::cast<Witness>(this->Transform(llvm::cast<Value>(value)));
  }

  // For elements, dispatch on the element kind and recursively transform.
  auto operator()(Nonnull<const Element*> elem) -> Nonnull<const Element*> {
    return TransformDerived<Nonnull<const Element*>>(elem);
  }

  // Preserve vtable during transformation.
  auto operator()(Nonnull<const VTable* const> vtable)
      -> Nonnull<const VTable* const> {
    return vtable;
  }

  // Preserve class value ptr during transformation.
  auto operator()(Nonnull<const NominalClassValue** const> value_ptr)
      -> Nonnull<const NominalClassValue** const> {
    return value_ptr;
  }
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_VALUE_TRANSFORM_H_
