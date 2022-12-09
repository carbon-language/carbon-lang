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

  // Transform `optional<T>` by transforming the `T` if it's present.
  template <typename T>
  auto operator()(const std::optional<T>& v) -> std::optional<T> {
    if (!v) {
      return std::nullopt;
    }
    return Transform(*v);
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
  template <typename NodeT>
  auto operator()(Nonnull<const NodeT*> node)
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

  // For values, dispatch on the value kind and recursively transform.
  auto operator()(Nonnull<const Value*> value) -> Nonnull<const Value*> {
    switch (value->kind()) {
#define CARBON_VALUE_KIND(T)                      \
  case Value::Kind::T:                            \
    static_assert(IsRecursivelyTransformable<T>); \
    return this->Transform(llvm::cast<T>(value));
#include "explorer/interpreter/value_kinds.def"
    }
  }
  auto operator()(Nonnull<const Witness*> value) -> Nonnull<const Witness*> {
    return llvm::cast<Witness>(this->Transform(llvm::cast<Value>(value)));
  }
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_VALUE_TRANSFORM_H_
