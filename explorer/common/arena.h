// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_COMMON_ARENA_H_
#define CARBON_EXPLORER_COMMON_ARENA_H_

#include <any>
#include <map>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "explorer/common/nonnull.h"
#include "llvm/ADT/Hashing.h"

namespace Carbon {

// Allocates and maintains ownership of arbitrary objects, so that their
// lifetimes all end at the same time. It can also canonicalize the allocated
// objects (see the documentation of New).
class Arena {
  // CanonicalizeAllocation<T>::value is true if canonicalization is enabled
  // for T, and false otherwise.
  template <typename T, typename = void>
  struct CanonicalizeAllocation;

 public:
  // Values of this type can be passed as the first argument to New in order to
  // have the address of the created object written to the given pointer before
  // the constructor is run. This is used during cloning to support pointer
  // cycles within the AST.
  template <typename T>
  struct WriteAddressTo {
    Nonnull<T**> target;
  };
  template <typename T>
  WriteAddressTo(T** target) -> WriteAddressTo<T>;

  // Returns a pointer to an object constructed as if by `T(args...)`, owned
  // by this Arena.
  //
  // If T::EnableCanonicalizedAllocation exists and names a type, this method
  // will canonicalize the allocated objects, meaning that two calls to this
  // method with the same T and equal arguments will return pointers to the same
  // object. If canonicalization is enabled, all types in Args... must be
  // canonicalizable, meaning that they are copyable, and either are
  // equality-comparable and have a hash_value overload as defined in
  // llvm/ADT/Hashing.h, or are `Decompose`able into canonicalizable values.
  // Canonicalization also supports certain standard library types; in
  // particular, std::vector<T> and std::optional<T> are canonicalizable if T
  // is.
  //
  // Canonically-allocated objects must not be mutated, because those mutations
  // would be visible to all users that happened to allocate a T object with the
  // same constructor arguments. To help enforce this, the returned pointer will
  // be const when canonicalization is enabled. Since that means there is no way
  // to allocate a mutable instance of T, canonicalization should only be
  // enabled for types that are inherently immutable.
  //
  // Canonicalization does not guarantee that equal objects will be identical,
  // but it can substantially reduce the incidence of equal-but-not-identical
  // objects, which can facilitate various optimizations.
  template <
      typename T, typename... Args,
      typename std::enable_if_t<std::is_constructible_v<T, Args...> &&
                                !CanonicalizeAllocation<T>::value>* = nullptr>
  auto New(Args&&... args) -> Nonnull<T*>;

  template <
      typename T, typename... Args,
      typename std::enable_if_t<std::is_constructible_v<T, Args...> &&
                                CanonicalizeAllocation<T>::value>* = nullptr>
  auto New(Args&&... args) -> Nonnull<const T*>;

  // Allocates an object in the arena, writing its address to the given pointer.
  template <
      typename T, typename U, typename... Args,
      typename std::enable_if_t<std::is_constructible_v<T, Args...>>* = nullptr>
  void New(WriteAddressTo<U> addr, Args&&... args);

  auto allocated() -> int64_t { return allocated_; }

 private:
  // Virtualizes arena entries so that a single vector can contain many types,
  // avoiding templated statics.
  class ArenaEntry {
   public:
    virtual ~ArenaEntry() = default;
  };

  // Templated destruction of a pointer.
  template <typename T>
  class ArenaEntryTyped;

  // FIXME clean up and document implementation details of canonicalization.

  struct DummyCallback {
    template <typename... Ts>
    void operator()(const Ts&...);
  };

  template <typename T, typename = void>
  struct IsDecomposable : public std::false_type {};

  template <typename T>
  struct IsDecomposable<
      T,
      std::void_t<decltype(std::declval<const T>().Decompose(DummyCallback{}))>>
      : public std::true_type {};

  static constexpr int MaxPriority = 10;
  template <int N>
  struct Priority : Priority<N + 1> {
    static_assert(N < MaxPriority);
  };

  template <>
  struct Priority<MaxPriority> {};

  // Hash functor for tuples of canonicalizable types.
  struct ArgsHash {
    template <typename... Ts>
    auto operator()(const std::tuple<Ts...>& t) const -> size_t {
      return std::apply(
          [](const auto&... elements) {
            return CustomHashCombine(elements...);
          },
          t);
    }

    template <typename... Ts>
    static auto CustomHashCombine(const Ts&... ts) -> llvm::hash_code {
      return llvm::hash_combine(CustomHashValue(Priority<0>{}, ts)...);
    }

    template <typename Iterator>
    static auto CustomHashCombineRange(Iterator begin, Iterator end)
        -> llvm::hash_code {
      llvm::hash_code result = llvm::hash_combine();
      while (begin != end) {
        result = CustomHashCombine(result, *begin);
        ++begin;
      }
      return result;
    }

    template <typename T>
    static constexpr auto IsLlvmHashable() -> bool {
      using llvm::hash_value;
      auto probe = [](const auto& t) -> decltype(hash_value(t)) {};
      return std::is_invocable_r_v<llvm::hash_code, decltype(probe), const T&>;
    }

    template <typename T, typename = void>
    struct IsCustomHashable;

    // We have to exclude implicit conversions to avoid ambiguity, because
    // hash_code is implicitly convertible from size_t.
    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, llvm::hash_code>>>
    static auto CustomHashValue(Priority<0>, T code) -> llvm::hash_code {
      return code;
    }

    template <typename T,
              typename = std::enable_if_t<IsCustomHashable<T>::value>>
    static auto CustomHashValue(Priority<0>, const std::vector<T>& v)
        -> llvm::hash_code {
      return CustomHashCombineRange(v.begin(), v.end());
    }

    // We need this because of optional<Decomposable> ctor parameters.
    template <typename T,
              typename = std::enable_if_t<IsCustomHashable<T>::value>>
    static auto CustomHashValue(Priority<0>, const std::optional<T>& opt)
        -> llvm::hash_code {
      if (opt.has_value()) {
        return CustomHashCombine(*opt);
      } else {
        return CustomHashCombine(std::nullopt);
      }
    }

    static auto CustomHashValue(Priority<0>, std::nullopt_t)
        -> llvm::hash_code {
      return llvm::hash_combine();
    }

    template <typename T, typename = std::enable_if_t<IsLlvmHashable<T>()>>
    static auto CustomHashValue(Priority<1>, const T& t) -> llvm::hash_code {
      using llvm::hash_value;
      return hash_value(t);
    }

    template <typename T,
              typename = std::enable_if_t<IsDecomposable<T>::value &&
                                          std::is_copy_constructible_v<T>>>
    static auto CustomHashValue(Priority<2>, const T& t) -> llvm::hash_code {
      return t.Decompose([](auto&&... us) { return CustomHashCombine(us...); });
    }

    template <typename T, typename>
    struct IsCustomHashable : public std::false_type {};

    template <typename T>
    struct IsCustomHashable<T, std::void_t<decltype(CustomHashValue(
                                   Priority<0>{}, std::declval<const T>()))>>
        : public std::true_type {};
  };

  struct ArgsEqual {
    template <typename... Ts>
    auto operator()(const std::tuple<Ts...>& lhs,
                    const std::tuple<Ts...>& rhs) const -> bool {
      return std::apply(
          [&](const auto&... lhs_elements) {
            return std::apply(
                [&](const auto&... rhs_elements) {
                  return (Equals(Priority<0>{}, lhs_elements, rhs_elements) &&
                          ...);
                },
                rhs);
          },
          lhs);
    }

    template <typename T, typename = void>
    struct SupportsEquals;

    template <typename T, typename = std::enable_if_t<SupportsEquals<T>::value>>
    static auto Equals(Priority<0>, const std::vector<T>& lhs,
                       const std::vector<T>& rhs) -> bool {
      if (lhs.size() != rhs.size()) {
        return false;
      }
      auto lhs_it = lhs.begin();
      auto rhs_it = rhs.begin();
      while (lhs_it != lhs.end()) {
        if (!Equals(Priority<0>{}, *lhs_it, *rhs_it)) {
          return false;
        }
        ++lhs_it;
        ++rhs_it;
      }
      return true;
    }

    template <typename T, typename = std::enable_if_t<SupportsEquals<T>::value>>
    static auto Equals(Priority<0>, const std::optional<T>& lhs,
                       const std::optional<T>& rhs) -> bool {
      if (lhs.has_value() && rhs.has_value()) {
        return Equals(Priority<0>{}, *lhs, *rhs);
      } else {
        return lhs.has_value() == rhs.has_value();
      }
    }

    static auto Equals(Priority<0>, std::nullopt_t, std::nullopt_t) -> bool {
      return true;
    }

    template <typename T,
              typename = std::enable_if_t<std::is_convertible_v<
                  decltype(std::declval<const T>() == std::declval<const T>()),
                  bool>>>
    static auto Equals(Priority<1>, const T& lhs, const T& rhs) -> bool {
      return lhs == rhs;
    }

    template <typename T,
              typename = std::enable_if_t<IsDecomposable<T>::value &&
                                          std::is_copy_constructible_v<T>>>
    static auto Equals(Priority<2>, const T& lhs, const T& rhs) -> bool {
      return lhs.Decompose([&](auto&&... lhs_elements) {
        return rhs.Decompose([&](auto&&... rhs_elements) {
          return (Equals(Priority<0>{}, lhs_elements, rhs_elements) && ...);
        });
      });
    }

    template <typename T, typename>
    struct SupportsEquals : public std::false_type {};

    template <typename T>
    struct SupportsEquals<
        T, std::void_t<decltype(Equals(Priority<0>{}, std::declval<const T>(),
                                       std::declval<const T>()))>>
        : public std::true_type {};
  };

  // Factory metafunction for globally unique type IDs.
  // &TypeId<T>::id == &TypeId<U>::id if and only if std::is_same_v<T,U>.
  //
  // Inspired by llvm::Any::TypeId.
  template <typename T>
  struct TypeId {
    // This is only used for an address to compare; the value is unimportant.
    static char id;
  };

  // A canonicalization table maps a tuple of constructor argument values to
  // a non-null pointer to a T object constructed with those arguments.
  template <typename T, typename... Args>
  using CanonicalizationTable =
      std::unordered_map<std::tuple<Args...>, Nonnull<const T*>, ArgsHash,
                         ArgsEqual>;

  // Allocates an object in the arena. Unlike New, this will always allocate
  // and construct a new object.
  template <typename T, typename... Args>
  auto UniqueNew(Args&&... args) -> Nonnull<T*>;

  // Returns a pointer to the canonical instance of T constructed from
  // `args...`, or null if there is no such instance yet. Returns a mutable
  // reference so that a null entry can be updated.
  template <typename T, typename... Args>
  auto CanonicalInstance(const Args&... args) -> const T*&;

  // Manages allocations in an arena for destruction at shutdown.
  std::vector<std::unique_ptr<ArenaEntry>> arena_;
  int64_t allocated_ = 0;

  // Maps a CanonicalizationTable type to a unique instance of that type for
  // this arena. For a key equal to &TypeId<T>::id for some T, the corresponding
  // value contains a T*.
  std::map<char*, std::any> canonical_tables_;
};

// ---------------------------------------
// Implementation details only below here.
// ---------------------------------------

template <typename T, typename... Args,
          typename std::enable_if_t<std::is_constructible_v<T, Args...> &&
                                    !Arena::CanonicalizeAllocation<T>::value>*>
auto Arena::New(Args&&... args) -> Nonnull<T*> {
  return UniqueNew<T>(std::forward<Args>(args)...);
}

template <typename T, typename... Args,
          typename std::enable_if_t<std::is_constructible_v<T, Args...> &&
                                    Arena::CanonicalizeAllocation<T>::value>*>
auto Arena::New(Args&&... args) -> Nonnull<const T*> {
  const T*& canonical_instance = CanonicalInstance<T>(args...);
  if (canonical_instance == nullptr) {
    canonical_instance = UniqueNew<T>(std::forward<Args>(args)...);
  }
  return canonical_instance;
}

template <typename T, typename U, typename... Args,
          typename std::enable_if_t<std::is_constructible_v<T, Args...>>*>
void Arena::New(WriteAddressTo<U> addr, Args&&... args) {
  static_assert(!CanonicalizeAllocation<T>::value,
                "This form of New does not support canonicalization yet");
  arena_.push_back(
      std::make_unique<ArenaEntryTyped<T>>(addr, std::forward<Args>(args)...));
  allocated_ += sizeof(T);
}

template <typename T, typename... Args>
auto Arena::UniqueNew(Args&&... args) -> Nonnull<T*> {
  auto smart_ptr =
      std::make_unique<ArenaEntryTyped<T>>(std::forward<Args>(args)...);
  Nonnull<T*> ptr = smart_ptr->Instance();
  arena_.push_back(std::move(smart_ptr));
  allocated_ += sizeof(T);
  return ptr;
}

template <typename T, typename>
struct Arena::CanonicalizeAllocation : public std::false_type {};

template <typename T>
struct Arena::CanonicalizeAllocation<
    T, std::void_t<typename T::EnableCanonicalizedAllocation>>
    : public std::true_type {};

template <typename T, typename... Args>
auto Arena::CanonicalInstance(const Args&... args) -> const T*& {
  using MapType = CanonicalizationTable<T, Args...>;
  std::any& wrapped_table = canonical_tables_[&TypeId<MapType>::id];
  if (!wrapped_table.has_value()) {
    wrapped_table.emplace<MapType>();
  }
  MapType& table = std::any_cast<MapType&>(wrapped_table);
  return table[typename MapType::key_type(args...)];
}

// Templated destruction of a pointer.
template <typename T>
class Arena::ArenaEntryTyped : public ArenaEntry {
 public:
  struct WriteAddressHelper {};

  template <typename... Args>
  explicit ArenaEntryTyped(Args&&... args)
      : instance_(std::forward<Args>(args)...) {}

  template <typename... Args>
  explicit ArenaEntryTyped(WriteAddressHelper, Args&&... args)
      : ArenaEntryTyped(std::forward<Args>(args)...) {}

  template <typename U, typename... Args>
  explicit ArenaEntryTyped(WriteAddressTo<U> write_address, Args&&... args)
      : ArenaEntryTyped(
            (*write_address.target = &instance_, WriteAddressHelper{}),
            std::forward<Args>(args)...) {}

  auto Instance() -> Nonnull<T*> { return Nonnull<T*>(&instance_); }

 private:
  T instance_;
};

template <typename T>
char Arena::TypeId<T>::id = 1;

}  // namespace Carbon

#endif  // CARBON_EXPLORER_COMMON_ARENA_H_
