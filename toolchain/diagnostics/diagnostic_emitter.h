// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_EMITTER_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_EMITTER_H_

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#include "common/check.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/diagnostics/diagnostic_consumer.h"
#include "toolchain/diagnostics/diagnostic_kind.h"

namespace Carbon::Diagnostics {

namespace Internal {

// Disable type deduction based on `args`; the type of `diagnostic_base`
// determines the diagnostic's parameter types.
template <typename Arg>
using NoTypeDeduction = std::type_identity_t<Arg>;

}  // namespace Internal

template <typename LocT, typename AnnotateFn>
class AnnotationScope;

// The result of `DiagnosticConvert::ConvertLoc`. This is non-templated to allow
// sharing across converters.
struct ConvertedLoc {
  // Becomes Message::loc.
  Loc loc;
  // Becomes Diagnostic::last_byte_offset.
  int32_t last_byte_offset;
};

// Used by types to indicate a diagnostic type conversion that results in the
// provided StorageType. For example, to convert NameId to a std::string, we
// write:
//
// struct NameId {
//   using DiagnosticType = Diagnostics::TypeInfo<std::string>;
// };
template <typename StorageTypeT>
struct TypeInfo {
  using StorageType = StorageTypeT;
};

// Manages the creation of reports, the testing if diagnostics are enabled, and
// the collection of reports.
//
// This class is parameterized by a location type, allowing different
// diagnostic clients to provide location information in whatever form is most
// convenient for them, such as a position within a buffer when lexing, a token
// when parsing, or a parse tree node when type-checking, and to allow unit
// tests to be decoupled from any concrete location representation.
template <typename LocT>
class Emitter {
 public:
  // A builder-pattern type to provide a fluent interface for constructing
  // a more complex diagnostic. See `Emitter::Build` for the
  // expected usage.
  // This is nodiscard to protect against accidentally building a diagnostic
  // without emitting it.
  class [[nodiscard]] Builder {
   public:
    // Builder is move-only and cannot be copied.
    Builder(Builder&&) noexcept = default;
    auto operator=(Builder&&) noexcept -> Builder& = default;

    // Adds a note diagnostic attached to the main diagnostic being built.
    // The API mirrors the main emission API: `Emitter::Emit`.
    // For the expected usage see the builder API: `Emitter::Build`.
    template <typename... Args>
    auto Note(LocT loc, const DiagnosticBase<Args...>& diagnostic_base,
              Internal::NoTypeDeduction<Args>... args) -> Builder&;

    // Emits the built diagnostic and its attached notes.
    // For the expected usage see the builder API: `Emitter::Build`.
    template <typename... Args>
    auto Emit() & -> void;

    // Prevent trivial uses of the builder; always `static_assert`s.
    template <typename... Args>
    auto Emit() && -> void;

    // Returns true if this Builder may emit a diagnostic. Can be used
    // to avoid excess work computing notes, etc, if no diagnostic is going to
    // be emitted anyway.
    explicit operator bool() { return emitter_; }

   private:
    friend class Emitter<LocT>;

    template <typename... Args>
    explicit Builder(Emitter<LocT>* emitter, LocT loc,
                     const DiagnosticBase<Args...>& diagnostic_base,
                     llvm::SmallVector<llvm::Any> args);

    // Create a null `Builder` that will not emit anything. Notes will
    // be silently ignored.
    Builder() : emitter_(nullptr) {}

    // Adds a message to the diagnostic, handling conversion of the location and
    // arguments.
    template <typename... Args>
    auto AddMessage(LocT loc, const DiagnosticBase<Args...>& diagnostic_base,
                    llvm::SmallVector<llvm::Any> args) -> void;

    // Adds a message to the diagnostic, handling conversion of the arguments. A
    // Loc must be provided instead of a LocT in order to
    // avoid potential recursion.
    template <typename... Args>
    auto AddMessageWithLoc(Loc loc,
                           const DiagnosticBase<Args...>& diagnostic_base,
                           llvm::SmallVector<llvm::Any> args) -> void;

    // Handles the cast of llvm::Any to Args types for formatv.
    // TODO: Custom formatting can be provided with an format_provider, but that
    // affects all formatv calls. Consider replacing formatv with a custom call
    // that allows diagnostic-specific formatting.
    template <typename... Args, size_t... N>
    static auto FormatFn(const Message& message,
                         std::index_sequence<N...> /*indices*/) -> std::string;

    Emitter<LocT>* emitter_;
    Diagnostic diagnostic_;
  };

  // `consumer` is required to outlive the diagnostic emitter.
  explicit Emitter(Consumer* consumer) : consumer_(consumer) {}

  virtual ~Emitter() = default;

  // Emits an error.
  //
  // When passing arguments, they may be buffered. As a consequence, lifetimes
  // may outlive the `Emit` call.
  template <typename... Args>
  auto Emit(LocT loc, const DiagnosticBase<Args...>& diagnostic_base,
            Internal::NoTypeDeduction<Args>... args) -> void;

  // A fluent interface for building a diagnostic and attaching notes for added
  // context or information. For example:
  //
  //   emitter_.Build(loc1, MyDiagnostic)
  //     .Note(loc2, MyDiagnosticNote)
  //     .Emit();
  template <typename... Args>
  auto Build(LocT loc, const DiagnosticBase<Args...>& diagnostic_base,
             Internal::NoTypeDeduction<Args>... args) -> Builder;

  // Create a null `Builder` that will not emit anything. Notes will
  // be silently ignored.
  auto BuildSuppressed() -> Builder { return Builder(); }

 protected:
  // Callback type used to report context messages from ConvertLoc.
  // Note that the first parameter type is Loc rather than
  // LocT, because ConvertLoc must not recurse.
  using ContextFnT =
      llvm::function_ref<auto(Loc, const DiagnosticBase<>&)->void>;

  // Converts a LocT to a Loc and its `last_byte_offset` (see
  // `Message`). ConvertLoc may invoke context_fn to provide context
  // messages.
  virtual auto ConvertLoc(LocT loc, ContextFnT context_fn) const
      -> ConvertedLoc = 0;

  // Converts arg types as needed. Most children don't customize conversion, so
  // the default returns the argument unchanged.
  virtual auto ConvertArg(llvm::Any arg) const -> llvm::Any { return arg; }

 private:
  // Converts an argument to llvm::Any for storage, handling input to storage
  // type conversion when needed.
  template <typename Arg>
  auto MakeAny(Arg arg) -> llvm::Any;

  template <typename OtherLocT, typename AnnotateFn>
  friend class AnnotationScope;

  Consumer* consumer_;
  llvm::SmallVector<llvm::function_ref<auto(Builder& builder)->void>>
      annotate_fns_;
};

// This relies on `void*` location handling on `Emitter`.
//
// TODO: Based on how this ends up used or if we get more distinct emitters, it
// might be worth considering having diagnostics specify that they don't apply
// to source-location carrying emitters. For example, this might look like a
// `CARBON_NO_LOC_DIAGNOSTIC` macro, or some other factoring. But it might end
// up being more noise than it is worth.
class NoLocEmitter : public Emitter<void*> {
 public:
  using Emitter::Emitter;

  // Emits an error. This specialization only applies to
  // `NoLocEmitter`.
  template <typename... Args>
  auto Emit(const DiagnosticBase<Args...>& diagnostic_base,
            Internal::NoTypeDeduction<Args>... args) -> void {
    Emitter::Emit(nullptr, diagnostic_base, args...);
  }

 protected:
  auto ConvertLoc(void* /*loc*/, ContextFnT /*context_fn*/) const
      -> ConvertedLoc override {
    return {.loc = {.filename = ""}, .last_byte_offset = -1};
  }
};

// An RAII object that denotes a scope in which any diagnostic produced should
// be annotated in some way.
//
// This object is given a function `annotate` that will be called with a
// `Builder& builder` for any diagnostic that is emitted through the
// given emitter. That function can annotate the diagnostic by calling
// `builder.Note` to add notes.
template <typename LocT, typename AnnotateFn>
class AnnotationScope {
 public:
  AnnotationScope(Emitter<LocT>* emitter, AnnotateFn annotate)
      : emitter_(emitter), annotate_(std::move(annotate)) {
    emitter_->annotate_fns_.push_back(annotate_);
  }
  ~AnnotationScope() { emitter_->annotate_fns_.pop_back(); }

 private:
  Emitter<LocT>* emitter_;
  // Make a copy of the annotation function to ensure that it lives long enough.
  AnnotateFn annotate_;
};

template <typename LocT, typename AnnotateFn>
AnnotationScope(Emitter<LocT>* emitter, AnnotateFn annotate)
    -> AnnotationScope<LocT, AnnotateFn>;

// ============================================================================
// Only internal implementation details below this point.
// ============================================================================

namespace Internal {

// Determines whether there's a DiagnosticType member on Arg.
// Used by Emitter.
template <typename Arg>
concept HasDiagnosticType = requires { typename Arg::DiagnosticType; };

// The default implementation with no conversion.
template <typename Arg>
struct DiagnosticTypeForArg : public TypeInfo<Arg> {};

// Exposes a custom conversion for an argument type.
template <typename Arg>
  requires HasDiagnosticType<Arg>
struct DiagnosticTypeForArg<Arg> : public Arg::DiagnosticType {};

}  // namespace Internal

template <typename LocT>
template <typename... Args>
auto Emitter<LocT>::Builder::Note(
    LocT loc, const DiagnosticBase<Args...>& diagnostic_base,
    Internal::NoTypeDeduction<Args>... args) -> Builder& {
  if (!emitter_) {
    return *this;
  }
  CARBON_CHECK(diagnostic_base.Level == Level::Note ||
                   diagnostic_base.Level == Level::LocationInfo,
               "{0}", static_cast<int>(diagnostic_base.Level));
  AddMessage(loc, diagnostic_base, {emitter_->MakeAny<Args>(args)...});
  return *this;
}

template <typename LocT>
template <typename... Args>
auto Emitter<LocT>::Builder::Emit() & -> void {
  if (!emitter_) {
    return;
  }
  for (auto annotate_fn : llvm::reverse(emitter_->annotate_fns_)) {
    annotate_fn(*this);
  }
  emitter_->consumer_->HandleDiagnostic(std::move(diagnostic_));
}

namespace Internal {
template <typename LocT>
concept AlwaysFalse = false;
}  // namespace Internal

template <typename LocT>
template <typename... Args>
auto Emitter<LocT>::Builder::Emit() && -> void {
  // TODO: This is required by clang-16, but `false` may work in newer clang
  // versions. Replace when possible.
  static_assert(Internal::AlwaysFalse<LocT>,
                "Use `emitter.Emit(...)` or "
                "`emitter.Build(...).Note(...).Emit(...)` "
                "instead of `emitter.Build(...).Emit(...)`");
}

template <typename LocT>
template <typename... Args>
Emitter<LocT>::Builder::Builder(Emitter<LocT>* emitter, LocT loc,
                                const DiagnosticBase<Args...>& diagnostic_base,
                                llvm::SmallVector<llvm::Any> args)
    : emitter_(emitter), diagnostic_({.level = diagnostic_base.Level}) {
  AddMessage(loc, diagnostic_base, std::move(args));
  CARBON_CHECK(diagnostic_base.Level != Level::Note);
}

template <typename LocT>
template <typename... Args>
auto Emitter<LocT>::Builder::AddMessage(
    LocT loc, const DiagnosticBase<Args...>& diagnostic_base,
    llvm::SmallVector<llvm::Any> args) -> void {
  if (!emitter_) {
    return;
  }
  auto converted = emitter_->ConvertLoc(
      loc,
      [&](Loc context_loc, const DiagnosticBase<>& context_diagnostic_base) {
        AddMessageWithLoc(context_loc, context_diagnostic_base, {});
      });
  // Use the last byte offset from the first message.
  if (diagnostic_.messages.empty()) {
    diagnostic_.last_byte_offset = converted.last_byte_offset;
  }
  AddMessageWithLoc(converted.loc, diagnostic_base, args);
}

template <typename LocT>
template <typename... Args>
auto Emitter<LocT>::Builder::AddMessageWithLoc(
    Loc loc, const DiagnosticBase<Args...>& diagnostic_base,
    llvm::SmallVector<llvm::Any> args) -> void {
  if (!emitter_) {
    return;
  }
  diagnostic_.messages.emplace_back(
      Message{.kind = diagnostic_base.Kind,
              .level = diagnostic_base.Level,
              .loc = loc,
              .format = diagnostic_base.Format,
              .format_args = std::move(args),
              .format_fn = [](const Message& message) -> std::string {
                return FormatFn<Args...>(
                    message, std::make_index_sequence<sizeof...(Args)>());
              }});
}

template <typename LocT>
template <typename... Args, size_t... N>
auto Emitter<LocT>::Builder::FormatFn(const Message& message,
                                      std::index_sequence<N...> /*indices*/)
    -> std::string {
  static_assert(sizeof...(Args) == sizeof...(N), "Invalid template args");
  CARBON_CHECK(message.format_args.size() == sizeof...(Args),
               "Argument count mismatch on {0}: {1} != {2}", message.kind,
               message.format_args.size(), sizeof...(Args));
  return llvm::formatv(
      message.format.data(),
      llvm::any_cast<
          typename Internal::DiagnosticTypeForArg<Args>::StorageType>(
          message.format_args[N])...);
}

template <typename LocT>
template <typename... Args>
auto Emitter<LocT>::Emit(LocT loc,
                         const DiagnosticBase<Args...>& diagnostic_base,
                         Internal::NoTypeDeduction<Args>... args) -> void {
  Builder builder(this, loc, diagnostic_base, {MakeAny<Args>(args)...});
  builder.Emit();
}

template <typename LocT>
template <typename... Args>
auto Emitter<LocT>::Build(LocT loc,
                          const DiagnosticBase<Args...>& diagnostic_base,
                          Internal::NoTypeDeduction<Args>... args) -> Builder {
  return Builder(this, loc, diagnostic_base, {MakeAny<Args>(args)...});
}

template <typename LocT>
template <typename Arg>
auto Emitter<LocT>::MakeAny(Arg arg) -> llvm::Any {
  llvm::Any converted = ConvertArg(arg);
  using Storage = Internal::DiagnosticTypeForArg<Arg>::StorageType;
  CARBON_CHECK(llvm::any_cast<Storage>(&converted),
               "Failed to convert argument of type {0} to its storage type {1}",
               typeid(Arg).name(), typeid(Storage).name());
  return converted;
}

}  // namespace Carbon::Diagnostics

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_EMITTER_H_
