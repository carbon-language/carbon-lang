// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_EMITTER_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_EMITTER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "common/check.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/TypeName.h"
#include "toolchain/diagnostics/consumer.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/diagnostics/kind.h"

namespace Carbon::Diagnostics {

namespace Internal {

// Disable type deduction based on `args`; the type of `diagnostic_base`
// determines the diagnostic's parameter types.
template <typename Arg>
using NoTypeDeduction = std::type_identity_t<Arg>;

}  // namespace Internal

// The result of `Emitter::ConvertLoc`. This is non-templated to allow
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

    // Attaches a range of source to the diagnostic, along with the phrase
    // `label` declares saying what that range has to do with it; see
    // `CARBON_DIAGNOSTIC_LABEL`. The API mirrors the main emission API:
    // `Emitter::Emit`. For the expected usage see the builder API:
    // `Emitter::Build`.
    template <typename... Args>
    auto Attach(LocT loc, const LabelBase<Args...>& label,
                Internal::NoTypeDeduction<Args>... args) -> Builder&;

    // Adds a range the problem covers beyond the message's location: the
    // operand of an `&` that has no address to take, or the whole string
    // literal that is missing its terminator. See `Message::ranges`.
    auto ExtraRange(LocT loc) -> Builder&;

    // Emits the built diagnostic and its attached labels.
    // For the expected usage see the builder API: `Emitter::Build`.
    template <typename... Args>
    auto Emit() & -> void;

    // Prevent trivial uses of the builder; always `static_assert`s.
    template <typename... Args>
    auto Emit() && -> void;

   private:
    friend class Emitter<LocT>;

    template <typename... Args>
    explicit Builder(Emitter<LocT>* emitter, LocT loc,
                     const DiagnosticBase<Args...>& diagnostic_base,
                     llvm::SmallVector<llvm::Any> args);

    // Converts `loc`, collecting the path by which it was reached into
    // `location_info` and recording the offset the diagnostic sorts by.
    auto ConvertLoc(LocT loc, llvm::SmallVector<LocationInfo, 0>& location_info)
        -> Loc;

    // Attaches a label, handling conversion of the location and arguments.
    template <typename... Args>
    auto AttachLabel(LocT loc, const LabelBase<Args...>& label,
                     llvm::SmallVector<llvm::Any> args) -> void;

    // Returns a function that formats `Args` from the storage types they
    // convert to.
    //
    // TODO: Custom formatting can be provided with an format_provider, but that
    // affects all formatv calls. Consider replacing formatv with a custom call
    // that allows diagnostic-specific formatting.
    template <typename... Args>
    static auto MakeFormatFn() -> Diagnostics::FormatFn;

    // Applies `format` to `args`, which must be the storage types for `Args`.
    // Both a message and a label format this way, from argument lists of their
    // own.
    template <typename... Args, size_t... N>
    static auto FormatArgs(llvm::StringLiteral format,
                           llvm::ArrayRef<llvm::Any> args,
                           std::index_sequence<N...> /*indices*/)
        -> std::string;

    Emitter<LocT>* emitter_;

    // The diagnostic's parts, assembled by `Emit`. A `Message` can't be
    // default-constructed -- a `Kind` has no value meaning "no kind" -- so the
    // diagnostic itself only exists once there is a message to put in it.
    Level level_;
    bool is_on_scope_;
    int32_t last_byte_offset_ = -1;
    // Absent only while the constructor is converting the message's location.
    std::optional<Message> message_;
    llvm::SmallVector<Label, 0> labels_;
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

  // A fluent interface for building a diagnostic and attaching the source that
  // explains it. For example:
  //
  //   emitter_.Build(loc1, MyDiagnostic)
  //     .Attach(loc2, MyDiagnosticLabel)
  //     .Emit();
  template <typename... Args>
  auto Build(LocT loc, const DiagnosticBase<Args...>& diagnostic_base,
             Internal::NoTypeDeduction<Args>... args) -> Builder;

  // Adds a flush function to flush pending diagnostics that might be enqueued
  // and not yet emitted. The flush function will be called whenever `Flush` is
  // called.
  //
  // No mechanism is provided to unregister a flush function, so the function
  // must ensure that it remains callable until the emitter is destroyed.
  //
  // This is used to register a handler to flush diagnostics from Clang.
  auto AddFlushFn(std::function<auto()->void> flush_fn) -> void {
    flush_fns_.push_back(std::move(flush_fn));
  }

  // Flush all pending diagnostics that are queued externally, such as Clang
  // diagnostics. This should not be called when the external source might be in
  // the middle of producing a diagnostic, such as between Clang producing an
  // error and producing the attached notes.
  //
  // This is called automatically before a `ContextScope` is entered or left,
  // so that a pending diagnostic gets the labels of the scopes that were active
  // when it was produced. Destruction does not flush; see the `Flush` test.
  auto Flush() -> void {
    for (auto& flush_fn : flush_fns_) {
      flush_fn();
    }
  }

 protected:
  // Callback type used to report the path a location was reached by from
  // ConvertLoc. Note that the first parameter type is Loc rather than
  // LocT, because ConvertLoc must not recurse.
  using ContextFnT =
      llvm::function_ref<auto(Loc, const LocationInfoBase&)->void>;

  // Converts a LocT to a Loc and its `last_byte_offset` (see
  // `Diagnostic::last_byte_offset`). Calls `context_fn` for each step in the
  // path by which the location was reached.
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

  template <typename OtherLocT, typename ContextFn>
  friend class ContextScope;

  Consumer* consumer_;
  llvm::SmallVector<std::function<auto()->void>, 1> flush_fns_;
  llvm::SmallVector<llvm::function_ref<auto(Builder& builder)->void>>
      context_fns_;
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

// An RAII object that denotes a scope in which every diagnostic emitted gets a
// label naming the operation underway.
//
// This object is given a function `context` that will be called with a
// `Builder& builder` for any diagnostic that is emitted through the given
// emitter, and that function attaches the label by calling `builder.Attach`.
// The message is left to the emitting code. A caller that can state the
// problem better than the code detecting it passes a diagnoser instead; see
// `/toolchain/docs/diagnostics.md`.
template <typename LocT, typename ContextFn>
class ContextScope {
 public:
  ContextScope(Emitter<LocT>* emitter, ContextFn context)
    requires requires(ContextFn context, Emitter<LocT>::Builder& builder) {
      { context(builder) } -> std::same_as<void>;
    }
      : emitter_(emitter), context_(std::move(context)) {
    emitter_->Flush();
    emitter_->context_fns_.push_back(context_);
  }
  ~ContextScope() {
    emitter_->Flush();
    emitter_->context_fns_.pop_back();
  }

 private:
  Emitter<LocT>* emitter_;
  // Make a copy of the context function to ensure that it lives long enough.
  ContextFn context_;
};

template <typename LocT, typename ContextFn>
ContextScope(Emitter<LocT>* emitter, ContextFn context)
    -> ContextScope<LocT, ContextFn>;

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
auto Emitter<LocT>::Builder::Attach(LocT loc, const LabelBase<Args...>& label,
                                    Internal::NoTypeDeduction<Args>... args)
    -> Builder& {
  AttachLabel(LocT(loc), label, {emitter_->MakeAny<Args>(args)...});
  return *this;
}

template <typename LocT>
auto Emitter<LocT>::Builder::ExtraRange(LocT loc) -> Builder& {
  CARBON_CHECK(message_,
               "A builder has its message from the moment it exists.");
  // A range has no `location_info` of its own, so this one is dropped.
  llvm::SmallVector<LocationInfo, 0> location_info;
  message_->ranges.push_back(ConvertLoc(LocT(loc), location_info));
  return *this;
}

template <typename LocT>
template <typename... Args>
auto Emitter<LocT>::Builder::Emit() & -> void {
  // Scope labels follow the ones the emitting code attached, innermost scope
  // first, so that the most specific operation is listed first.
  for (auto context_fn : llvm::reverse(emitter_->context_fns_)) {
    context_fn(*this);
  }
  CARBON_CHECK(message_,
               "A builder has its message from the moment it exists.");
  emitter_->consumer_->HandleDiagnostic(
      Diagnostic{.level = level_,
                 .is_on_scope = is_on_scope_,
                 .last_byte_offset = last_byte_offset_,
                 .message = *std::move(message_),
                 .labels = std::move(labels_)});
}

template <typename LocT>
template <typename... Args>
auto Emitter<LocT>::Builder::Emit() && -> void {
  static_assert(false,
                "Use `emitter.Emit(...)` or "
                "`emitter.Build(...).Attach(...).Emit(...)` "
                "instead of `emitter.Build(...).Emit(...)`");
}

template <typename LocT>
template <typename... Args>
Emitter<LocT>::Builder::Builder(Emitter<LocT>* emitter, LocT loc,
                                const DiagnosticBase<Args...>& diagnostic_base,
                                llvm::SmallVector<llvm::Any> args)
    : emitter_(emitter),
      level_(diagnostic_base.Level),
      is_on_scope_(diagnostic_base.IsOnScope) {
  llvm::SmallVector<LocationInfo, 0> location_info;
  Loc loc_value = ConvertLoc(LocT(loc), location_info);
  message_ = Message{.kind = diagnostic_base.Kind,
                     .level = diagnostic_base.Level,
                     .loc = std::move(loc_value),
                     .location_info = std::move(location_info),
                     .format = diagnostic_base.Format,
                     .format_args = std::move(args),
                     .format_fn = MakeFormatFn<Args...>()};
}

template <typename LocT>
auto Emitter<LocT>::Builder::ConvertLoc(
    LocT loc, llvm::SmallVector<LocationInfo, 0>& location_info) -> Loc {
  ConvertedLoc converted = emitter_->ConvertLoc(
      loc, [&](Loc step_loc, const LocationInfoBase& step) {
        location_info.push_back(LocationInfo{
            .loc = step_loc, .name = step.Name, .format = step.Format});
      });
  // The diagnostic sorts by its message's location.
  //
  // TODO: A message location reached through an import leaves the offset at
  // -1, which decides the order diagnostics are printed in. Decide whether that
  // is intended; changing it needs its own testdata update.
  if (!message_ && location_info.empty()) {
    last_byte_offset_ = converted.last_byte_offset;
  }
  return converted.loc;
}

template <typename LocT>
template <typename... Args>
auto Emitter<LocT>::Builder::AttachLabel(LocT loc,
                                         const LabelBase<Args...>& label,
                                         llvm::SmallVector<llvm::Any> args)
    -> void {
  llvm::SmallVector<LocationInfo, 0> location_info;
  Loc loc_value = ConvertLoc(LocT(loc), location_info);
  labels_.push_back(Label{.loc = std::move(loc_value),
                          .location_info = std::move(location_info),
                          .name = label.Name,
                          .format = label.Format,
                          .format_args = std::move(args),
                          .format_fn = MakeFormatFn<Args...>()});
}

template <typename LocT>
template <typename... Args>
auto Emitter<LocT>::Builder::MakeFormatFn() -> Diagnostics::FormatFn {
  return [](llvm::StringLiteral format,
            llvm::ArrayRef<llvm::Any> args) -> std::string {
    CARBON_CHECK(args.size() == sizeof...(Args),
                 "Argument count mismatch on `{0}`: {1} != {2}", format,
                 args.size(), sizeof...(Args));
    return FormatArgs<Args...>(format, args,
                               std::make_index_sequence<sizeof...(Args)>());
  };
}

template <typename LocT>
template <typename... Args, size_t... N>
auto Emitter<LocT>::Builder::FormatArgs(llvm::StringLiteral format,
                                        llvm::ArrayRef<llvm::Any> args,
                                        std::index_sequence<N...> /*indices*/)
    -> std::string {
  return llvm::formatv(
      // `format` is a StringLiteral which is always NUL terminated, so data()
      // is okay here. Too bad there's no c_str() method available.
      // NOLINTNEXTLINE(bugprone-suspicious-stringview-data-usage)
      format.data(),
      llvm::any_cast<
          typename Internal::DiagnosticTypeForArg<Args>::StorageType>(
          args[N])...);
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
               llvm::getTypeName<Arg>(), llvm::getTypeName<Storage>());
  return converted;
}

}  // namespace Carbon::Diagnostics

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_EMITTER_H_
