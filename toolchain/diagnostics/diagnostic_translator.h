// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_TRANSLATOR_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_TRANSLATOR_H_

#include <cstdint>

#include "common/check.h"
#include "llvm/ADT/Any.h"
#include "toolchain/diagnostics/diagnostic.h"

namespace Carbon {

// Known diagnostic type translations. These are enumerated because `llvm::Any`
// doesn't expose the contained type; instead, we infer it from a given
// diagnostic.
enum class DiagnosticTypeTranslation : int8_t {
  None,
  NameId,
  TypeId,
};

// An interface that can translate some representation of a location into a
// diagnostic location.
template <typename LocationT>
class DiagnosticTranslator {
 public:
  virtual ~DiagnosticTranslator() = default;

  virtual auto TranslateLocation(LocationT loc) const -> DiagnosticLocation = 0;

  // Translates arg types as needed. Not all uses support translation, so the
  // default simply errors.
  virtual auto TranslateArg(DiagnosticTypeTranslation translation,
                            llvm::Any /*arg*/) const -> llvm::Any {
    CARBON_FATAL() << "Unexpected call to TranslateArg: "
                   << static_cast<int8_t>(translation);
  }
};

// Used by types to indicate a DiagnosticTypeTranslation that results in the
// provided StorageType. For example, to translate NameId to a std::string, we
// write:
//
// struct NameId {
//   using DiagnosticType =
//       DiagnosticTypeInfo<std::string, DiagnosticTypeTranslation::NameId>;
// };
template <typename StorageTypeT, DiagnosticTypeTranslation TranslationV>
struct DiagnosticTypeInfo {
  using StorageType = StorageTypeT;
  static constexpr DiagnosticTypeTranslation Translation = TranslationV;
};

namespace Internal {

// Determines whether there's a DiagnosticType member on Arg.
// Used by DiagnosticEmitter.
template <typename Arg>
concept HasDiagnosticType =
    requires { std::type_identity<typename Arg::DiagnosticType>(); };

// The default implementation with no translation.
template <typename Arg, typename /*Unused*/ = void>
struct DiagnosticTypeForArg
    : public DiagnosticTypeInfo<Arg, DiagnosticTypeTranslation::None> {};

// Exposes a custom translation for an argument type.
template <typename Arg>
  requires HasDiagnosticType<Arg>
struct DiagnosticTypeForArg<Arg> : public Arg::DiagnosticType {};

}  // namespace Internal

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_TRANSLATOR_H_
