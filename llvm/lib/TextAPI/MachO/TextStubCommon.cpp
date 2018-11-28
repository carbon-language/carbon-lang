//===- lib/TextAPI/TextStubCommon.cpp - Text Stub Common --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Implememts common Text Stub YAML mappings.
///
//===----------------------------------------------------------------------===//

#include "TextStubCommon.h"
#include "TextAPIContext.h"

using namespace llvm::MachO;

namespace llvm {
namespace yaml {

void ScalarTraits<FlowStringRef>::output(const FlowStringRef &Value, void *Ctx,
                                         raw_ostream &OS) {
  ScalarTraits<StringRef>::output(Value, Ctx, OS);
}
StringRef ScalarTraits<FlowStringRef>::input(StringRef Value, void *Ctx,
                                             FlowStringRef &Out) {
  return ScalarTraits<StringRef>::input(Value, Ctx, Out.value);
}
QuotingType ScalarTraits<FlowStringRef>::mustQuote(StringRef Name) {
  return ScalarTraits<StringRef>::mustQuote(Name);
}

void ScalarEnumerationTraits<ObjCConstraint>::enumeration(
    IO &IO, ObjCConstraint &Constraint) {
  IO.enumCase(Constraint, "none", ObjCConstraint::None);
  IO.enumCase(Constraint, "retain_release", ObjCConstraint::Retain_Release);
  IO.enumCase(Constraint, "retain_release_for_simulator",
              ObjCConstraint::Retain_Release_For_Simulator);
  IO.enumCase(Constraint, "retain_release_or_gc",
              ObjCConstraint::Retain_Release_Or_GC);
  IO.enumCase(Constraint, "gc", ObjCConstraint::GC);
}

void ScalarTraits<Platform>::output(const Platform &Value, void *,
                                    raw_ostream &OS) {
  switch (Value) {
  default:
    llvm_unreachable("unexpected platform");
    break;
  case Platform::macOS:
    OS << "macosx";
    break;
  case Platform::iOS:
    OS << "ios";
    break;
  case Platform::watchOS:
    OS << "watchos";
    break;
  case Platform::tvOS:
    OS << "tvos";
    break;
  case Platform::bridgeOS:
    OS << "bridgeos";
    break;
  }
}
StringRef ScalarTraits<Platform>::input(StringRef Scalar, void *,
                                        Platform &Value) {
  Value = StringSwitch<Platform>(Scalar)
              .Case("macosx", Platform::macOS)
              .Case("ios", Platform::iOS)
              .Case("watchos", Platform::watchOS)
              .Case("tvos", Platform::tvOS)
              .Case("bridgeos", Platform::bridgeOS)
              .Default(Platform::unknown);

  if (Value == Platform::unknown)
    return "unknown platform";
  return {};
}
QuotingType ScalarTraits<Platform>::mustQuote(StringRef) {
  return QuotingType::None;
}

void ScalarBitSetTraits<ArchitectureSet>::bitset(IO &IO,
                                                 ArchitectureSet &Archs) {
#define ARCHINFO(arch, type, subtype)                                          \
  IO.bitSetCase(Archs, #arch, 1U << static_cast<int>(Architecture::arch));
#include "llvm/TextAPI/MachO/Architecture.def"
#undef ARCHINFO
}

void ScalarTraits<Architecture>::output(const Architecture &Value, void *,
                                        raw_ostream &OS) {
  OS << Value;
}
StringRef ScalarTraits<Architecture>::input(StringRef Scalar, void *,
                                            Architecture &Value) {
  Value = getArchitectureFromName(Scalar);
  return {};
}
QuotingType ScalarTraits<Architecture>::mustQuote(StringRef) {
  return QuotingType::None;
}

void ScalarTraits<PackedVersion>::output(const PackedVersion &Value, void *,
                                         raw_ostream &OS) {
  OS << Value;
}
StringRef ScalarTraits<PackedVersion>::input(StringRef Scalar, void *,
                                             PackedVersion &Value) {
  if (!Value.parse32(Scalar))
    return "invalid packed version string.";
  return {};
}
QuotingType ScalarTraits<PackedVersion>::mustQuote(StringRef) {
  return QuotingType::None;
}

void ScalarTraits<SwiftVersion>::output(const SwiftVersion &Value, void *,
                                        raw_ostream &OS) {
  switch (Value) {
  case 1:
    OS << "1.0";
    break;
  case 2:
    OS << "1.1";
    break;
  case 3:
    OS << "2.0";
    break;
  case 4:
    OS << "3.0";
    break;
  default:
    OS << (unsigned)Value;
    break;
  }
}
StringRef ScalarTraits<SwiftVersion>::input(StringRef Scalar, void *,
                                            SwiftVersion &Value) {
  Value = StringSwitch<SwiftVersion>(Scalar)
              .Case("1.0", 1)
              .Case("1.1", 2)
              .Case("2.0", 3)
              .Case("3.0", 4)
              .Default(0);
  if (Value != SwiftVersion(0))
    return {};

  if (Scalar.getAsInteger(10, Value))
    return "invalid Swift ABI version.";

  return StringRef();
}
QuotingType ScalarTraits<SwiftVersion>::mustQuote(StringRef) {
  return QuotingType::None;
}

void ScalarTraits<UUID>::output(const UUID &Value, void *, raw_ostream &OS) {
  OS << Value.first << ": " << Value.second;
}
StringRef ScalarTraits<UUID>::input(StringRef Scalar, void *, UUID &Value) {
  auto Split = Scalar.split(':');
  auto Arch = Split.first.trim();
  auto UUID = Split.second.trim();
  if (UUID.empty())
    return "invalid uuid string pair";
  Value.first = getArchitectureFromName(Arch);
  Value.second = UUID;
  return {};
}
QuotingType ScalarTraits<UUID>::mustQuote(StringRef) {
  return QuotingType::Single;
}

} // end namespace yaml.
} // end namespace llvm.
