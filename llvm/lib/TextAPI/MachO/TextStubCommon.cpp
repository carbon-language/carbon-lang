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

void ScalarEnumerationTraits<ObjCConstraintType>::enumeration(
    IO &IO, ObjCConstraintType &Constraint) {
  IO.enumCase(Constraint, "none", ObjCConstraintType::None);
  IO.enumCase(Constraint, "retain_release", ObjCConstraintType::Retain_Release);
  IO.enumCase(Constraint, "retain_release_for_simulator",
              ObjCConstraintType::Retain_Release_For_Simulator);
  IO.enumCase(Constraint, "retain_release_or_gc",
              ObjCConstraintType::Retain_Release_Or_GC);
  IO.enumCase(Constraint, "gc", ObjCConstraintType::GC);
}

void ScalarTraits<PlatformType>::output(const PlatformType &Value, void *,
                                        raw_ostream &OS) {
  switch (Value) {
  case PLATFORM_MACOS:
    OS << "macosx";
    break;
  case PLATFORM_IOS:
    OS << "ios";
    break;
  case PLATFORM_WATCHOS:
    OS << "watchos";
    break;
  case PLATFORM_TVOS:
    OS << "tvos";
    break;
  case PLATFORM_BRIDGEOS:
    OS << "bridgeos";
    break;
  }
}
StringRef ScalarTraits<PlatformType>::input(StringRef Scalar, void *,
                                            PlatformType &Value) {
  int Result = StringSwitch<unsigned>(Scalar)
                   .Case("macosx", PLATFORM_MACOS)
                   .Case("ios", PLATFORM_IOS)
                   .Case("watchos", PLATFORM_WATCHOS)
                   .Case("tvos", PLATFORM_TVOS)
                   .Case("bridgeos", PLATFORM_BRIDGEOS)
                   .Default(0);

  if (!Result)
    return "unknown platform";

  Value = static_cast<PlatformType>(Result);
  return {};
}
QuotingType ScalarTraits<PlatformType>::mustQuote(StringRef) {
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
