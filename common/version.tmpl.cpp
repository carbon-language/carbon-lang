// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/version.h"

#include <string_view>

namespace Carbon {

// A simplistic string-to-integer routine that is consteval for compile-time
// extracting specific components of the version from the string form. We use
// `std::string_view` for its broader `constexpr` API.
static consteval auto ToInt(std::string_view str) -> int {
  int result = 0;
  while (true) {
    result += str.front() - '0';
    str.remove_prefix(1);
    if (str.empty()) {
      break;
    }
    result *= 10;
  }
  return result;
}

static consteval auto MajorVersion(std::string_view str) -> int {
  return ToInt(str.substr(0, str.find('.')));
}

static consteval auto MinorVersion(std::string_view str) -> int {
  str.remove_prefix(str.find('.') + 1);
  return ToInt(str.substr(0, str.find('.')));
}

static consteval auto PatchVersion(std::string_view str) -> int {
  str.remove_prefix(str.find('.') + 1);
  str.remove_prefix(str.find('.') + 1);
  // Note that searching for `-` may find the end of the string if there is no
  // pre-release component, but that produces the correct result here.
  return ToInt(str.substr(0, str.find('-')));
}

// The major, minor, and patch versions are always provided and stable. They
// don't depend on build stamping or introduce caching issues. Provide normal
// strong definitions.
constexpr int Version::Major = MajorVersion("$VERSION");
constexpr int Version::Minor = MinorVersion("$VERSION");
constexpr int Version::Patch = PatchVersion("$VERSION");

}  // namespace Carbon
