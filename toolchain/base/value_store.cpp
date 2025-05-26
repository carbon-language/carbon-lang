// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/value_store.h"

#include "common/ostream.h"

#ifdef LLVM_ADDRESS_SANITIZER_BUILD

namespace Carbon {

static bool g_poison_verbose = false;
static llvm::StringRef g_poison_stop_condition;

auto SetPoisonVerbose(bool v) -> void { g_poison_verbose = v; }
auto SetPoisonStop(llvm::StringRef s) -> void { g_poison_stop_condition = s; }

namespace Internal {

static auto ShouldPrint() -> bool { return g_poison_verbose; }

struct Condition {
  std::string_view label;
  uint64_t counter;
};

static auto ParseStopCondition() -> Condition {
  if (g_poison_stop_condition.empty()) {
    return {.label = std::string_view(), .counter = static_cast<uint64_t>(-1)};
  }

  auto [label, counter_str] = g_poison_stop_condition.split(':');
  if (counter_str.empty()) {
    llvm::errs()
        << "ERROR: --poison_stop condition should be 'label:counter'\n";
    std::exit(1);
  }
  uint64_t counter;
  // Note: getAsInteger returns false on success @_@.
  if (counter_str.getAsInteger(0, counter)) {
    llvm::errs() << "ERROR: counter could not be parsed as an integer\n";
    llvm::errs()
        << "ERROR: --poison_stop condition should be 'label:counter'\n";
    llvm::errs() << " NOTE: found label:'" << label << "' counter:'"
                 << counter_str << "'\n";
    std::exit(1);
  }
  return {.label = static_cast<std::string_view>(label), .counter = counter};
}

static auto ShouldStop(std::string_view label, uint64_t counter) -> bool {
  static auto condition = ParseStopCondition();
  return counter >= condition.counter && label == condition.label;
}

auto LogPoison(std::string_view label, int element) -> void {
  if (!ShouldPrint()) {
    return;
  }
  static uint64_t counter = 0;
  if (element < 0) {
    llvm::errs() << "++ " << label << " PoisonAll (" << label << ":" << counter
                 << ")\n";
  } else {
    llvm::errs() << "++ " << label << " PoisonElement " << element << " ("
                 << label << ":" << counter << ")\n";
  }
  if (ShouldStop(label, counter)) {
    llvm::errs() << "*** Stopping on poison event. Stack trace below.\n";
    std::abort();
  }
  ++counter;
}

auto LogUnpoison(std::string_view label, int element) -> void {
  if (!ShouldPrint()) {
    return;
  }
  if (element < 0) {
    llvm::errs() << "-- " << label << " UnpoisonAll\n";
  } else {
  }
  llvm::errs() << "-- " << label << " UnpoisonElement " << element << '\n';
}

}  // namespace Internal
}  // namespace Carbon

#endif
