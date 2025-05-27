// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/value_store.h"

#include "common/ostream.h"

#ifdef LLVM_ADDRESS_SANITIZER_BUILD

namespace Carbon {

namespace Internal {

// Gets the global (thread-safe) decision to print poison logs.
//
// Initially this is called on each thread (with the same `val`) when deciding
// to print logs or not, and the first such thread to win which sets the static
// global bool. Further calls (which do not need to pass a value) just query and
// get back that `Condition`.
static auto GetShouldPrint(std::optional<bool> val = std::nullopt) -> bool {
  static auto should_print = *val;
  return should_print;
}

struct Condition {
  std::string label;
  uint64_t counter;
};

static auto ParseStopCondition(llvm::StringRef condition) -> Condition {
  if (condition.empty()) {
    return {.counter = static_cast<uint64_t>(-1)};
  }

  auto [label, counter_str] = condition.split(':');
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
  return {.label = std::string(label), .counter = counter};
}

// Gets the global (thread-safe) stop Condition.
//
// Initially this is called on each thread (with the same `condition_str`) when
// setting the stop condition, and the first such thread to win which sets the
// static global `Condition`. Further calls without a string input just query
// and get back that `Condition`.
static auto GetStopCondition(llvm::StringRef condition_str = "") -> Condition {
  static Condition condition = ParseStopCondition(condition_str);
  return condition;
}

auto LogPoison(llvm::StringRef label, int element) -> void {
  if (!GetShouldPrint()) {
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
  auto condition = GetStopCondition();
  if (counter >= condition.counter && label == condition.label) {
    llvm::errs() << "*** Stopping on poison event. Stack trace below.\n";
    std::abort();
  }
  ++counter;
}

auto LogUnpoison(llvm::StringRef label, int element) -> void {
  if (!GetShouldPrint()) {
    return;
  }
  if (element < 0) {
    llvm::errs() << "-- " << label << " UnpoisonAll\n";
  } else {
  }
  llvm::errs() << "-- " << label << " UnpoisonElement " << element << '\n';
}

}  // namespace Internal

auto SetPoisonVerbose(bool v) -> void { Internal::GetShouldPrint(v); }
auto SetPoisonStop(llvm::StringRef s) -> void { Internal::GetStopCondition(s); }

}  // namespace Carbon

#endif
