// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/value_store.h"

#include "absl/flags/flag.h"
#include "common/ostream.h"

#ifdef LLVM_ADDRESS_SANITIZER_BUILD

ABSL_FLAG(bool, poison_verbose, false,
          "Print out ASAN poison events in the ValueStore.");
ABSL_FLAG(
    std::string, poison_stop, "",
    "Set a condition after which we should stop on the next poison. The "
    "format is <label>:<counter>. Once the label is poisoned with a "
    "counter value >= <counter>, the program will abort and print the "
    "stack trace of the poisoning. Note that the poison counter can vary "
    "slightly from run to run, so you can use a smaller number to catch the "
    "poison event that you want. The counter values are printed when "
    "--poison_verbose is used.");

namespace Carbon::Internal {

static auto ShouldPrint() -> bool {
  static bool print = absl::GetFlag(FLAGS_poison_verbose);
  return print;
}

struct Condition {
  std::string_view label;
  uint64_t counter;
};

static auto ParseStopCondition() -> Condition {
  static std::string condition_string = absl::GetFlag(FLAGS_poison_stop);
  llvm::StringRef condition = condition_string;
  if (condition.empty()) {
    return {.label = std::string_view(), .counter = static_cast<uint64_t>(-1)};
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

}  // namespace Carbon::Internal

#endif
