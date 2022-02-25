//===- TestPassManager.cpp - Test pass manager functionality --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace {
struct TestModulePass
    : public PassWrapper<TestModulePass, OperationPass<ModuleOp>> {
  void runOnOperation() final {}
  StringRef getArgument() const final { return "test-module-pass"; }
  StringRef getDescription() const final {
    return "Test a module pass in the pass manager";
  }
};
struct TestFunctionPass
    : public PassWrapper<TestFunctionPass, OperationPass<FuncOp>> {
  void runOnOperation() final {}
  StringRef getArgument() const final { return "test-function-pass"; }
  StringRef getDescription() const final {
    return "Test a function pass in the pass manager";
  }
};
class TestOptionsPass
    : public PassWrapper<TestOptionsPass, OperationPass<FuncOp>> {
public:
  struct Options : public PassPipelineOptions<Options> {
    ListOption<int> listOption{*this, "list",
                               llvm::cl::MiscFlags::CommaSeparated,
                               llvm::cl::desc("Example list option")};
    ListOption<std::string> stringListOption{
        *this, "string-list", llvm::cl::MiscFlags::CommaSeparated,
        llvm::cl::desc("Example string list option")};
    Option<std::string> stringOption{*this, "string",
                                     llvm::cl::desc("Example string option")};
  };
  TestOptionsPass() = default;
  TestOptionsPass(const TestOptionsPass &) {}
  TestOptionsPass(const Options &options) {
    listOption = options.listOption;
    stringOption = options.stringOption;
    stringListOption = options.stringListOption;
  }

  void runOnOperation() final {}
  StringRef getArgument() const final { return "test-options-pass"; }
  StringRef getDescription() const final {
    return "Test options parsing capabilities";
  }

  ListOption<int> listOption{*this, "list", llvm::cl::MiscFlags::CommaSeparated,
                             llvm::cl::desc("Example list option")};
  ListOption<std::string> stringListOption{
      *this, "string-list", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc("Example string list option")};
  Option<std::string> stringOption{*this, "string",
                                   llvm::cl::desc("Example string option")};
};

/// A test pass that always aborts to enable testing the crash recovery
/// mechanism of the pass manager.
class TestCrashRecoveryPass
    : public PassWrapper<TestCrashRecoveryPass, OperationPass<>> {
  void runOnOperation() final { abort(); }
  StringRef getArgument() const final { return "test-pass-crash"; }
  StringRef getDescription() const final {
    return "Test a pass in the pass manager that always crashes";
  }
};

/// A test pass that always fails to enable testing the failure recovery
/// mechanisms of the pass manager.
class TestFailurePass : public PassWrapper<TestFailurePass, OperationPass<>> {
  void runOnOperation() final { signalPassFailure(); }
  StringRef getArgument() const final { return "test-pass-failure"; }
  StringRef getDescription() const final {
    return "Test a pass in the pass manager that always fails";
  }
};

/// A test pass that contains a statistic.
struct TestStatisticPass
    : public PassWrapper<TestStatisticPass, OperationPass<>> {
  TestStatisticPass() = default;
  TestStatisticPass(const TestStatisticPass &) {}
  StringRef getArgument() const final { return "test-stats-pass"; }
  StringRef getDescription() const final { return "Test pass statistics"; }

  Statistic opCount{this, "num-ops", "Number of operations counted"};

  void runOnOperation() final {
    getOperation()->walk([&](Operation *) { ++opCount; });
  }
};
} // namespace

static void testNestedPipeline(OpPassManager &pm) {
  // Nest a module pipeline that contains:
  /// A module pass.
  auto &modulePM = pm.nest<ModuleOp>();
  modulePM.addPass(std::make_unique<TestModulePass>());
  /// A nested function pass.
  auto &nestedFunctionPM = modulePM.nest<FuncOp>();
  nestedFunctionPM.addPass(std::make_unique<TestFunctionPass>());

  // Nest a function pipeline that contains a single pass.
  auto &functionPM = pm.nest<FuncOp>();
  functionPM.addPass(std::make_unique<TestFunctionPass>());
}

static void testNestedPipelineTextual(OpPassManager &pm) {
  (void)parsePassPipeline("test-pm-nested-pipeline", pm);
}

namespace mlir {
void registerPassManagerTestPass() {
  PassRegistration<TestOptionsPass>();

  PassRegistration<TestModulePass>();

  PassRegistration<TestFunctionPass>();

  PassRegistration<TestCrashRecoveryPass>();
  PassRegistration<TestFailurePass>();

  PassRegistration<TestStatisticPass>();

  PassPipelineRegistration<>("test-pm-nested-pipeline",
                             "Test a nested pipeline in the pass manager",
                             testNestedPipeline);
  PassPipelineRegistration<>("test-textual-pm-nested-pipeline",
                             "Test a nested pipeline in the pass manager",
                             testNestedPipelineTextual);
  PassPipelineRegistration<>(
      "test-dump-pipeline",
      "Dumps the pipeline build so far for debugging purposes",
      [](OpPassManager &pm) {
        pm.printAsTextualPipeline(llvm::errs());
        llvm::errs() << "\n";
      });

  PassPipelineRegistration<TestOptionsPass::Options>
      registerOptionsPassPipeline(
          "test-options-pass-pipeline",
          "Parses options using pass pipeline registration",
          [](OpPassManager &pm, const TestOptionsPass::Options &options) {
            pm.addPass(std::make_unique<TestOptionsPass>(options));
          });
}
} // namespace mlir
