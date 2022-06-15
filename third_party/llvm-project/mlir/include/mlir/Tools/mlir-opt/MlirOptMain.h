//===- MlirOptMain.h - MLIR Optimizer Driver main ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIROPT_MLIROPTMAIN_H
#define MLIR_TOOLS_MLIROPT_MLIROPTMAIN_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <cstdlib>
#include <memory>

namespace llvm {
class raw_ostream;
class MemoryBuffer;
} // namespace llvm

namespace mlir {
class DialectRegistry;
class PassPipelineCLParser;
class PassManager;

/// This defines the function type used to setup the pass manager. This can be
/// used to pass in a callback to setup a default pass pipeline to be applied on
/// the loaded IR.
using PassPipelineFn = llvm::function_ref<LogicalResult(PassManager &pm)>;

/// Perform the core processing behind `mlir-opt`:
/// - outputStream is the stream where the resulting IR is printed.
/// - buffer is the in-memory file to parser and process.
/// - passPipeline is the specification of the pipeline that will be applied.
/// - registry should contain all the dialects that can be parsed in the source.
/// - splitInputFile will look for a "-----" marker in the input file, and load
/// each chunk in an individual ModuleOp processed separately.
/// - verifyDiagnostics enables a verification mode where comments starting with
/// "expected-(error|note|remark|warning)" are parsed in the input and matched
/// against emitted diagnostics.
/// - verifyPasses enables the IR verifier in-between each pass in the pipeline.
/// - allowUnregisteredDialects allows to parse and create operation without
/// registering the Dialect in the MLIRContext.
/// - preloadDialectsInContext will trigger the upfront loading of all
///   dialects from the global registry in the MLIRContext. This option is
///   deprecated and will be removed soon.
LogicalResult MlirOptMain(llvm::raw_ostream &outputStream,
                          std::unique_ptr<llvm::MemoryBuffer> buffer,
                          const PassPipelineCLParser &passPipeline,
                          DialectRegistry &registry, bool splitInputFile,
                          bool verifyDiagnostics, bool verifyPasses,
                          bool allowUnregisteredDialects,
                          bool preloadDialectsInContext = false);

/// Support a callback to setup the pass manager.
/// - passManagerSetupFn is the callback invoked to setup the pass manager to
///   apply on the loaded IR.
LogicalResult MlirOptMain(llvm::raw_ostream &outputStream,
                          std::unique_ptr<llvm::MemoryBuffer> buffer,
                          PassPipelineFn passManagerSetupFn,
                          DialectRegistry &registry, bool splitInputFile,
                          bool verifyDiagnostics, bool verifyPasses,
                          bool allowUnregisteredDialects,
                          bool preloadDialectsInContext = false);

/// Implementation for tools like `mlir-opt`.
/// - toolName is used for the header displayed by `--help`.
/// - registry should contain all the dialects that can be parsed in the source.
/// - preloadDialectsInContext will trigger the upfront loading of all
///   dialects from the global registry in the MLIRContext. This option is
///   deprecated and will be removed soon.
LogicalResult MlirOptMain(int argc, char **argv, llvm::StringRef toolName,
                          DialectRegistry &registry,
                          bool preloadDialectsInContext = false);

/// Helper wrapper to return the result of MlirOptMain directly from main.
///
/// Example:
///
///     int main(int argc, char **argv) {
///       // ...
///       return mlir::asMainReturnCode(mlir::MlirOptMain(
///           argc, argv, /* ... */);
///     }
///
inline int asMainReturnCode(LogicalResult r) {
  return r.succeeded() ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace mlir

#endif // MLIR_TOOLS_MLIROPT_MLIROPTMAIN_H
