//===- ExecutionEngine.cpp - C API for MLIR JIT ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/ExecutionEngine.h"
#include "mlir/CAPI/ExecutionEngine.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/Support/TargetSelect.h"

using namespace mlir;

extern "C" MlirExecutionEngine
mlirExecutionEngineCreate(MlirModule op, int optLevel, int numPaths,
                          const MlirStringRef *sharedLibPaths) {
  static bool initOnce = [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmParser(); // needed for inline_asm
    llvm::InitializeNativeTargetAsmPrinter();
    return true;
  }();
  (void)initOnce;

  mlir::registerLLVMDialectTranslation(*unwrap(op)->getContext());

  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Failed to create a JITTargetMachineBuilder for the host\n";
    return MlirExecutionEngine{nullptr};
  }
  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Failed to create a TargetMachine for the host\n";
    return MlirExecutionEngine{nullptr};
  }

  SmallVector<StringRef> libPaths;
  for (unsigned i = 0; i < static_cast<unsigned>(numPaths); ++i)
    libPaths.push_back(sharedLibPaths[i].data);

  // Create a transformer to run all LLVM optimization passes at the
  // specified optimization level.
  auto llvmOptLevel = static_cast<llvm::CodeGenOpt::Level>(optLevel);
  auto transformer = mlir::makeOptimizingTransformer(
      llvmOptLevel, /*sizeLevel=*/0, /*targetMachine=*/tmOrError->get());
  ExecutionEngineOptions jitOptions;
  jitOptions.transformer = transformer;
  jitOptions.jitCodeGenOptLevel = llvmOptLevel;
  jitOptions.sharedLibPaths = libPaths;
  auto jitOrError = ExecutionEngine::create(unwrap(op), jitOptions);
  if (!jitOrError) {
    consumeError(jitOrError.takeError());
    return MlirExecutionEngine{nullptr};
  }
  return wrap(jitOrError->release());
}

extern "C" void mlirExecutionEngineDestroy(MlirExecutionEngine jit) {
  delete (unwrap(jit));
}

extern "C" MlirLogicalResult
mlirExecutionEngineInvokePacked(MlirExecutionEngine jit, MlirStringRef name,
                                void **arguments) {
  const std::string ifaceName = ("_mlir_ciface_" + unwrap(name)).str();
  llvm::Error error = unwrap(jit)->invokePacked(
      ifaceName, MutableArrayRef<void *>{arguments, (size_t)0});
  if (error)
    return wrap(failure());
  return wrap(success());
}

extern "C" void *mlirExecutionEngineLookupPacked(MlirExecutionEngine jit,
                                                 MlirStringRef name) {
  auto expectedFPtr = unwrap(jit)->lookupPacked(unwrap(name));
  if (!expectedFPtr)
    return nullptr;
  return reinterpret_cast<void *>(*expectedFPtr);
}

extern "C" void *mlirExecutionEngineLookup(MlirExecutionEngine jit,
                                           MlirStringRef name) {
  auto expectedFPtr = unwrap(jit)->lookup(unwrap(name));
  if (!expectedFPtr)
    return nullptr;
  return reinterpret_cast<void *>(*expectedFPtr);
}

extern "C" void mlirExecutionEngineRegisterSymbol(MlirExecutionEngine jit,
                                                  MlirStringRef name,
                                                  void *sym) {
  unwrap(jit)->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbolMap;
    symbolMap[interner(unwrap(name))] =
        llvm::JITEvaluatedSymbol::fromPointer(sym);
    return symbolMap;
  });
}

extern "C" void mlirExecutionEngineDumpToObjectFile(MlirExecutionEngine jit,
                                                    MlirStringRef name) {
  unwrap(jit)->dumpToObjectFile(unwrap(name));
}
