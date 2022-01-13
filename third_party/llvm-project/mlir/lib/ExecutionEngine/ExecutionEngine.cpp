//===- ExecutionEngine.cpp - MLIR Execution engine and utils --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the execution engine for MLIR modules based on LLVM Orc
// JIT engine.
//
//===----------------------------------------------------------------------===//
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/ToolOutputFile.h"

#define DEBUG_TYPE "execution-engine"

using namespace mlir;
using llvm::dbgs;
using llvm::Error;
using llvm::errs;
using llvm::Expected;
using llvm::LLVMContext;
using llvm::MemoryBuffer;
using llvm::MemoryBufferRef;
using llvm::Module;
using llvm::SectionMemoryManager;
using llvm::StringError;
using llvm::Triple;
using llvm::orc::DynamicLibrarySearchGenerator;
using llvm::orc::ExecutionSession;
using llvm::orc::IRCompileLayer;
using llvm::orc::JITTargetMachineBuilder;
using llvm::orc::MangleAndInterner;
using llvm::orc::RTDyldObjectLinkingLayer;
using llvm::orc::SymbolMap;
using llvm::orc::ThreadSafeModule;
using llvm::orc::TMOwningSimpleCompiler;

/// Wrap a string into an llvm::StringError.
static Error make_string_error(const Twine &message) {
  return llvm::make_error<StringError>(message.str(),
                                       llvm::inconvertibleErrorCode());
}

void SimpleObjectCache::notifyObjectCompiled(const Module *M,
                                             MemoryBufferRef ObjBuffer) {
  cachedObjects[M->getModuleIdentifier()] = MemoryBuffer::getMemBufferCopy(
      ObjBuffer.getBuffer(), ObjBuffer.getBufferIdentifier());
}

std::unique_ptr<MemoryBuffer> SimpleObjectCache::getObject(const Module *M) {
  auto I = cachedObjects.find(M->getModuleIdentifier());
  if (I == cachedObjects.end()) {
    LLVM_DEBUG(dbgs() << "No object for " << M->getModuleIdentifier()
                      << " in cache. Compiling.\n");
    return nullptr;
  }
  LLVM_DEBUG(dbgs() << "Object for " << M->getModuleIdentifier()
                    << " loaded from cache.\n");
  return MemoryBuffer::getMemBuffer(I->second->getMemBufferRef());
}

void SimpleObjectCache::dumpToObjectFile(StringRef outputFilename) {
  // Set up the output file.
  std::string errorMessage;
  auto file = openOutputFile(outputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return;
  }

  // Dump the object generated for a single module to the output file.
  assert(cachedObjects.size() == 1 && "Expected only one object entry.");
  auto &cachedObject = cachedObjects.begin()->second;
  file->os() << cachedObject->getBuffer();
  file->keep();
}

void ExecutionEngine::dumpToObjectFile(StringRef filename) {
  cache->dumpToObjectFile(filename);
}

void ExecutionEngine::registerSymbols(
    llvm::function_ref<SymbolMap(MangleAndInterner)> symbolMap) {
  auto &mainJitDylib = jit->getMainJITDylib();
  cantFail(mainJitDylib.define(
      absoluteSymbols(symbolMap(llvm::orc::MangleAndInterner(
          mainJitDylib.getExecutionSession(), jit->getDataLayout())))));
}

// Setup LLVM target triple from the current machine.
bool ExecutionEngine::setupTargetTriple(Module *llvmModule) {
  // Setup the machine properties from the current architecture.
  auto targetTriple = llvm::sys::getDefaultTargetTriple();
  std::string errorMessage;
  auto target = llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
  if (!target) {
    errs() << "NO target: " << errorMessage << "\n";
    return true;
  }

  std::string cpu(llvm::sys::getHostCPUName());
  llvm::SubtargetFeatures features;
  llvm::StringMap<bool> hostFeatures;

  if (llvm::sys::getHostCPUFeatures(hostFeatures))
    for (auto &f : hostFeatures)
      features.AddFeature(f.first(), f.second);

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetTriple, cpu, features.getString(), {}, {}));
  if (!machine) {
    errs() << "Unable to create target machine\n";
    return true;
  }
  llvmModule->setDataLayout(machine->createDataLayout());
  llvmModule->setTargetTriple(targetTriple);
  return false;
}

static std::string makePackedFunctionName(StringRef name) {
  return "_mlir_" + name.str();
}

// For each function in the LLVM module, define an interface function that wraps
// all the arguments of the original function and all its results into an i8**
// pointer to provide a unified invocation interface.
static void packFunctionArguments(Module *module) {
  auto &ctx = module->getContext();
  llvm::IRBuilder<> builder(ctx);
  DenseSet<llvm::Function *> interfaceFunctions;
  for (auto &func : module->getFunctionList()) {
    if (func.isDeclaration()) {
      continue;
    }
    if (interfaceFunctions.count(&func)) {
      continue;
    }

    // Given a function `foo(<...>)`, define the interface function
    // `mlir_foo(i8**)`.
    auto newType = llvm::FunctionType::get(
        builder.getVoidTy(), builder.getInt8PtrTy()->getPointerTo(),
        /*isVarArg=*/false);
    auto newName = makePackedFunctionName(func.getName());
    auto funcCst = module->getOrInsertFunction(newName, newType);
    llvm::Function *interfaceFunc = cast<llvm::Function>(funcCst.getCallee());
    interfaceFunctions.insert(interfaceFunc);

    // Extract the arguments from the type-erased argument list and cast them to
    // the proper types.
    auto bb = llvm::BasicBlock::Create(ctx);
    bb->insertInto(interfaceFunc);
    builder.SetInsertPoint(bb);
    llvm::Value *argList = interfaceFunc->arg_begin();
    SmallVector<llvm::Value *, 8> args;
    args.reserve(llvm::size(func.args()));
    for (auto &indexedArg : llvm::enumerate(func.args())) {
      llvm::Value *argIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), APInt(64, indexedArg.index()));
      llvm::Value *argPtrPtr = builder.CreateGEP(
          builder.getInt8PtrTy(), argList, argIndex);
      llvm::Value *argPtr = builder.CreateLoad(builder.getInt8PtrTy(),
                                               argPtrPtr);
      llvm::Type *argTy = indexedArg.value().getType();
      argPtr = builder.CreateBitCast(argPtr, argTy->getPointerTo());
      llvm::Value *arg = builder.CreateLoad(argTy, argPtr);
      args.push_back(arg);
    }

    // Call the implementation function with the extracted arguments.
    llvm::Value *result = builder.CreateCall(&func, args);

    // Assuming the result is one value, potentially of type `void`.
    if (!result->getType()->isVoidTy()) {
      llvm::Value *retIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), APInt(64, llvm::size(func.args())));
      llvm::Value *retPtrPtr =
          builder.CreateGEP(builder.getInt8PtrTy(), argList, retIndex);
      llvm::Value *retPtr = builder.CreateLoad(builder.getInt8PtrTy(),
                                               retPtrPtr);
      retPtr = builder.CreateBitCast(retPtr, result->getType()->getPointerTo());
      builder.CreateStore(result, retPtr);
    }

    // The interface function returns void.
    builder.CreateRetVoid();
  }
}

ExecutionEngine::ExecutionEngine(bool enableObjectCache,
                                 bool enableGDBNotificationListener,
                                 bool enablePerfNotificationListener)
    : cache(enableObjectCache ? new SimpleObjectCache() : nullptr),
      gdbListener(enableGDBNotificationListener
                      ? llvm::JITEventListener::createGDBRegistrationListener()
                      : nullptr),
      perfListener(enablePerfNotificationListener
                       ? llvm::JITEventListener::createPerfJITEventListener()
                       : nullptr) {}

Expected<std::unique_ptr<ExecutionEngine>> ExecutionEngine::create(
    ModuleOp m,
    llvm::function_ref<std::unique_ptr<llvm::Module>(ModuleOp,
                                                     llvm::LLVMContext &)>
        llvmModuleBuilder,
    llvm::function_ref<Error(llvm::Module *)> transformer,
    Optional<llvm::CodeGenOpt::Level> jitCodeGenOptLevel,
    ArrayRef<StringRef> sharedLibPaths, bool enableObjectCache,
    bool enableGDBNotificationListener, bool enablePerfNotificationListener) {
  auto engine = std::make_unique<ExecutionEngine>(
      enableObjectCache, enableGDBNotificationListener,
      enablePerfNotificationListener);

  std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext);
  auto llvmModule = llvmModuleBuilder ? llvmModuleBuilder(m, *ctx)
                                      : translateModuleToLLVMIR(m, *ctx);
  if (!llvmModule)
    return make_string_error("could not convert to LLVM IR");
  // FIXME: the triple should be passed to the translation or dialect conversion
  // instead of this.  Currently, the LLVM module created above has no triple
  // associated with it.
  setupTargetTriple(llvmModule.get());
  packFunctionArguments(llvmModule.get());

  auto dataLayout = llvmModule->getDataLayout();

  // Callback to create the object layer with symbol resolution to current
  // process and dynamically linked libraries.
  auto objectLinkingLayerCreator = [&](ExecutionSession &session,
                                       const Triple &TT) {
    auto objectLayer = std::make_unique<RTDyldObjectLinkingLayer>(
        session, []() { return std::make_unique<SectionMemoryManager>(); });

    // Register JIT event listeners if they are enabled.
    if (engine->gdbListener)
      objectLayer->registerJITEventListener(*engine->gdbListener);
    if (engine->perfListener)
      objectLayer->registerJITEventListener(*engine->perfListener);

    // COFF format binaries (Windows) need special handling to deal with
    // exported symbol visibility.
    // cf llvm/lib/ExecutionEngine/Orc/LLJIT.cpp LLJIT::createObjectLinkingLayer
    llvm::Triple targetTriple(llvm::Twine(llvmModule->getTargetTriple()));
    if (targetTriple.isOSBinFormatCOFF()) {
      objectLayer->setOverrideObjectFlagsWithResponsibilityFlags(true);
      objectLayer->setAutoClaimResponsibilityForObjectSymbols(true);
    }

    // Resolve symbols from shared libraries.
    for (auto libPath : sharedLibPaths) {
      auto mb = llvm::MemoryBuffer::getFile(libPath);
      if (!mb) {
        errs() << "Failed to create MemoryBuffer for: " << libPath
               << "\nError: " << mb.getError().message() << "\n";
        continue;
      }
      auto &JD = session.createBareJITDylib(std::string(libPath));
      auto loaded = DynamicLibrarySearchGenerator::Load(
          libPath.data(), dataLayout.getGlobalPrefix());
      if (!loaded) {
        errs() << "Could not load " << libPath << ":\n  " << loaded.takeError()
               << "\n";
        continue;
      }
      JD.addGenerator(std::move(*loaded));
      cantFail(objectLayer->add(JD, std::move(mb.get())));
    }

    return objectLayer;
  };

  // Callback to inspect the cache and recompile on demand. This follows Lang's
  // LLJITWithObjectCache example.
  auto compileFunctionCreator = [&](JITTargetMachineBuilder JTMB)
      -> Expected<std::unique_ptr<IRCompileLayer::IRCompiler>> {
    if (jitCodeGenOptLevel)
      JTMB.setCodeGenOptLevel(jitCodeGenOptLevel.getValue());
    auto TM = JTMB.createTargetMachine();
    if (!TM)
      return TM.takeError();
    return std::make_unique<TMOwningSimpleCompiler>(std::move(*TM),
                                                    engine->cache.get());
  };

  // Create the LLJIT by calling the LLJITBuilder with 2 callbacks.
  auto jit =
      cantFail(llvm::orc::LLJITBuilder()
                   .setCompileFunctionCreator(compileFunctionCreator)
                   .setObjectLinkingLayerCreator(objectLinkingLayerCreator)
                   .create());

  // Add a ThreadSafemodule to the engine and return.
  ThreadSafeModule tsm(std::move(llvmModule), std::move(ctx));
  if (transformer)
    cantFail(tsm.withModuleDo(
        [&](llvm::Module &module) { return transformer(&module); }));
  cantFail(jit->addIRModule(std::move(tsm)));
  engine->jit = std::move(jit);

  // Resolve symbols that are statically linked in the current process.
  llvm::orc::JITDylib &mainJD = engine->jit->getMainJITDylib();
  mainJD.addGenerator(
      cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
          dataLayout.getGlobalPrefix())));

  return std::move(engine);
}

Expected<void (*)(void **)> ExecutionEngine::lookup(StringRef name) const {
  auto expectedSymbol = jit->lookup(makePackedFunctionName(name));

  // JIT lookup may return an Error referring to strings stored internally by
  // the JIT. If the Error outlives the ExecutionEngine, it would want have a
  // dangling reference, which is currently caught by an assertion inside JIT
  // thanks to hand-rolled reference counting. Rewrap the error message into a
  // string before returning. Alternatively, ORC JIT should consider copying
  // the string into the error message.
  if (!expectedSymbol) {
    std::string errorMessage;
    llvm::raw_string_ostream os(errorMessage);
    llvm::handleAllErrors(expectedSymbol.takeError(),
                          [&os](llvm::ErrorInfoBase &ei) { ei.log(os); });
    return make_string_error(os.str());
  }

  auto rawFPtr = expectedSymbol->getAddress();
  auto fptr = reinterpret_cast<void (*)(void **)>(rawFPtr);
  if (!fptr)
    return make_string_error("looked up function is null");
  return fptr;
}

Error ExecutionEngine::invokePacked(StringRef name,
                                    MutableArrayRef<void *> args) {
  auto expectedFPtr = lookup(name);
  if (!expectedFPtr)
    return expectedFPtr.takeError();
  auto fptr = *expectedFPtr;

  (*fptr)(args.data());

  return Error::success();
}
