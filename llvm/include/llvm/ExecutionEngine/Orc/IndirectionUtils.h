//===-- IndirectionUtils.h - Utilities for adding indirections --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains utilities for adding indirections and breaking up modules.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_INDIRECTIONUTILS_H
#define LLVM_EXECUTIONENGINE_ORC_INDIRECTIONUTILS_H

#include "JITSymbol.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include <sstream>

namespace llvm {
namespace orc {

/// @brief Base class for JITLayer independent aspects of
///        JITCompileCallbackManager.
class JITCompileCallbackManagerBase {
public:

  typedef std::function<TargetAddress()> CompileFtor;
  typedef std::function<void(TargetAddress)> UpdateFtor;

  /// @brief Handle to a newly created compile callback. Can be used to get an
  ///        IR constant representing the address of the trampoline, and to set
  ///        the compile and update actions for the callback.
  class CompileCallbackInfo {
  public:
    CompileCallbackInfo(TargetAddress Addr, CompileFtor &Compile,
                        UpdateFtor &Update)
      : Addr(Addr), Compile(Compile), Update(Update) {}

    TargetAddress getAddress() const { return Addr; }
    void setCompileAction(CompileFtor Compile) {
      this->Compile = std::move(Compile);
    }
    void setUpdateAction(UpdateFtor Update) {
      this->Update = std::move(Update);
    }
  private:
    TargetAddress Addr;
    CompileFtor &Compile;
    UpdateFtor &Update;
  };

  /// @brief Construct a JITCompileCallbackManagerBase.
  /// @param ErrorHandlerAddress The address of an error handler in the target
  ///                            process to be used if a compile callback fails.
  /// @param NumTrampolinesPerBlock Number of trampolines to emit if there is no
  ///                             available trampoline when getCompileCallback is
  ///                             called.
  JITCompileCallbackManagerBase(TargetAddress ErrorHandlerAddress,
                                unsigned NumTrampolinesPerBlock)
    : ErrorHandlerAddress(ErrorHandlerAddress),
      NumTrampolinesPerBlock(NumTrampolinesPerBlock) {}

  virtual ~JITCompileCallbackManagerBase() {}

  /// @brief Execute the callback for the given trampoline id. Called by the JIT
  ///        to compile functions on demand.
  TargetAddress executeCompileCallback(TargetAddress TrampolineID) {
    TrampolineMapT::iterator I = ActiveTrampolines.find(TrampolineID);
    // FIXME: Also raise an error in the Orc error-handler when we finally have
    //        one.
    if (I == ActiveTrampolines.end())
      return ErrorHandlerAddress;

    // Found a callback handler. Yank this trampoline out of the active list and
    // put it back in the available trampolines list, then try to run the
    // handler's compile and update actions.
    // Moving the trampoline ID back to the available list first means there's at
    // least one available trampoline if the compile action triggers a request for
    // a new one.
    AvailableTrampolines.push_back(I->first);
    auto CallbackHandler = std::move(I->second);
    ActiveTrampolines.erase(I);

    if (auto Addr = CallbackHandler.Compile()) {
      CallbackHandler.Update(Addr);
      return Addr;
    }
    return ErrorHandlerAddress;
  }

  /// @brief Get/create a compile callback with the given signature.
  virtual CompileCallbackInfo getCompileCallback(LLVMContext &Context) = 0;

protected:

  struct CallbackHandler {
    CompileFtor Compile;
    UpdateFtor Update;
  };

  TargetAddress ErrorHandlerAddress;
  unsigned NumTrampolinesPerBlock;

  typedef std::map<TargetAddress, CallbackHandler> TrampolineMapT;
  TrampolineMapT ActiveTrampolines;
  std::vector<TargetAddress> AvailableTrampolines;
};

/// @brief Manage compile callbacks.
template <typename JITLayerT, typename TargetT>
class JITCompileCallbackManager : public JITCompileCallbackManagerBase {
public:

  /// @brief Construct a JITCompileCallbackManager.
  /// @param JIT JIT layer to emit callback trampolines, etc. into.
  /// @param Context LLVMContext to use for trampoline & resolve block modules.
  /// @param ErrorHandlerAddress The address of an error handler in the target
  ///                            process to be used if a compile callback fails.
  /// @param NumTrampolinesPerBlock Number of trampolines to allocate whenever
  ///                               there is no existing callback trampoline.
  ///                               (Trampolines are allocated in blocks for
  ///                               efficiency.)
  JITCompileCallbackManager(JITLayerT &JIT, RuntimeDyld::MemoryManager &MemMgr,
                            LLVMContext &Context,
                            TargetAddress ErrorHandlerAddress,
                            unsigned NumTrampolinesPerBlock)
    : JITCompileCallbackManagerBase(ErrorHandlerAddress,
                                    NumTrampolinesPerBlock),
      JIT(JIT), MemMgr(MemMgr) {
    emitResolverBlock(Context);
  }

  /// @brief Get/create a compile callback with the given signature.
  CompileCallbackInfo getCompileCallback(LLVMContext &Context) final {
    TargetAddress TrampolineAddr = getAvailableTrampolineAddr(Context);
    auto &CallbackHandler =
      this->ActiveTrampolines[TrampolineAddr];

    return CompileCallbackInfo(TrampolineAddr, CallbackHandler.Compile,
                               CallbackHandler.Update);
  }

private:

  std::vector<std::unique_ptr<Module>>
  SingletonSet(std::unique_ptr<Module> M) {
    std::vector<std::unique_ptr<Module>> Ms;
    Ms.push_back(std::move(M));
    return Ms;
  }

  void emitResolverBlock(LLVMContext &Context) {
    std::unique_ptr<Module> M(new Module("resolver_block_module",
                                         Context));
    TargetT::insertResolverBlock(*M, *this);
    auto H = JIT.addModuleSet(SingletonSet(std::move(M)), &MemMgr,
                              static_cast<RuntimeDyld::SymbolResolver*>(
                                  nullptr));
    JIT.emitAndFinalize(H);
    auto ResolverBlockSymbol =
      JIT.findSymbolIn(H, TargetT::ResolverBlockName, false);
    assert(ResolverBlockSymbol && "Failed to insert resolver block");
    ResolverBlockAddr = ResolverBlockSymbol.getAddress();
  }

  TargetAddress getAvailableTrampolineAddr(LLVMContext &Context) {
    if (this->AvailableTrampolines.empty())
      grow(Context);
    assert(!this->AvailableTrampolines.empty() &&
           "Failed to grow available trampolines.");
    TargetAddress TrampolineAddr = this->AvailableTrampolines.back();
    this->AvailableTrampolines.pop_back();
    return TrampolineAddr;
  }

  void grow(LLVMContext &Context) {
    assert(this->AvailableTrampolines.empty() && "Growing prematurely?");
    std::unique_ptr<Module> M(new Module("trampoline_block", Context));
    auto GetLabelName =
      TargetT::insertCompileCallbackTrampolines(*M, ResolverBlockAddr,
                                                this->NumTrampolinesPerBlock,
                                                this->ActiveTrampolines.size());
    auto H = JIT.addModuleSet(SingletonSet(std::move(M)), &MemMgr,
                              static_cast<RuntimeDyld::SymbolResolver*>(
                                  nullptr));
    JIT.emitAndFinalize(H);
    for (unsigned I = 0; I < this->NumTrampolinesPerBlock; ++I) {
      std::string Name = GetLabelName(I);
      auto TrampolineSymbol = JIT.findSymbolIn(H, Name, false);
      assert(TrampolineSymbol && "Failed to emit trampoline.");
      this->AvailableTrampolines.push_back(TrampolineSymbol.getAddress());
    }
  }

  JITLayerT &JIT;
  RuntimeDyld::MemoryManager &MemMgr;
  TargetAddress ResolverBlockAddr;
};

/// @brief Get an update functor that updates the value of a named function
///        pointer.
template <typename JITLayerT>
JITCompileCallbackManagerBase::UpdateFtor
getLocalFPUpdater(JITLayerT &JIT, typename JITLayerT::ModuleSetHandleT H,
                  std::string Name) {
    // FIXME: Move-capture Name once we can use C++14.
    return [=,&JIT](TargetAddress Addr) {
      auto FPSym = JIT.findSymbolIn(H, Name, true);
      assert(FPSym && "Cannot find function pointer to update.");
      void *FPAddr = reinterpret_cast<void*>(
                       static_cast<uintptr_t>(FPSym.getAddress()));
      memcpy(FPAddr, &Addr, sizeof(uintptr_t));
    };
  }

/// @brief Build a function pointer of FunctionType with the given constant
///        address.
///
///   Usage example: Turn a trampoline address into a function pointer constant
/// for use in a stub.
Constant* createIRTypedAddress(FunctionType &FT, TargetAddress Addr);

/// @brief Create a function pointer with the given type, name, and initializer
///        in the given Module.
GlobalVariable* createImplPointer(PointerType &PT, Module &M,
                                  const Twine &Name, Constant *Initializer);

/// @brief Turn a function declaration into a stub function that makes an
///        indirect call using the given function pointer.
void makeStub(Function &F, GlobalVariable &ImplPointer);

typedef std::map<Module*, DenseSet<const GlobalValue*>> ModulePartitionMap;

/// @brief Extract subsections of a Module into the given Module according to
///        the given ModulePartitionMap.
void partition(Module &M, const ModulePartitionMap &PMap);

/// @brief Struct for trivial "complete" partitioning of a module.
class FullyPartitionedModule {
public:
  std::unique_ptr<Module> GlobalVars;
  std::unique_ptr<Module> Commons;
  std::vector<std::unique_ptr<Module>> Functions;

  FullyPartitionedModule() = default;
  FullyPartitionedModule(FullyPartitionedModule &&S)
      : GlobalVars(std::move(S.GlobalVars)), Commons(std::move(S.Commons)),
        Functions(std::move(S.Functions)) {}
};

/// @brief Extract every function in M into a separate module.
FullyPartitionedModule fullyPartition(Module &M);

} // End namespace orc.
} // End namespace llvm.

#endif // LLVM_EXECUTIONENGINE_ORC_INDIRECTIONUTILS_H
