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
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include <sstream>

namespace llvm {
namespace orc {

/// @brief Base class for JITLayer independent aspects of
///        JITCompileCallbackManager.
template <typename TargetT>
class JITCompileCallbackManagerBase {
public:

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

  /// @brief Execute the callback for the given trampoline id. Called by the JIT
  ///        to compile functions on demand.
  TargetAddress executeCompileCallback(TargetAddress TrampolineID) {
    typename TrampolineMapT::iterator I = ActiveTrampolines.find(TrampolineID);
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
    AvailableTrampolines.push_back(I->first - TargetT::CallSize);
    auto CallbackHandler = std::move(I->second);
    ActiveTrampolines.erase(I);

    if (auto Addr = CallbackHandler.Compile()) {
      CallbackHandler.Update(Addr);
      return Addr;
    }
    return ErrorHandlerAddress;
  }

protected:

  typedef std::function<TargetAddress()> CompileFtorT;
  typedef std::function<void(TargetAddress)> UpdateFtorT;

  struct CallbackHandler {
    CompileFtorT Compile;
    UpdateFtorT Update;
  };

  TargetAddress ErrorHandlerAddress;
  unsigned NumTrampolinesPerBlock;

  typedef std::map<TargetAddress, CallbackHandler> TrampolineMapT;
  TrampolineMapT ActiveTrampolines;
  std::vector<TargetAddress> AvailableTrampolines;
};

/// @brief Manage compile callbacks.
template <typename JITLayerT, typename TargetT>
class JITCompileCallbackManager :
    public JITCompileCallbackManagerBase<TargetT> {
public:

  typedef typename JITCompileCallbackManagerBase<TargetT>::CompileFtorT
    CompileFtorT;
  typedef typename JITCompileCallbackManagerBase<TargetT>::UpdateFtorT
    UpdateFtorT;

  /// @brief Construct a JITCompileCallbackManager.
  /// @param JIT JIT layer to emit callback trampolines, etc. into.
  /// @param Context LLVMContext to use for trampoline & resolve block modules.
  /// @param ErrorHandlerAddress The address of an error handler in the target
  ///                            process to be used if a compile callback fails.
  /// @param NumTrampolinesPerBlock Number of trampolines to allocate whenever
  ///                               there is no existing callback trampoline.
  ///                               (Trampolines are allocated in blocks for
  ///                               efficiency.)
  JITCompileCallbackManager(JITLayerT &JIT, LLVMContext &Context,
                            TargetAddress ErrorHandlerAddress,
                            unsigned NumTrampolinesPerBlock)
    : JITCompileCallbackManagerBase<TargetT>(ErrorHandlerAddress,
                                             NumTrampolinesPerBlock),
      JIT(JIT) {
    emitResolverBlock(Context);
  }

  /// @brief Handle to a newly created compile callback. Can be used to get an
  ///        IR constant representing the address of the trampoline, and to set
  ///        the compile and update actions for the callback.
  class CompileCallbackInfo {
  public:
    CompileCallbackInfo(Constant *Addr, CompileFtorT &Compile,
                        UpdateFtorT &Update)
      : Addr(Addr), Compile(Compile), Update(Update) {}

    Constant* getAddress() const { return Addr; }
    void setCompileAction(CompileFtorT Compile) {
      this->Compile = std::move(Compile);
    }
    void setUpdateAction(UpdateFtorT Update) {
      this->Update = std::move(Update);
    }
  private:
    Constant *Addr;
    CompileFtorT &Compile;
    UpdateFtorT &Update;
  };

  /// @brief Get/create a compile callback with the given signature.
  CompileCallbackInfo getCompileCallback(FunctionType &FT) {
    TargetAddress TrampolineAddr = getAvailableTrampolineAddr(FT.getContext());
    auto &CallbackHandler =
      this->ActiveTrampolines[TrampolineAddr + TargetT::CallSize];
    Constant *AddrIntVal =
      ConstantInt::get(Type::getInt64Ty(FT.getContext()), TrampolineAddr);
    Constant *AddrPtrVal =
      ConstantExpr::getCast(Instruction::IntToPtr, AddrIntVal,
                            PointerType::get(&FT, 0));

    return CompileCallbackInfo(AddrPtrVal, CallbackHandler.Compile,
                               CallbackHandler.Update);
  }

  /// @brief Get a functor for updating the value of a named function pointer.
  UpdateFtorT getLocalFPUpdater(typename JITLayerT::ModuleSetHandleT H,
                                std::string Name) {
    // FIXME: Move-capture Name once we can use C++14.
    return [=](TargetAddress Addr) {
      auto FPSym = JIT.findSymbolIn(H, Name, true);
      assert(FPSym && "Cannot find function pointer to update.");
      void *FPAddr = reinterpret_cast<void*>(
                       static_cast<uintptr_t>(FPSym.getAddress()));
      memcpy(FPAddr, &Addr, sizeof(uintptr_t));
    };
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
    auto H = JIT.addModuleSet(SingletonSet(std::move(M)), nullptr);
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
    auto H = JIT.addModuleSet(SingletonSet(std::move(M)), nullptr);
    JIT.emitAndFinalize(H);
    for (unsigned I = 0; I < this->NumTrampolinesPerBlock; ++I) {
      std::string Name = GetLabelName(I);
      auto TrampolineSymbol = JIT.findSymbolIn(H, Name, false);
      assert(TrampolineSymbol && "Failed to emit trampoline.");
      this->AvailableTrampolines.push_back(TrampolineSymbol.getAddress());
    }
  }

  JITLayerT &JIT;
  TargetAddress ResolverBlockAddr;
};

GlobalVariable* createImplPointer(Function &F, const Twine &Name,
                                  Constant *Initializer);

void makeStub(Function &F, GlobalVariable &ImplPointer);

typedef std::map<Module*, DenseSet<const GlobalValue*>> ModulePartitionMap;

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

FullyPartitionedModule fullyPartition(Module &M);

} // End namespace orc.
} // End namespace llvm.

#endif // LLVM_EXECUTIONENGINE_ORC_INDIRECTIONUTILS_H
