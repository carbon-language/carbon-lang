//===- llvm/PassManager.h - Container for Passes -----------------*- C++ -*--=//
//
// This file defines the PassManager class.  This class is used to hold,
// maintain, and optimize execution of Passes.  The PassManager class ensures
// that analysis results are available before a pass runs, and that Pass's are
// destroyed when the PassManager is destroyed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSMANAGER_H
#define LLVM_PASSMANAGER_H

class Pass;
class Module;
template<class UnitType> class PassManagerT;

class PassManager {
  PassManagerT<Module> *PM;    // This is a straightforward Pimpl class
public:
  PassManager();
  ~PassManager();

  /// add - Add a pass to the queue of passes to run.  This passes ownership of
  /// the Pass to the PassManager.  When the PassManager is destroyed, the pass
  /// will be destroyed as well, so there is no need to delete the pass.  This
  /// implies that all passes MUST be allocated with 'new'.
  ///
  void add(Pass *P);

  /// run - Execute all of the passes scheduled for execution.  Keep track of
  /// whether any of the passes modifies the module, and if so, return true.
  ///
  bool run(Module &M);
};

class FunctionPass;
class ImmutablePass;
class Function;

class FunctionPassManager {
  PassManagerT<Function> *PM;    // This is a straightforward Pimpl class
public:
  FunctionPassManager();
  ~FunctionPassManager();

  /// add - Add a pass to the queue of passes to run.  This passes
  /// ownership of the FunctionPass to the PassManager.  When the
  /// PassManager is destroyed, the pass will be destroyed as well, so
  /// there is no need to delete the pass.  This implies that all
  /// passes MUST be allocated with 'new'.
  ///
  void add(FunctionPass *P);

  /// add - ImmutablePasses are not FunctionPasses, so we have a 
  /// special hack to get them into a FunctionPassManager.
  ///
  void add(ImmutablePass *IP);

  /// run - Execute all of the passes scheduled for execution.  Keep
  /// track of whether any of the passes modifies the function, and if
  /// so, return true.
  ///
  bool run(Function &M);
};

#endif
