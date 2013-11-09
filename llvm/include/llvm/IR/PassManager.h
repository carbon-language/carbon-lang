//===- PassManager.h - LegacyContainer for Passes --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This header defines various interfaces for pass management in LLVM. There
/// is no "pass" interface in LLVM per se. Instead, an instance of any class
/// which supports a method to 'run' it over a unit of IR can be used as
/// a pass. A pass manager is generally a tool to collect a sequence of passes
/// which run over a particular IR construct, and run each of them in sequence
/// over each such construct in the containing IR construct. As there is no
/// containing IR construct for a Module, a manager for passes over modules
/// forms the base case which runs its managed passes in sequence over the
/// single module provided.
///
/// The core IR library provides managers for running passes over
/// modules and functions.
///
/// * FunctionPassManager can run over a Module, runs each pass over
///   a Function.
/// * ModulePassManager must be directly run, runs each pass over the Module.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/polymorphic_ptr.h"
#include "llvm/IR/Module.h"
#include <vector>

namespace llvm {

class Module;
class Function;

/// \brief Implementation details of the pass manager interfaces.
namespace detail {

/// \brief Template for the abstract base class used to dispatch
/// polymorphically over pass objects.
template <typename T> struct PassConcept {
  // Boiler plate necessary for the container of derived classes.
  virtual ~PassConcept() {}
  virtual PassConcept *clone() = 0;

  /// \brief The polymorphic API which runs the pass over a given IR entity.
  virtual bool run(T Arg) = 0;
};

/// \brief A template wrapper used to implement the polymorphic API.
///
/// Can be instantiated for any object which provides a \c run method
/// accepting a \c T. It requires the pass to be a copyable
/// object.
template <typename T, typename PassT> struct PassModel : PassConcept<T> {
  PassModel(PassT Pass) : Pass(llvm_move(Pass)) {}
  virtual PassModel *clone() { return new PassModel(Pass); }
  virtual bool run(T Arg) { return Pass.run(Arg); }
  PassT Pass;
};

}

class ModulePassManager {
public:
  ModulePassManager(Module *M) : M(M) {}

  template <typename ModulePassT> void addPass(ModulePassT Pass) {
    Passes.push_back(new ModulePassModel<ModulePassT>(llvm_move(Pass)));
  }

  void run() {
    for (unsigned Idx = 0, Size = Passes.size(); Idx != Size; ++Idx)
      Passes[Idx]->run(M);
  }

private:
  // Pull in the concept type and model template specialized for modules.
  typedef detail::PassConcept<Module *> ModulePassConcept;
  template <typename PassT>
  struct ModulePassModel : detail::PassModel<Module *, PassT> {
    ModulePassModel(PassT Pass) : detail::PassModel<Module *, PassT>(Pass) {}
  };

  Module *M;
  std::vector<polymorphic_ptr<ModulePassConcept> > Passes;
};

class FunctionPassManager {
public:
  template <typename FunctionPassT> void addPass(FunctionPassT Pass) {
    Passes.push_back(new FunctionPassModel<FunctionPassT>(llvm_move(Pass)));
  }

  bool run(Module *M) {
    bool Changed = false;
    for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
      for (unsigned Idx = 0, Size = Passes.size(); Idx != Size; ++Idx)
        Changed |= Passes[Idx]->run(I);
    return Changed;
  }

private:
  // Pull in the concept type and model template specialized for functions.
  typedef detail::PassConcept<Function *> FunctionPassConcept;
  template <typename PassT>
  struct FunctionPassModel : detail::PassModel<Function *, PassT> {
    FunctionPassModel(PassT Pass)
        : detail::PassModel<Function *, PassT>(Pass) {}
  };

  std::vector<polymorphic_ptr<FunctionPassConcept> > Passes;
};

}
