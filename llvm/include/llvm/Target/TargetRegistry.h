//===-- Target/TargetRegistry.h - Target Registration -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes the TargetRegistry interface, which tools can use to access
// the appropriate target specific classes (TargetMachine, AsmPrinter, etc.)
// which have been registered.
//
// Target specific class implementations should register themselves using the
// appropriate TargetRegistry interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETREGISTRY_H
#define LLVM_TARGET_TARGETREGISTRY_H

#include <string>
#include <cassert>

namespace llvm {
  class FunctionPass;
  class Module;
  class TargetMachine;
  class formatted_raw_ostream;

  /// Target - Wrapper for Target specific information.
  ///
  /// For registration purposes, this is a POD type so that targets can be
  /// registered without the use of static constructors.
  ///
  /// Targets should implement a single global instance of this class (which
  /// will be zero initialized), and pass that instance to the TargetRegistry as
  /// part of their initialization.
  class Target {
  private:
    typedef unsigned (*TripleMatchQualityFnTy)(const std::string &TT);
    typedef unsigned (*ModuleMatchQualityFnTy)(const Module &M);
    typedef unsigned (*JITMatchQualityFnTy)();

    typedef TargetMachine *(*TargetMachineCtorTy)(const Module &, 
                                                  const std::string &);
    typedef FunctionPass *(*AsmPrinterCtorTy)(formatted_raw_ostream &,
                                              TargetMachine &,
                                              bool);

    friend struct TargetRegistry;
    // FIXME: Temporary hack, please remove.
    friend struct TargetMachineRegistry;

    /// Next - The next registered target in the linked list, maintained by the
    /// TargetRegistry.
    Target *Next;

    /// TripleMatchQualityFn - The target function for rating the match quality
    /// of a triple.
    TripleMatchQualityFnTy TripleMatchQualityFn;

    /// ModuleMatchQualityFn - The target function for rating the match quality
    /// of a module.
    ModuleMatchQualityFnTy ModuleMatchQualityFn;

    /// JITMatchQualityFn - The target function for rating the match quality
    /// with the host.
    JITMatchQualityFnTy JITMatchQualityFn;

    /// Name - The target name.
    const char *Name;

    /// ShortDesc - A short description of the target.
    const char *ShortDesc;

    /// TargetMachineCtorFn - Construction function for this target's
    /// TargetMachine, if registered.
    TargetMachineCtorTy TargetMachineCtorFn;

    /// AsmPrinterCtorFn - Construction function for this target's AsmPrinter,
    /// if registered.
    AsmPrinterCtorTy AsmPrinterCtorFn;

  public:
    /// getName - Get the target name.
    const char *getName() const { return Name; }

    /// getShortDescription - Get a short description of the target.
    const char *getShortDescription() const { return ShortDesc; }

    /// createTargetMachine - Create a target specific machine implementation.
    TargetMachine *createTargetMachine(const Module &M,
                                       const std::string &Features) const {
      if (!TargetMachineCtorFn)
        return 0;
      return TargetMachineCtorFn(M, Features);
    }

    /// createAsmPrinter - Create a target specific assembly printer pass.
    FunctionPass *createAsmPrinter(formatted_raw_ostream &OS,
                                   TargetMachine &M,
                                   bool Verbose) const {
      if (!AsmPrinterCtorFn)
        return 0;
      return AsmPrinterCtorFn(OS, M, Verbose);
    }
  };

  /// TargetRegistry - Generic interface to target specific features.
  //
  // FIXME: Provide Target* iterator.
  struct TargetRegistry {
    /// @name Registry Access
    /// @{

    /// getClosestStaticTargetForTriple - Given a target triple, pick the most
    /// capable target for that triple.
    static const Target *getClosestStaticTargetForTriple(const std::string &TT,
                                                         std::string &Error);

    /// getClosestStaticTargetForModule - Given an LLVM module, pick the best
    /// target that is compatible with the module.  If no close target can be
    /// found, this returns null and sets the Error string to a reason.
    static const Target *getClosestStaticTargetForModule(const Module &M,
                                                        std::string &Error);

    /// getClosestTargetForJIT - Pick the best target that is compatible with
    /// the current host.  If no close target can be found, this returns null
    /// and sets the Error string to a reason.
    //
    // FIXME: Do we still need this interface, clients can always look for the
    // match for the host triple.
    static const Target *getClosestTargetForJIT(std::string &Error);

    /// @}
    /// @name Target Registration
    /// @{

    /// RegisterTarget - Register the given target.
    /// 
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    ///
    /// @param T - The target being registered.
    /// @param Name - The target name. This should be a static string.
    /// @param ShortDesc - A short target description. This should be a static
    /// string. 
    /// @param TQualityFn - The triple match quality computation function for
    /// this target.
    /// @param MQualityFn - The module match quality computation function for
    /// this target.
    /// @param JITMatchQualityFn - The JIT match quality computation function
    /// for this target.
    static void RegisterTarget(Target &T,
                               const char *Name,
                               const char *ShortDesc,
                               Target::TripleMatchQualityFnTy TQualityFn,
                               Target::ModuleMatchQualityFnTy MQualityFn,
                               Target::JITMatchQualityFnTy JITQualityFn);
                               
    /// RegisterTargetMachine - Register a TargetMachine implementation for the
    /// given target.
    /// 
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    /// 
    /// @param T - The target being registered.
    /// @param Fn - A function to construct a TargetMachine for the target.
    static void RegisterTargetMachine(Target &T, 
                                      Target::TargetMachineCtorTy Fn) {
      assert(!T.TargetMachineCtorFn && "Constructor already registered!");
      T.TargetMachineCtorFn = Fn;
    }

    /// RegisterAsmPrinter - Register an AsmPrinter implementation for the given
    /// target.
    /// 
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    ///
    /// @param T - The target being registered.
    /// @param Fn - A function to construct an AsmPrinter for the target.
    static void RegisterAsmPrinter(Target &T, Target::AsmPrinterCtorTy Fn) {
      assert(!T.AsmPrinterCtorFn && "Constructor already registered!");
      T.AsmPrinterCtorFn = Fn;
    }

    /// @}
  };

}

#endif
