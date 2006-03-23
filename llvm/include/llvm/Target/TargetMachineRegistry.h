//===-- Target/TargetMachineRegistry.h - Target Registration ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes two classes: the TargetMachineRegistry class, which allows
// tools to inspect all of registered targets, and the RegisterTarget class,
// which TargetMachine implementations should use to register themselves with
// the system.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETMACHINEREGISTRY_H
#define LLVM_TARGET_TARGETMACHINEREGISTRY_H

#include "llvm/Support/CommandLine.h"

namespace llvm {
  class Module;
  class TargetMachine;

  struct TargetMachineRegistry {
    struct Entry;

    /// TargetMachineRegistry::getList - This static method returns the list of
    /// target machines that are registered with the system.
    static const Entry *getList() { return List; }

    /// getClosestStaticTargetForModule - Given an LLVM module, pick the best
    /// target that is compatible with the module.  If no close target can be
    /// found, this returns null and sets the Error string to a reason.
    static const Entry *getClosestStaticTargetForModule(const Module &M,
                                                        std::string &Error);

    /// getClosestTargetForJIT - Given an LLVM module, pick the best target that
    /// is compatible with the current host and the specified module.  If no
    /// close target can be found, this returns null and sets the Error string
    /// to a reason.
    static const Entry *getClosestTargetForJIT(std::string &Error);


    /// Entry - One instance of this struct is created for each target that is
    /// registered.
    struct Entry {
      const char *Name;
      const char *ShortDesc;
      TargetMachine *(*CtorFn)(const Module &, const std::string &);
      unsigned (*ModuleMatchQualityFn)(const Module &M);
      unsigned (*JITMatchQualityFn)();

      const Entry *getNext() const { return Next; }

    protected:
      Entry(const char *N, const char *SD,
            TargetMachine *(*CF)(const Module &, const std::string &),
            unsigned (*MMF)(const Module &M), unsigned (*JMF)());
    private:
      const Entry *Next;  // Next entry in the linked list.
    };

  private:
    static const Entry *List;
  };

  //===--------------------------------------------------------------------===//
  /// RegisterTarget - This class is used to make targets automatically register
  /// themselves with the tool they are linked.  Targets should define an
  /// instance of this and implement the static methods described in the
  /// TargetMachine comments..
  template<class TargetMachineImpl>
  struct RegisterTarget : public TargetMachineRegistry::Entry {
    RegisterTarget(const char *Name, const char *ShortDesc) :
      TargetMachineRegistry::Entry(Name, ShortDesc, &Allocator,
                                   &TargetMachineImpl::getModuleMatchQuality,
                                   &TargetMachineImpl::getJITMatchQuality) {
    }
  private:
    static TargetMachine *Allocator(const Module &M, const std::string &FS) {
      return new TargetMachineImpl(M, 0, FS);
    }
  };

  /// TargetRegistrationListener - This class allows code to listen for targets
  /// that are dynamically registered, and be notified of it when they are.
  class TargetRegistrationListener {
    TargetRegistrationListener **Prev, *Next;
  public:
    TargetRegistrationListener();
    virtual ~TargetRegistrationListener();

    TargetRegistrationListener *getNext() const { return Next; }

    virtual void targetRegistered(const TargetMachineRegistry::Entry *E) = 0;
  };


  //===--------------------------------------------------------------------===//
  /// TargetNameParser - This option can be used to provide a command line
  /// option to choose among the various registered targets (commonly -march).
  class TargetNameParser : public TargetRegistrationListener,
    public cl::parser<const TargetMachineRegistry::Entry*> {
  public:
    void initialize(cl::Option &O) {
      for (const TargetMachineRegistry::Entry *E =
             TargetMachineRegistry::getList(); E; E = E->getNext())
        Values.push_back(std::make_pair(E->Name,
                                        std::make_pair(E, E->ShortDesc)));
      cl::parser<const TargetMachineRegistry::Entry*>::initialize(O);
    }

    virtual void targetRegistered(const TargetMachineRegistry::Entry *E) {
      Values.push_back(std::make_pair(E->Name,
                                      std::make_pair(E, E->ShortDesc)));
    }
  };
}

#endif
