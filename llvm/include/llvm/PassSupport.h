//===- llvm/PassSupport.h - Pass Support code -------------------*- C++ -*-===//
//
// This file defines stuff that is used to define and "use" Passes.  This file
// is automatically #included by Pass.h, so:
//
//           NO .CPP FILES SHOULD INCLUDE THIS FILE DIRECTLY
//
// Instead, #include Pass.h.
//
// This file defines Pass registration code and classes used for it.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASS_SUPPORT_H
#define LLVM_PASS_SUPPORT_H

// No need to include Pass.h, we are being included by it!

#include <typeinfo>
class TargetData;

//===---------------------------------------------------------------------------
// PassInfo class - An instance of this class exists for every pass known by the
// system, and can be obtained from a live Pass by calling its getPassInfo()
// method.  These objects are set up by the RegisterPass<> template, defined
// below.
//
class PassInfo {
  const char           *PassName;      // Nice name for Pass
  const char           *PassArgument;  // Command Line argument to run this pass
  const std::type_info &TypeInfo;      // type_info object for this Pass class

  Pass *(*NormalCtor)();               // No argument ctor
  Pass *(*DataCtor)(const TargetData&);// Ctor taking TargetData object...

public:
  // PassInfo ctor - Do not call this directly, this should only be invoked
  // through RegisterPass.
  PassInfo(const char *name, const char *arg, const std::type_info &ti, 
           Pass *(*normal)(), Pass *(*data)(const TargetData &))
    : PassName(name), PassArgument(arg), TypeInfo(ti), NormalCtor(normal), 
      DataCtor(data) {
  }

  // getPassName - Return the friendly name for the pass, never returns null
  const char *getPassName() const { return PassName; }

  // getPassArgument - Return the command line option that may be passed to
  // 'opt' that will cause this pass to be run.  This will return null if there
  // is no argument.
  //
  const char *getPassArgument() const { return PassArgument; }

  // getTypeInfo - Return the type_info object for the pass...
  const std::type_info &getTypeInfo() const { return TypeInfo; }

  // getNormalCtor - Return a pointer to a function, that when called, creates
  // an instance of the pass and returns it.  This pointer may be null if there
  // is no default constructor for the pass.
  
  Pass *(*getNormalCtor() const)() {
    return NormalCtor;
  }

  // getDataCtor - Return a pointer to a function that creates an instance of
  // the pass and returns it.  This returns a constructor for a version of the
  // pass that takes a TArgetData object as a parameter.
  //
  Pass *(*getDataCtor() const)(const TargetData &) {
    return DataCtor;
  }
};


//===---------------------------------------------------------------------------
// RegisterPass<t> template - This template class is used to notify the system
// that a Pass is available for use, and registers it into the internal database
// maintained by the PassManager.  Unless this template is used, opt, for
// example will not be able to see the pass and attempts to create the pass will
// fail. This template is used in the follow manner (at global scope, in your
// .cpp file):
// 
// static RegisterPass<YourPassClassName> tmp("passopt", "My Pass Name");
//
// This statement will cause your pass to be created by calling the default
// constructor exposed by the pass.  If you have a different constructor that
// must be called, create a global constructor function (which takes the
// arguments you need and returns a Pass*) and register your pass like this:
//
// Pass *createMyPass(foo &opt) { return new MyPass(opt); }
// static RegisterPass<PassClassName> tmp("passopt", "My Name", createMyPass);
// 
struct RegisterPassBase {
  // getPassInfo - Get the pass info for the registered class...
  const PassInfo *getPassInfo() const { return PIObj; }

  ~RegisterPassBase();   // Intentionally non-virtual...

protected:
  PassInfo *PIObj;       // The PassInfo object for this pass
  void registerPass(PassInfo *);
};

template<typename PassName>
Pass *callDefaultCtor() { return new PassName(); }

template<typename PassName>
struct RegisterPass : public RegisterPassBase {
  
  // Register Pass using default constructor...
  RegisterPass(const char *PassArg, const char *Name) {
    registerPass(new PassInfo(Name, PassArg, typeid(PassName),
                              callDefaultCtor<PassName>, 0));
  }

  // Register Pass using default constructor explicitly...
  RegisterPass(const char *PassArg, const char *Name,
               Pass *(*ctor)()) {
    registerPass(new PassInfo(Name, PassArg, typeid(PassName), ctor, 0));
  }

  // Register Pass using TargetData constructor...
  RegisterPass(const char *PassArg, const char *Name,
               Pass *(*datactor)(const TargetData &)) {
    registerPass(new PassInfo(Name, PassArg, typeid(PassName), 0, datactor));
  }

  // Generic constructor version that has an unknown ctor type...
  template<typename CtorType>
  RegisterPass(const char *PassArg, const char *Name, CtorType *Fn) {
    registerPass(new PassInfo(Name, PassArg, typeid(PassName), 0, 0));
  }
};


//===---------------------------------------------------------------------------
// PassRegistrationListener class - This class is meant to be derived from by
// clients that are interested in which passes get registered and unregistered
// at runtime (which can be because of the RegisterPass constructors being run
// as the program starts up, or may be because a shared object just got loaded).
// Deriving from the PassRegistationListener class automatically registers your
// object to receive callbacks indicating when passes are loaded and removed.
//
struct PassRegistrationListener {

  // PassRegistrationListener ctor - Add the current object to the list of
  // PassRegistrationListeners...
  PassRegistrationListener();

  // dtor - Remove object from list of listeners...
  virtual ~PassRegistrationListener();

  // Callback functions - These functions are invoked whenever a pass is loaded
  // or removed from the current executable.
  //
  virtual void passRegistered(const PassInfo *P) {}
  virtual void passUnregistered(const PassInfo *P) {}

  // enumeratePasses - Iterate over the registered passes, calling the
  // passEnumerate callback on each PassInfo object.
  //
  void enumeratePasses();

  // passEnumerate - Callback function invoked when someone calls
  // enumeratePasses on this PassRegistrationListener object.
  //
  virtual void passEnumerate(const PassInfo *P) {}
};

#endif
