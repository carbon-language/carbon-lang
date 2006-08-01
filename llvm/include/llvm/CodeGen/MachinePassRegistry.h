//===-- llvm/CodeGen/MachinePassRegistry.h ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the James M. Laskey and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the mechanics for machine function pass registries.  A
// function pass registry (MachinePassRegistry) is auto filled by the static
// constructors of MachinePassRegistryNode.  Further there is a command line
// parser (RegisterPassParser) which listens to each registry for additions
// and deletions, so that the appropriate command option is updated.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEPASSREGISTRY_H
#define LLVM_CODEGEN_MACHINEPASSREGISTRY_H

#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {


//===----------------------------------------------------------------------===// 
///
/// MachinePassRegistryListener - Listener to adds and removals of nodes in
/// registration list.
///
//===----------------------------------------------------------------------===//
class MachinePassRegistryListener {
public:
  MachinePassRegistryListener() {}
  virtual ~MachinePassRegistryListener() {}
  virtual void NotifyAdd(const char *N, const char *D) = 0;
  virtual void NotifyRemove(const char *N, const char *D) = 0;
};


//===----------------------------------------------------------------------===// 
///
/// MachinePassRegistryNode - Machine pass node stored in registration list.
///
//===----------------------------------------------------------------------===//
template<typename FunctionPassCtor>
class MachinePassRegistryNode {

private:

  MachinePassRegistryNode<FunctionPassCtor> *Next;// Next function pass in list.
  const char *Name;                     // Name of function pass.
  const char *Description;              // Description string.
  FunctionPassCtor Ctor;                // Function pass creator.
  
public:

  MachinePassRegistryNode(const char *N, const char *D, FunctionPassCtor C)
  : Next(NULL)
  , Name(N)
  , Description(D)
  , Ctor(C)
  {}

  // Accessors
  MachinePassRegistryNode<FunctionPassCtor> *getNext()
                                          const { return Next; }
  MachinePassRegistryNode<FunctionPassCtor> **getNextAddress()
                                                { return &Next; }
  const char *getName()                   const { return Name; }
  const char *getDescription()            const { return Description; }
  FunctionPassCtor getCtor()              const { return Ctor; }
  void setNext(MachinePassRegistryNode<FunctionPassCtor> *N) { Next = N; }
  
};


//===----------------------------------------------------------------------===// 
///
/// MachinePassRegistry - Track the registration of machine passes.
///
//===----------------------------------------------------------------------===//
template<typename FunctionPassCtor>
class MachinePassRegistry {

private:

  MachinePassRegistryNode<FunctionPassCtor> *List;
                                        // List of registry nodes.
  FunctionPassCtor Default;             // Default function pass creator.
  MachinePassRegistryListener* Listener;// Listener for list adds are removes.
  
public:

  // NO CONSTRUCTOR - we don't want static constructor ordering to mess
  // with the registry.

  // Accessors.
  //
  MachinePassRegistryNode<FunctionPassCtor> *getList()  { return List; }
  FunctionPassCtor getDefault()                         { return Default; }
  void setDefault(FunctionPassCtor C)                   { Default = C; }
  void setListener(MachinePassRegistryListener *L)      { Listener = L; }

  /// Add - Adds a function pass to the registration list.
  ///
 void Add(MachinePassRegistryNode<FunctionPassCtor> *Node) {
    Node->setNext(List);
    List = Node;
    if (Listener) Listener->NotifyAdd(Node->getName(), Node->getDescription());
  }


  /// Remove - Removes a function pass from the registration list.
  ///
  void Remove(MachinePassRegistryNode<FunctionPassCtor> *Node) {
    for (MachinePassRegistryNode<FunctionPassCtor> **I = &List;
         *I; I = (*I)->getNextAddress()) {
      if (*I == Node) {
        if (Listener) Listener->NotifyRemove(Node->getName(),
                                             Node->getDescription());
        *I = (*I)->getNext();
        break;
      }
    }
  }


  /// FInd - Finds and returns a function pass in registration list, otherwise
  /// returns NULL.
  MachinePassRegistryNode<FunctionPassCtor> *Find(const char *Name) {
    for (MachinePassRegistryNode<FunctionPassCtor> *I = List;
         I; I = I->getNext()) {
      if (std::string(Name) == std::string(I->getName())) return I;
    }
    return NULL;
  }


};


//===----------------------------------------------------------------------===//
///
/// RegisterRegAlloc class - Track the registration of register allocators.
///
//===----------------------------------------------------------------------===//
class RegisterRegAlloc : public MachinePassRegistryNode<FunctionPass *(*)()> {

public:

  typedef FunctionPass *(*FunctionPassCtor)();

  static MachinePassRegistry<FunctionPassCtor> Registry;

  RegisterRegAlloc(const char *N, const char *D, FunctionPassCtor C)
  : MachinePassRegistryNode<FunctionPassCtor>(N, D, C)
  { Registry.Add(this); }
  ~RegisterRegAlloc() { Registry.Remove(this); }
  

  // Accessors.
  //
  RegisterRegAlloc *getNext() const {
    return (RegisterRegAlloc *)
           MachinePassRegistryNode<FunctionPassCtor>::getNext();
  }
  static RegisterRegAlloc *getList() {
    return (RegisterRegAlloc *)Registry.getList();
  }
  static FunctionPassCtor getDefault() {
    return Registry.getDefault();
  }
  static void setDefault(FunctionPassCtor C) {
    Registry.setDefault(C);
  }
  static void setListener(MachinePassRegistryListener *L) {
    Registry.setListener(L);
  }


  /// FirstCtor - Finds the first register allocator in registration
  /// list and returns its creator function, otherwise return NULL.
  static FunctionPassCtor FirstCtor() {
    MachinePassRegistryNode<FunctionPassCtor> *Node = Registry.getList();
    return Node ? Node->getCtor() : NULL;
  }
  
  /// FindCtor - Finds a register allocator in registration list and returns
  /// its creator function, otherwise return NULL.
  static FunctionPassCtor FindCtor(const char *N) {
    MachinePassRegistryNode<FunctionPassCtor> *Node = Registry.Find(N);
    return Node ? Node->getCtor() : NULL;
  }
  
};


//===----------------------------------------------------------------------===//
///
/// RegisterScheduler class - Track the registration of instruction schedulers.
///
//===----------------------------------------------------------------------===//

class SelectionDAGISel;
class ScheduleDAG;
class SelectionDAG;
class MachineBasicBlock;

class RegisterScheduler : public
  MachinePassRegistryNode<
       ScheduleDAG *(*)(SelectionDAGISel*, SelectionDAG*, MachineBasicBlock*)> {

public:

  typedef ScheduleDAG *(*FunctionPassCtor)(SelectionDAGISel*, SelectionDAG*,
                                           MachineBasicBlock*);

  static MachinePassRegistry<FunctionPassCtor> Registry;

  RegisterScheduler(const char *N, const char *D, FunctionPassCtor C)
  : MachinePassRegistryNode<FunctionPassCtor>(N, D, C)
  { Registry.Add(this); }
  ~RegisterScheduler() { Registry.Remove(this); }


  // Accessors.
  //
  RegisterScheduler *getNext() const {
    return (RegisterScheduler *)
           MachinePassRegistryNode<FunctionPassCtor>::getNext();
  }
  static RegisterScheduler *getList() {
    return (RegisterScheduler *)Registry.getList();
  }
  static FunctionPassCtor getDefault() {
    return Registry.getDefault();
  }
  static void setDefault(FunctionPassCtor C) {
    Registry.setDefault(C);
  }
  static void setListener(MachinePassRegistryListener *L) {
    Registry.setListener(L);
  }


  /// FirstCtor - Finds the first instruction scheduler in registration
  /// list and returns its creator function, otherwise return NULL.
  static FunctionPassCtor FirstCtor() {
    MachinePassRegistryNode<FunctionPassCtor> *Node = Registry.getList();
    return Node ? Node->getCtor() : NULL;
  }
  
  
  /// FindCtor - Finds a instruction scheduler in registration list and returns
  /// its creator function, otherwise return NULL.
  static FunctionPassCtor FindCtor(const char *N) {
    MachinePassRegistryNode<FunctionPassCtor> *Node = Registry.Find(N);
    return Node ? Node->getCtor() : NULL;
  }
  
};


//===----------------------------------------------------------------------===//
///
/// RegisterPassParser class - Handle the addition of new machine passes.
///
//===----------------------------------------------------------------------===//
template<class RegistryClass>
class RegisterPassParser : public MachinePassRegistryListener,
                           public cl::parser<const char *> {
public:
  RegisterPassParser() {}
  ~RegisterPassParser() { RegistryClass::setListener(NULL); }

  void initialize(cl::Option &O) {
    cl::parser<const char *>::initialize(O);
    
    // Add existing passes to option.
    for (RegistryClass *Node = RegistryClass::getList();
         Node; Node = Node->getNext()) {
      addLiteralOption(Node->getName(), Node->getName(),
                       Node->getDescription());
    }
    
    // Make sure we listen for list changes.
    RegistryClass::setListener(this);
  }

  // Implement the MachinePassRegistryListener callbacks.
  //
  virtual void NotifyAdd(const char *N, const char *D) {
    addLiteralOption(N, N, D);
  }
  virtual void NotifyRemove(const char *N, const char *D) {
    removeLiteralOption(N);
  }

  // ValLessThan - Provide a sorting comparator for Values elements...
  typedef std::pair<const char*, std::pair<const char*, const char*> > ValType;
  static bool ValLessThan(const ValType &VT1, const ValType &VT2) {
    return std::string(VT1.first) < std::string(VT2.first);
  }

  // printOptionInfo - Print out information about this option.  Override the
  // default implementation to sort the table before we print...
  virtual void printOptionInfo(const cl::Option &O, unsigned GlobalWidth) const{
    RegisterPassParser *PNP = const_cast<RegisterPassParser*>(this);
    std::sort(PNP->Values.begin(), PNP->Values.end(), ValLessThan);
    cl::parser<const char *>::printOptionInfo(O, GlobalWidth);
  }
};


} // end namespace llvm

#endif
