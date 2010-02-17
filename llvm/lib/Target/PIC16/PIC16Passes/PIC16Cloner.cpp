//===-- PIC16Cloner.cpp - PIC16 LLVM Cloner for shared functions -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains code to clone all functions that are shared between
// the main line code (ML) and interrupt line code (IL). It clones all such
// shared functions and their automatic global vars by adding the .IL suffix. 
//
// This pass is supposed to be run on the linked .bc module.
// It traveses the module call graph twice. Once starting from the main function
// and marking each reached function as "ML". Again, starting from the ISR
// and cloning any reachable function that was marked as "ML". After cloning
// the function, it remaps all the call sites in IL functions to call the
// cloned functions. 
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CallGraph.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "PIC16Cloner.h"
#include "../PIC16ABINames.h"
#include <vector>

using namespace llvm;
using std::vector;
using std::string;
using std::map;

namespace llvm {
  char PIC16Cloner::ID = 0;

  ModulePass *createPIC16ClonerPass() { return new PIC16Cloner(); }
}

// We currently intend to run these passes in opt, which does not have any
// diagnostic support. So use these functions for now. In future
// we will probably write our own driver tool.
//
void PIC16Cloner::reportError(string ErrorString) {
  errs() << "ERROR : " << ErrorString << "\n";
  exit(1);
}

void PIC16Cloner::
reportError (string ErrorString, vector<string> &Values) {
  unsigned ValCount = Values.size();
  string TargetString;
  for (unsigned i=0; i<ValCount; ++i) {
    TargetString = "%";
    TargetString += ((char)i + '0');
    ErrorString.replace(ErrorString.find(TargetString), TargetString.length(), 
                        Values[i]);
  }
  errs() << "ERROR : " << ErrorString << "\n";
  exit(1);
}


// Entry point
//
bool PIC16Cloner::runOnModule(Module &M) {
   CallGraph &CG = getAnalysis<CallGraph>();
   // Initially record that no interrupt has been found
   foundISR = false;

   // First mark the MainLine. 
   for (CallGraph::iterator it = CG.begin() ; it != CG.end(); it++)
   {
     // External calling node doesn't have any function associated 
     // with it
     if (!it->first)
       continue;
     
     if (it->first->getName().str() == "main") {
       // See if the main itself is interrupt function then report an error.
       if (PAN::isISR(it->first->getSection()))
          reportError("Function 'main' can't be interrupt function");
       else  { 
         // Function main itself is MainLine function.
         it->second->getFunction()->setSection("ML");
         // mark the hierarchy 
         markCallGraph(it->second, "ML");
         // MainLine has been marked now break ; don't search any further.
         break;
       }
     } 
   }

   // When the MainLine has been marked only then mark the InterruptLine.
   for (CallGraph::iterator it = CG.begin() ; it != CG.end(); it++)
   {
     // External calling node doesn't have any function associated 
     // with it
     if (!it->first)
       continue;

     if (PAN::isISR(it->first->getSection())) {

       if (foundISR)
         reportError("More than one interrupt functions defined in the module");

       foundISR = true;
       
       markCallGraph(it->second, "IL");
       // InterruptLine handled now break; don't search any further.
       break;
     }
   } 
   
  return true;
}

// For PIC16, automatic variables of a function are emitted as globals.
// Clone the auto variables of a function  and put them in ValueMap, 
// this ValueMap will be used while
// Cloning the code of function itself.
//
void PIC16Cloner::CloneAutos(Function *F) {
  // We'll need to update module's globals list as well. So keep a reference
  // handy.
  Module *M = F->getParent();
  Module::GlobalListType &Globals = M->getGlobalList();

  // Clear the leftovers in ValueMap by any previous cloning.
  ValueMap.clear();

  // Find the auto globls for this function and clone them, and put them
  // in ValueMap.
  std::string FnName = F->getName().str();
  std::string VarName, ClonedVarName;
  for (Module::global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I) {
    VarName = I->getName().str();
    if (PAN::isLocalToFunc(FnName, VarName)) {
      // Auto variable for current function found. Clone it.
      GlobalVariable *GV = I;

      const Type *InitTy = GV->getInitializer()->getType();
      GlobalVariable *ClonedGV = 
        new GlobalVariable(InitTy, false, GV->getLinkage(), 
                           GV->getInitializer());
      ClonedGV->setName(PAN::getCloneVarName(FnName, VarName));
      // Add these new globals to module's globals list.
      Globals.push_back(ClonedGV);
 
      // Update ValueMap.
      ValueMap[GV] = ClonedGV;
     }
  }
}


// Mark all reachable functions from the given node, with the given mark.
//
void PIC16Cloner::markCallGraph(CallGraphNode *CGN, string StringMark) {
  string AlternateMark;
  if (StringMark == "ML")
    AlternateMark = "IL";
  else
    AlternateMark = "ML";

  // Mark all the called functions
  for(CallGraphNode::iterator cgn_it = CGN->begin(); 
              cgn_it != CGN->end(); ++cgn_it) {
     Function *CalledF = cgn_it->second->getFunction();

     // If calling an external function then CallGraphNode
     // will not be associated with any function.
     if (!CalledF)
       continue;
  
     // Issue diagnostic if interrupt function is being called.
     if (PAN::isISR(CalledF->getSection())) {
       vector<string> Values;
       Values.push_back(CalledF->getName().str());
       reportError("Interrupt function (%0) can't be called", Values); 
     }

     // Has already been mark 
     if (CalledF->getSection().find(StringMark) != string::npos) {
        // Should we do anything here?
     } else {
       // Mark now
       if (CalledF->getSection().find(AlternateMark) != string::npos) {
         // Function is alternatively marked. It should be a shared one.
         // Create IL copy. Passing called function as first argument
         // and the caller as the second argument.

         // Before making IL copy, first ensure that this function has a 
         // body. If the function does have a body. It can't be cloned.
         // Such a case may occur when the function has been declarated
         // in the C source code but its body exists in assembly file.
         if (!CalledF->isDeclaration()) {
           // FIXME: Not implemented yet. Clone the function here.
         }
       } else {
         // Function is not marked. It should be marked now.
         CalledF->setSection(StringMark);
       }
     }
     // Before going any further mark all the called function by current
     // function.
     markCallGraph(cgn_it->second ,StringMark);
  } // end of loop of all called functions.
}

// Clone the given function and return it.
// Note: it uses the ValueMap member of the class, which is already populated
// by cloneAutos by the time we reach here. 
// FIXME: Should we just pass ValueMap's ref as a parameter here? rather
// than keeping the ValueMap as a member.
Function *
PIC16Cloner::cloneFunction(Function *OrgF) {
   Function *ClonedF;

   // See if we already cloned it. Return that. 
   cloned_map_iterator cm_it = ClonedFunctionMap.find(OrgF);
   if(cm_it != ClonedFunctionMap.end()) {
     ClonedF = cm_it->second;
     return ClonedF;
   }

   // Clone does not exist. 
   // First clone the autos, and populate ValueMap.
   CloneAutos(OrgF);

   // Now create the clone.
   ClonedF = CloneFunction(OrgF, ValueMap);

   // The new function should be for interrupt line. Therefore should have 
   // the name suffixed with IL and section attribute marked with IL. 
   ClonedF->setName(PAN::getCloneFnName(OrgF->getName()));
   ClonedF->setSection("IL");

   // Add the newly created function to the module.
   OrgF->getParent()->getFunctionList().push_back(ClonedF);

   // Update the ClonedFunctionMap to record this cloning activity.
   ClonedFunctionMap[OrgF] = ClonedF;

   return ClonedF;
}


