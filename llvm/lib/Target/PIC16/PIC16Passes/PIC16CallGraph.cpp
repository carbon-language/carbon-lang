#include "llvm/Analysis/CallGraph.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>
using namespace llvm;
using std::vector;
using std::string;

namespace {
  class PIC16CallGraph : public ModulePass { 
  public:
    static char ID; // Class identification 
    PIC16CallGraph() : ModulePass(&ID)  {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      AU.addRequired<CallGraph>();
    }
    virtual bool runOnModule(Module &M) {
      // Initially record that no interrupt has been found
      InterruptFound = false;

      CallGraph &CG = getAnalysis<CallGraph>();
      for (CallGraph::iterator it = CG.begin() ; it != CG.end(); it++)
      {
        // External calling node doesn't have any function associated 
        // with it
        if (!it->first)
          continue;

        if (it->first->getName().str() == "main") {
          // See if the main itself is interrupt function then report an error.
          if (it->first->getSection().find("interrupt") != string::npos)
             reportError("Function 'main' can't be interrupt function");
          else  { 
            // Set the MainLine tag for function main also. 
            it->second->getFunction()->setSection("ML");
            // mark the hierarchy 
            markFunctionHierarchy(it->second, "ML");
          }
        } else if (it->first->getSection().find("interrupt") != string::npos) {
          if (InterruptFound)
            reportError("More than one interrupt functions defined in the module"); 

          InterruptFound = true;
          markFunctionHierarchy(it->second, "IL");
        }
      }
     return false;
    }
  private: // Functions
    // Makr function hierarchy for the MainLine or InterruptLine.
    void markFunctionHierarchy(CallGraphNode *CGN, string StringMark);

    // Error reporting for PIC16Pass
    void reportError(string ErrorString, vector<string> &Values);
    void reportError(string ErrorString);
  private: // Data
    // Records if the interrupt function has already been found.
    // If more than one interrupt function is found then an error
    // should be thrown.
    bool InterruptFound; 
  };
  char PIC16CallGraph::ID =0;
  static RegisterPass<PIC16CallGraph>
  Y("pic16cg", "PIC16 CallGraph Construction");

}  // End of anonymous namespace

void PIC16CallGraph::reportError(string ErrorString) {
  errs() << "ERROR : " << ErrorString << "\n";
  exit(1);
}

void PIC16CallGraph::
reportError (string ErrorString, vector<string> &Values) {
  unsigned ValCount = Values.size();
  string TargetString;
  for (unsigned i=0; i<ValCount; ++i) {
    TargetString = "%";
    TargetString += ((char)i + '0');
    ErrorString.replace(ErrorString.find(TargetString), TargetString.length(), Values[i]);
  }
  errs() << "ERROR : " << ErrorString << "\n";
  exit(1);
  //llvm_report_error(ErrorString);
}

void PIC16CallGraph::
markFunctionHierarchy(CallGraphNode *CGN, string StringMark) {
  string AlternateMark;
  string SharedMark = "SL";
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
     if (CalledF->getSection().find("interrupt") != string::npos) {
       vector<string> Values;
       Values.push_back(CalledF->getName().str());
       reportError("Interrupt function (%0) can't be called", Values); 
     }

     // If already shared mark then no need to check any further.
     // Also a great potential for recursion.
     if (CalledF->getSection().find(SharedMark) != string::npos) {
       continue;
     }

     // Has already been mark 
     if (CalledF->getSection().find(StringMark) != string::npos) {
       // Issue diagnostic
       // Potential for recursion.
     } else {
       // Mark now
       if (CalledF->getSection().find(AlternateMark) != string::npos) {
         // Function is alternatively marked. It should be a shared one.
        
         // Shared functions should be clone. Clone here. 
         Function *ClonedFunc = CloneFunction(CalledF); 

         // Add the newly created function to the module.
         CalledF->getParent()->getFunctionList().push_back(ClonedFunc);

         // The new function should be for interrupt line. Therefore should have the
         // name suffixed with IL and section attribute marked with IL. 
         ClonedFunc->setName(CalledF->getName().str() + ".IL");
         ClonedFunc->setSection("IL");

         // Original function now should be for MainLine only. 
         CalledF->setSection("ML");

         // Update the CallSite 
         CallSite CS = cgn_it->first;
         CS.getInstruction()->getOperand(0)->setName(CalledF->getName().str() + ".shared");
       } else {
         // Function is not marked. It should be marked now.
         CalledF->setSection(StringMark);
       }
     }
     
     // Before going any further mark all the called function by current
     // function.
     markFunctionHierarchy(cgn_it->second ,StringMark);
  } // end of loop of all called functions.

}
