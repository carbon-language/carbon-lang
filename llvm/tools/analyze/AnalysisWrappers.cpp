//===- AnalysisWrappers.cpp - Wrappers around non-pass analyses -----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines pass wrappers around LLVM analyses that don't make sense to
// be passes.  It provides a nice standard pass interface to these classes so
// that they can be printed out by analyze.
//
// These classes are separated out of analyze.cpp so that it is more clear which
// code is the integral part of the analyze tool, and which part of the code is
// just making it so more passes are available.
//
//===----------------------------------------------------------------------===//

#include "llvm/iPHINode.h"
#include "llvm/Type.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Analysis/InstForest.h"
#include "llvm/Analysis/Expressions.h"
#include "llvm/Analysis/InductionVariable.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/InstIterator.h"

using namespace llvm;

namespace {
  struct InstForestHelper : public FunctionPass {
    Function *F;
    virtual bool runOnFunction(Function &Func) { F = &Func; return false; }

    void print(std::ostream &OS) const {
      std::cout << InstForest<char>(F);
    }
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };

  RegisterAnalysis<InstForestHelper> P1("instforest", "InstForest Printer");

  struct IndVars : public FunctionPass {
    Function *F;
    LoopInfo *LI;
    virtual bool runOnFunction(Function &Func) {
      F = &Func; LI = &getAnalysis<LoopInfo>();
      return false;
    }

    void print(std::ostream &OS) const {
      for (inst_iterator I = inst_begin(*F), E = inst_end(*F); I != E; ++I)
        if (PHINode *PN = dyn_cast<PHINode>(*I)) {
          InductionVariable IV(PN, LI);
          if (IV.InductionType != InductionVariable::Unknown)
            IV.print(OS);
        }
    }
    
    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LoopInfo>();
      AU.setPreservesAll();
    }
  };

  RegisterAnalysis<IndVars> P6("indvars", "Induction Variable Analysis");


  struct Exprs : public FunctionPass {
    Function *F;
    virtual bool runOnFunction(Function &Func) { F = &Func; return false; }

    void print(std::ostream &OS) const {
      OS << "Classified expressions for: " << F->getName() << "\n";
      for (inst_iterator I = inst_begin(*F), E = inst_end(*F); I != E; ++I) {
        OS << *I;
      
        if ((*I)->getType() == Type::VoidTy) continue;
        ExprType R = ClassifyExpression(*I);
        if (R.Var == *I) continue;  // Doesn't tell us anything
      
        OS << "\t\tExpr =";
        switch (R.ExprTy) {
        case ExprType::ScaledLinear:
          WriteAsOperand(OS << "(", (Value*)R.Scale) << " ) *";
          // fall through
        case ExprType::Linear:
          WriteAsOperand(OS << "(", R.Var) << " )";
          if (R.Offset == 0) break;
          else OS << " +";
          // fall through
        case ExprType::Constant:
          if (R.Offset) WriteAsOperand(OS, (Value*)R.Offset);
          else OS << " 0";
          break;
        }
        OS << "\n\n";
      }
    }
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };

  RegisterAnalysis<Exprs> P7("exprs", "Expression Printer");
}
