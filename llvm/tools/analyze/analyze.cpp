//===----------------------------------------------------------------------===//
// The LLVM analyze utility
//
// This utility is designed to print out the results of running various analysis
// passes on a program.  This is useful for understanding a program, or for 
// debugging an analysis pass.
//
//  analyze --help           - Output information about command line switches
//  analyze --quiet          - Do not print analysis name before output
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/iPHINode.h"
#include "llvm/Type.h"
#include "llvm/PassManager.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Analysis/Writer.h"
#include "llvm/Analysis/InstForest.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/IntervalPartition.h"
#include "llvm/Analysis/Expressions.h"
#include "llvm/Analysis/InductionVariable.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/DataStructure.h"
#include "llvm/Analysis/FindUnsafePointerTypes.h"
#include "llvm/Analysis/FindUsedTypes.h"
#include "llvm/Support/InstIterator.h"
#include "Support/CommandLine.h"
#include <algorithm>

using std::ostream;
using std::string;

//===----------------------------------------------------------------------===//
// printPass - Specify how to print out a pass.  For most passes, the standard
// way of using operator<< works great, so we use it directly...
//
template<class PassType>
static void printPass(PassType &P, ostream &O, Module &M) {
  O << P;
}

template<class PassType>
static void printPass(PassType &P, ostream &O, Function &F) {
  O << P;
}

// Other classes require more information to print out information, so we
// specialize the template here for them...
//
template<>
static void printPass(LocalDataStructures &P, ostream &O, Module &M) {
  P.print(O, &M);
}
template<>
static void printPass(BUDataStructures &P, ostream &O, Module &M) {
  P.print(O, &M);
}

template<>
static void printPass(FindUsedTypes &FUT, ostream &O, Module &M) {
  FUT.printTypes(O, &M);
}

template<>
static void printPass(FindUnsafePointerTypes &FUPT, ostream &O, Module &M) {
  FUPT.printResults(&M, O);
}



template <class PassType, class PassName>
class PassPrinter;  // Do not implement

template <class PassName>
class PassPrinter<Pass, PassName> : public Pass {
  const AnalysisID ID;
public:
  PassPrinter(AnalysisID id) : ID(id) {}

  const char *getPassName() const { return "IP Pass Printer"; }
  
  virtual bool run(Module &M) {
    printPass(getAnalysis<PassName>(ID), std::cout, M);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired(ID);
  }
};

template <class PassName>
class PassPrinter<FunctionPass, PassName> : public FunctionPass {
  const AnalysisID ID;
public:
  PassPrinter(AnalysisID id) : ID(id) {}

    const char *getPassName() const { return "Function Pass Printer"; }
  
  virtual bool runOnFunction(Function &F) {
    std::cout << "Running on function '" << F.getName() << "'\n";
    printPass(getAnalysis<PassName>(ID), std::cout, F);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired(ID);
    AU.setPreservesAll();
  }
};



template <class PassType, class PassName, AnalysisID &ID>
Pass *New() {
  return new PassPrinter<PassType, PassName>(ID);
}
template <class PassType, class PassName>
Pass *New() {
  return new PassPrinter<PassType, PassName>(PassName::ID);
}


Pass *createPrintFunctionPass() {
  return new PrintFunctionPass("", &std::cout);
}
Pass *createPrintModulePass() {
  return new PrintModulePass(&std::cout);
}

struct InstForestHelper : public FunctionPass {
  const char *getPassName() const { return "InstForest Printer"; }

  void doit(Function &F) {
    std::cout << InstForest<char>(&F);
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
};

struct IndVars : public FunctionPass {
  const char *getPassName() const { return "IndVars Printer"; }

  void doit(Function &F) {
    LoopInfo &LI = getAnalysis<LoopInfo>();
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I)
      if (PHINode *PN = dyn_cast<PHINode>(*I)) {
        InductionVariable IV(PN, &LI);
        if (IV.InductionType != InductionVariable::Unknown)
          std::cout << IV;
      }
  }

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired(LoopInfo::ID);
    AU.setPreservesAll();
  }
};

struct Exprs : public FunctionPass {
  const char *getPassName() const { return "Expression Printer"; }

  static void doit(Function &F) {
    std::cout << "Classified expressions for: " << F.getName() << "\n";
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      std::cout << *I;
      
      if ((*I)->getType() == Type::VoidTy) continue;
      analysis::ExprType R = analysis::ClassifyExpression(*I);
      if (R.Var == *I) continue;  // Doesn't tell us anything
      
      std::cout << "\t\tExpr =";
      switch (R.ExprTy) {
      case analysis::ExprType::ScaledLinear:
        WriteAsOperand(std::cout << "(", (Value*)R.Scale) << " ) *";
        // fall through
      case analysis::ExprType::Linear:
        WriteAsOperand(std::cout << "(", R.Var) << " )";
        if (R.Offset == 0) break;
        else std::cout << " +";
        // fall through
      case analysis::ExprType::Constant:
        if (R.Offset) WriteAsOperand(std::cout, (Value*)R.Offset);
        else std::cout << " 0";
        break;
      }
      std::cout << "\n\n";
    }
  }
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
};


template<class TraitClass>
struct PrinterPass : public TraitClass {
  PrinterPass() {}

  virtual bool runOnFunction(Function &F) {
    std::cout << "Running on function '" << F.getName() << "'\n";

    TraitClass::doit(F);
    return false;
  }
};


template<class PassClass>
Pass *Create() {
  return new PassClass();
}



enum Ans {
  // global analyses
  print, intervals, exprs, instforest, loops, indvars,

  // ip analyses
  printmodule, callgraph, datastructure, budatastructure,
  printusedtypes, unsafepointertypes,

  domset, idom, domtree, domfrontier,
  postdomset, postidom, postdomtree, postdomfrontier,
};

static cl::opt<string>
InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"),
              cl::value_desc("filename"));

static cl::opt<bool> Quiet("q", cl::desc("Don't print analysis pass names"));
static cl::alias    QuietA("quiet", cl::desc("Alias for -q"),
                           cl::aliasopt(Quiet));

static cl::list<enum Ans>
AnalysesList(cl::desc("Analyses available:"),
             cl::values(
  clEnumVal(print          , "Print each function"),
  clEnumVal(intervals      , "Print Interval Partitions"),
  clEnumVal(exprs          , "Classify Expressions"),
  clEnumVal(instforest     , "Print Instruction Forest"),
  clEnumVal(loops          , "Print natural loops"),
  clEnumVal(indvars        , "Print Induction Variables"),

  clEnumVal(printmodule    , "Print entire module"),
  clEnumVal(callgraph      , "Print Call Graph"),
  clEnumVal(datastructure  , "Print data structure information"),
  clEnumVal(budatastructure, "Print bottom-up data structure information"),
  clEnumVal(printusedtypes , "Print types used by module"),
  clEnumVal(unsafepointertypes, "Print unsafe pointer types"),

  clEnumVal(domset         , "Print Dominator Sets"),
  clEnumVal(idom           , "Print Immediate Dominators"),
  clEnumVal(domtree        , "Print Dominator Tree"),
  clEnumVal(domfrontier    , "Print Dominance Frontier"),

  clEnumVal(postdomset     , "Print Postdominator Sets"),
  clEnumVal(postidom       , "Print Immediate Postdominators"),
  clEnumVal(postdomtree    , "Print Post Dominator Tree"),
  clEnumVal(postdomfrontier, "Print Postdominance Frontier"),
0));


struct {
  enum Ans AnID;
  Pass *(*PassConstructor)();
} AnTable[] = {
  // Global analyses
  { print             , createPrintFunctionPass                 },
  { intervals         , New<FunctionPass, IntervalPartition>    },
  { loops             , New<FunctionPass, LoopInfo>             },
  { instforest        , Create<PrinterPass<InstForestHelper> >  },
  { indvars           , Create<PrinterPass<IndVars> >           },
  { exprs             , Create<PrinterPass<Exprs> >             },

  // IP Analyses...
  { printmodule       , createPrintModulePass             },
  { printusedtypes    , New<Pass, FindUsedTypes>          },
  { callgraph         , New<Pass, CallGraph>              },
  { datastructure     , New<Pass, LocalDataStructures>    },
  { budatastructure   , New<Pass, BUDataStructures>       },
  { unsafepointertypes, New<Pass, FindUnsafePointerTypes> },

  // Dominator analyses
  { domset            , New<FunctionPass, DominatorSet>        },
  { idom              , New<FunctionPass, ImmediateDominators> },
  { domtree           , New<FunctionPass, DominatorTree>       },
  { domfrontier       , New<FunctionPass, DominanceFrontier>   },

  { postdomset        , New<FunctionPass, DominatorSet, DominatorSet::PostDomID> },
  { postidom          , New<FunctionPass, ImmediateDominators, ImmediateDominators::PostDomID> },
  { postdomtree       , New<FunctionPass, DominatorTree, DominatorTree::PostDomID> },
  { postdomfrontier   , New<FunctionPass, DominanceFrontier, DominanceFrontier::PostDomID> },
};

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm analysis printer tool\n");

  Module *CurMod = 0;
  try {
    CurMod = ParseBytecodeFile(InputFilename);
    if (!CurMod && !(CurMod = ParseAssemblyFile(InputFilename))){
      std::cerr << "Input file didn't read correctly.\n";
      return 1;
    }
  } catch (const ParseException &E) {
    std::cerr << E.getMessage() << "\n";
    return 1;
  }

  // Create a PassManager to hold and optimize the collection of passes we are
  // about to build...
  //
  PassManager Analyses;

  // Loop over all of the analyses looking for analyses to run...
  for (unsigned i = 0; i < AnalysesList.size(); ++i) {
    enum Ans AnalysisPass = AnalysesList[i];

    for (unsigned j = 0; j < sizeof(AnTable)/sizeof(AnTable[0]); ++j) {
      if (AnTable[j].AnID == AnalysisPass) {
        Analyses.add(AnTable[j].PassConstructor());
        break;                       // get an error later
      }
    }
  }  

  Analyses.run(*CurMod);

  delete CurMod;
  return 0;
}
