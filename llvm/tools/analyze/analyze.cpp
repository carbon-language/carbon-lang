//===------------------------------------------------------------------------===
// LLVM 'Analyze' UTILITY 
//
// This utility is designed to print out the results of running various analysis
// passes on a program.  This is useful for understanding a program, or for 
// debugging an analysis pass.
//
//  analyze --help           - Output information about command line switches
//  analyze --quiet          - Do not print analysis name before output
//
//===------------------------------------------------------------------------===

#include <iostream>
#include "llvm/Instruction.h"
#include "llvm/Module.h"
#include "llvm/Method.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Analysis/Writer.h"

#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/IntervalPartition.h"
#include "llvm/Analysis/Expressions.h"

static void PrintMethod(Method *M) {
  cout << M;
}

static void PrintIntervalPartition(Method *M) {
  cout << cfg::IntervalPartition(M);
}

static void PrintClassifiedExprs(Method *M) {
  cout << "Classified expressions for: " << M->getName() << endl;
  Method::inst_iterator I = M->inst_begin(), E = M->inst_end();
  for (; I != E; ++I) {
    cout << *I;

    if ((*I)->getType() == Type::VoidTy) continue;
    analysis::ExprType R = analysis::ClassifyExpression(*I);
    if (R.Var == *I) continue;  // Doesn't tell us anything

    cout << "\t\tExpr =";
    switch (R.ExprTy) {
    case analysis::ExprType::ScaledLinear:
      WriteAsOperand(cout << "(", (Value*)R.Scale) << " ) *";
      // fall through
    case analysis::ExprType::Linear:
      WriteAsOperand(cout << "(", R.Var) << " )";
      if (R.Offset == 0) break;
      else cout << " +";
      // fall through
    case analysis::ExprType::Constant:
      if (R.Offset) WriteAsOperand(cout, (Value*)R.Offset); else cout << " 0";
      break;
    }
    cout << endl << endl;
  }
}


static void PrintDominatorSets(Method *M) {
  cout << cfg::DominatorSet(M);
}
static void PrintImmediateDominators(Method *M) {
  cout << cfg::ImmediateDominators(M);
}
static void PrintDominatorTree(Method *M) {
  cout << cfg::DominatorTree(M);
}
static void PrintDominanceFrontier(Method *M) {
  cout << cfg::DominanceFrontier(M);
}

static void PrintPostDominatorSets(Method *M) {
  cout << cfg::DominatorSet(M, true);
}
static void PrintImmediatePostDoms(Method *M) {
  cout << cfg::ImmediateDominators(cfg::DominatorSet(M, true));
}
static void PrintPostDomTree(Method *M) {
  cout << cfg::DominatorTree(cfg::DominatorSet(M, true));
}
static void PrintPostDomFrontier(Method *M) {
  cout << cfg::DominanceFrontier(cfg::DominatorSet(M, true));
}

enum Ans {
  print, intervals, exprclassify,
  domset, idom, domtree, domfrontier,
  postdomset, postidom, postdomtree, postdomfrontier,
};

cl::String InputFilename ("", "Load <arg> file to analyze", cl::NoFlags, "-");
cl::Flag   Quiet         ("q", "Don't print analysis pass names");
cl::Alias  QuietA        ("quiet", "Alias for -q", cl::NoFlags, Quiet);
cl::EnumList<enum Ans> AnalysesList(cl::NoFlags,
  clEnumVal(print          , "Print each Method"),
  clEnumVal(intervals      , "Print Interval Partitions"),
  clEnumVal(exprclassify   , "Classify Expressions"),

  clEnumVal(domset         , "Print Dominator Sets"),
  clEnumVal(idom           , "Print Immediate Dominators"),
  clEnumVal(domtree        , "Print Dominator Tree"),
  clEnumVal(domfrontier    , "Print Dominance Frontier"),

  clEnumVal(postdomset     , "Print Postdominator Sets"),
  clEnumVal(postidom       , "Print Immediate Postdominators"),
  clEnumVal(postdomtree    , "Print Post Dominator Tree"),
  clEnumVal(postdomfrontier, "Print Postdominance Frontier"),
0);

struct {
  enum Ans AnID;
  void (*AnPtr)(Method *M);
} AnTable[] = {
  { print          , PrintMethod              },
  { intervals      , PrintIntervalPartition   },
  { exprclassify   , PrintClassifiedExprs     },

  { domset         , PrintDominatorSets       },
  { idom           , PrintImmediateDominators },
  { domtree        , PrintDominatorTree       },
  { domfrontier    , PrintDominanceFrontier   },

  { postdomset     , PrintPostDominatorSets   },
  { postidom       , PrintImmediatePostDoms   },
  { postdomtree    , PrintPostDomTree         },
  { postdomfrontier, PrintPostDomFrontier     },
};

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm analysis printer tool\n");

  Module *C = ParseBytecodeFile(InputFilename);
  if (!C && !(C = ParseAssemblyFile(InputFilename))) {
    cerr << "Input file didn't read correctly.\n";
    return 1;
  }

  // Loop over all of the methods in the module...
  for (Module::iterator I = C->begin(), E = C->end(); I != E; ++I) {
    Method *M = *I;
    if (M->isExternal()) continue;

    for (unsigned i = 0; i < AnalysesList.size(); ++i) {
      enum Ans AnalysisPass = AnalysesList[i];

      // Loop over all of the analyses to be run...
      unsigned j;
      for (j = 0; j < sizeof(AnTable)/sizeof(AnTable[0]); ++j) {
	if (AnalysisPass == AnTable[j].AnID) {
	  if (!Quiet)
	    cerr << "Running: " << AnalysesList.getArgDescription(AnalysisPass) 
		 << " analysis on '" << ((Value*)M)->getName() << "'!\n";
	  AnTable[j].AnPtr(M);
	  break;
	}
      }
      if (j == sizeof(AnTable)/sizeof(AnTable[0])) 
	cerr << "Analysis tables inconsistent!\n";
    }
  }

  delete C;
  return 0;
}
