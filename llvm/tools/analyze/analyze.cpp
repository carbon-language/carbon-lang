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
#include "llvm/Tools/CommandLine.h"
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
      WriteAsOperand(cout, (Value*)R.Scale) << " *";
      // fall through
    case analysis::ExprType::Linear:
      WriteAsOperand(cout, R.Var);
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

struct {
  const string ArgName, Name;
  void (*AnPtr)(Method *M);
} AnTable[] = {
  { "-print"          , "Print each Method"       , PrintMethod },
  { "-intervals"      , "Interval Partition"      , PrintIntervalPartition },
  { "-exprclassify"   , "Classify Expressions"    , PrintClassifiedExprs },

  { "-domset"         , "Dominator Sets"          , PrintDominatorSets },
  { "-idom"           , "Immediate Dominators"    , PrintImmediateDominators },
  { "-domtree"        , "Dominator Tree"          , PrintDominatorTree },
  { "-domfrontier"    , "Dominance Frontier"      , PrintDominanceFrontier },

  { "-postdomset"     , "Postdominator Sets"      , PrintPostDominatorSets },
  { "-postidom"       , "Immediate Postdominators", PrintImmediatePostDoms },
  { "-postdomtree"    , "Post Dominator Tree"     , PrintPostDomTree },
  { "-postdomfrontier", "Postdominance Frontier"  , PrintPostDomFrontier },
};

int main(int argc, char **argv) {
  ToolCommandLine Options(argc, argv, false);
  bool Quiet = false;

  for (int i = 1; i < argc; i++) {
    if (string(argv[i]) == string("--help")) {
      cerr << argv[0] << " usage:\n"
           << "  " << argv[0] << " --help\t - Print this usage information\n"
	   << "\t  --quiet\t - Do not print analysis name before output\n";
      for (unsigned j = 0; j < sizeof(AnTable)/sizeof(AnTable[0]); ++j) {
	cerr << "\t   " << AnTable[j].ArgName << "\t - Print " 
	     << AnTable[j].Name << endl;
      }
      return 1;
    } else if (string(argv[i]) == string("-q") ||
	       string(argv[i]) == string("--quiet")) {
      Quiet = true; argv[i] = 0;
    }
  }
  
  Module *C = ParseBytecodeFile(Options.getInputFilename());
  if (!C && !(C = ParseAssemblyFile(Options))) {
    cerr << "Input file didn't read correctly.\n";
    return 1;
  }

  // Loop over all of the methods in the module...
  for (Module::iterator I = C->begin(), E = C->end(); I != E; ++I) {
    Method *M = *I;
    if (M->isExternal()) continue;

    // Loop over all of the analyses to be run...
    for (int i = 1; i < argc; i++) {
      if (argv[i] == 0) continue;
      unsigned j;
      for (j = 0; j < sizeof(AnTable)/sizeof(AnTable[0]); j++) {
	if (string(argv[i]) == AnTable[j].ArgName) {
	  if (!Quiet)
	    cerr << "Running: " << AnTable[j].Name << " analysis on '"
		 << ((Value*)M)->getName() << "'!\n";
	  AnTable[j].AnPtr(M);
	  break;
	}
      }
      
      if (j == sizeof(AnTable)/sizeof(AnTable[0])) 
	cerr << "'" << argv[i] << "' argument unrecognized: ignored\n";
    }
  }

  delete C;
  return 0;
}
