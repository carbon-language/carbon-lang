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
#include "llvm/Analysis/InstForest.h"
#include "llvm/Analysis/Expressions.h"
#include "llvm/Analysis/InductionVariable.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/PassNameParser.h"
#include <algorithm>

using std::ostream;

#if 0

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
          IV.print(std::cout);
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
#endif

struct ModulePassPrinter : public Pass {
  Pass *PassToPrint;
  ModulePassPrinter(Pass *PI) : PassToPrint(PI) {}

  virtual bool run(Module &M) {
    std::cout << "Printing Analysis info for Pass "
              << PassToPrint->getPassName() << ":\n";
    PassToPrint->print(std::cout, &M);
    
    // Get and print pass...
    return false;
  }
};

struct FunctionPassPrinter : public FunctionPass {
  const PassInfo *PassToPrint;
  FunctionPassPrinter(const PassInfo *PI) : PassToPrint(PI) {}

  virtual bool runOnFunction(Function &F) {
    std::cout << "Printing Analysis info for function '" << F.getName()
              << "': Pass " << PassToPrint->getPassName() << ":\n";
    getAnalysis<Pass>(PassToPrint).print(std::cout, F.getParent());

    // Get and print pass...
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired(PassToPrint);
    AU.setPreservesAll();
  }
};

struct BasicBlockPassPrinter : public BasicBlockPass {
  const PassInfo *PassToPrint;
  BasicBlockPassPrinter(const PassInfo *PI) : PassToPrint(PI) {}

  virtual bool runOnBasicBlock(BasicBlock &BB) {
    std::cout << "Printing Analysis info for BasicBlock '" << BB.getName()
              << "': Pass " << PassToPrint->getPassName() << ":\n";
    getAnalysis<Pass>(PassToPrint).print(std::cout, BB.getParent()->getParent());

    // Get and print pass...
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired(PassToPrint);
    AU.setPreservesAll();
  }
};




static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"),
              cl::value_desc("filename"));

static cl::opt<bool> Quiet("q", cl::desc("Don't print analysis pass names"));
static cl::alias    QuietA("quiet", cl::desc("Alias for -q"),
                           cl::aliasopt(Quiet));

// The AnalysesList is automatically populated with registered Passes by the
// PassNameParser.
//
static cl::list<const PassInfo*, bool,
                FilteredPassNameParser<PassInfo::Analysis> >
AnalysesList(cl::desc("Analyses available:"));


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
  PassManager Passes;

  // Create a new optimization pass for each one specified on the command line
  for (unsigned i = 0; i < AnalysesList.size(); ++i) {
    const PassInfo *Analysis = AnalysesList[i];
    
    if (Analysis->getNormalCtor()) {
      Pass *P = Analysis->getNormalCtor()();
      Passes.add(P);

      if (BasicBlockPass *BBP = dynamic_cast<BasicBlockPass*>(P))
        Passes.add(new BasicBlockPassPrinter(Analysis));
      else if (FunctionPass *FP = dynamic_cast<FunctionPass*>(P))
        Passes.add(new FunctionPassPrinter(Analysis));
      else
        Passes.add(new ModulePassPrinter(P));

    } else
      cerr << "Cannot create pass: " << Analysis->getPassName() << "\n";
  }

  Passes.run(*CurMod);

  delete CurMod;
  return 0;
}
