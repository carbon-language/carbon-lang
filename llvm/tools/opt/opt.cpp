//===----------------------------------------------------------------------===//
// LLVM 'OPT' UTILITY 
//
// Optimizations may be specified an arbitrary number of times on the command
// line, they are run in the order specified.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Bytecode/WriteBytecodePass.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Transforms/ConstantMerge.h"
#include "llvm/Transforms/CleanupGCCOutput.h"
#include "llvm/Transforms/LevelChange.h"
#include "llvm/Transforms/FunctionInlining.h"
#include "llvm/Transforms/ChangeAllocations.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/IPO/SimpleStructMutation.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/PoolAllocate.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
#include "llvm/Transforms/Instrumentation/TraceValues.h"
#include "llvm/Transforms/Instrumentation/ProfilePaths.h"
#include "llvm/Target/TargetData.h"
#include "Support/CommandLine.h"
#include "Support/Signals.h"
#include <fstream>
#include <memory>

// FIXME: This should be parameterizable eventually for different target
// types...
static TargetData TD("opt target");

// Opts enum - All of the transformations we can do...
enum Opts {
  // Basic optimizations
  dce, die, constprop, gcse, licm, inlining, constmerge,
  strip, mstrip, mergereturn, simplifycfg,

  // Miscellaneous Transformations
  raiseallocs, lowerallocs, funcresolve, cleangcc, lowerrefs,

  // Printing and verifying...
  print, printm, verify,

  // More powerful optimizations
  indvars, instcombine, sccp, adce, raise, reassociate, mem2reg, pinodes,

  // Instrumentation
  trace, tracem, paths,

  // Interprocedural optimizations...
  internalize, globaldce, swapstructs, sortstructs, poolalloc,
};

static Pass *createPrintFunctionPass() {
  return new PrintFunctionPass("Current Function: \n", &cerr);
}

static Pass *createPrintModulePass() {
  return new PrintModulePass(&cerr);
}

static Pass *createLowerAllocationsPassNT() {
  return createLowerAllocationsPass(TD);
}

// OptTable - Correlate enum Opts to Pass constructors...
//
struct {
  enum Opts OptID;
  Pass * (*PassCtor)();
} OptTable[] = {
  { dce        , createDeadCodeEliminationPass    },
  { die        , createDeadInstEliminationPass    },
  { constprop  , createConstantPropogationPass    }, 
  { gcse       , createGCSEPass                   },
  { licm       , createLICMPass                   },
  { inlining   , createFunctionInliningPass       },
  { constmerge , createConstantMergePass          },
  { strip      , createSymbolStrippingPass        },
  { mstrip     , createFullSymbolStrippingPass    },
  { mergereturn, createUnifyFunctionExitNodesPass },
  { simplifycfg, createCFGSimplificationPass      },

  { indvars    , createIndVarSimplifyPass         },
  { instcombine, createInstructionCombiningPass   },
  { sccp       , createSCCPPass                   },
  { adce       , createAggressiveDCEPass          },
  { raise      , createRaisePointerReferencesPass },
  { reassociate, createReassociatePass            },
  { mem2reg    , createPromoteMemoryToRegister    },
  { pinodes    , createPiNodeInsertionPass        },
  { lowerrefs  , createDecomposeMultiDimRefsPass  },

  { trace      , createTraceValuesPassForBasicBlocks },
  { tracem     , createTraceValuesPassForFunction    },
  { paths      , createProfilePathsPass              },

  { print      , createPrintFunctionPass },
  { printm     , createPrintModulePass   },
  { verify     , createVerifierPass      },

  { raiseallocs, createRaiseAllocationsPass   },
  { lowerallocs, createLowerAllocationsPassNT },
  { cleangcc   , createCleanupGCCOutputPass   },
  { funcresolve, createFunctionResolvingPass  },

  { internalize, createInternalizePass  },
  { globaldce  , createGlobalDCEPass    },
  { swapstructs, createSwapElementsPass },
  { sortstructs, createSortElementsPass },
  { poolalloc  , createPoolAllocatePass },
};


// Command line option handling code...
//
cl::String InputFilename ("", "Load <arg> file to optimize", cl::NoFlags, "-");
cl::String OutputFilename("o", "Override output filename", cl::NoFlags, "");
cl::Flag   Force         ("f", "Overwrite output files", cl::NoFlags, false);
cl::Flag   PrintEachXForm("p", "Print module after each transformation");
cl::Flag   Quiet         ("q", "Don't print modifying pass names", 0, false);
cl::Alias  QuietA        ("quiet", "Alias for -q", cl::NoFlags, Quiet);
cl::EnumList<enum Opts> OptimizationList(cl::NoFlags,
  clEnumVal(dce        , "Dead Code Elimination"),
  clEnumVal(die        , "Dead Instruction Elimination"),
  clEnumVal(constprop  , "Simple constant propogation"),
  clEnumVal(gcse       , "Global Common Subexpression Elimination"),
  clEnumVal(licm       , "Loop Invariant Code Motion"),
 clEnumValN(inlining   , "inline", "Function integration"),
  clEnumVal(constmerge , "Merge identical global constants"),
  clEnumVal(strip      , "Strip symbols"),
  clEnumVal(mstrip     , "Strip module symbols"),
  clEnumVal(mergereturn, "Unify function exit nodes"),
  clEnumVal(simplifycfg, "CFG Simplification"),

  clEnumVal(indvars    , "Simplify Induction Variables"),
  clEnumVal(instcombine, "Combine redundant instructions"),
  clEnumVal(sccp       , "Sparse Conditional Constant Propogation"),
  clEnumVal(adce       , "Aggressive DCE"),
  clEnumVal(reassociate, "Reassociate expressions"),
  clEnumVal(mem2reg    , "Promote alloca locations to registers"),
  clEnumVal(pinodes    , "Insert Pi nodes after definitions"),

  clEnumVal(internalize, "Mark all fn's internal except for main"),
  clEnumVal(globaldce  , "Remove unreachable globals"),
  clEnumVal(swapstructs, "Swap structure types around"),
  clEnumVal(sortstructs, "Sort structure elements"),
  clEnumVal(poolalloc  , "Pool allocate disjoint datastructures"),

  clEnumVal(raiseallocs, "Raise allocations from calls to instructions"),
  clEnumVal(lowerallocs, "Lower allocations from instructions to calls (TD)"),
  clEnumVal(cleangcc   , "Cleanup GCC Output"),
  clEnumVal(funcresolve, "Resolve calls to foo(...) to foo(<concrete types>)"),
  clEnumVal(raise      , "Raise to Higher Level"),
  clEnumVal(trace      , "Insert BB and Function trace code"),
  clEnumVal(tracem     , "Insert Function trace code only"),
  clEnumVal(paths      , "Insert path profiling instrumentation"),
  clEnumVal(print      , "Print working function to stderr"),
  clEnumVal(printm     , "Print working module to stderr"),
  clEnumVal(verify     , "Verify module is well formed"),
  clEnumVal(lowerrefs  , "Decompose multi-dimensional structure/array refs to use one index per instruction"),
0);



int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv,
			      " llvm .bc -> .bc modular optimizer\n");

  // Load the input module...
  std::auto_ptr<Module> M(ParseBytecodeFile(InputFilename));
  if (M.get() == 0) {
    cerr << "bytecode didn't read correctly.\n";
    return 1;
  }

  // Figure out what stream we are supposed to write to...
  std::ostream *Out = &std::cout;  // Default to printing to stdout...
  if (OutputFilename != "") {
    if (!Force && std::ifstream(OutputFilename.c_str())) {
      // If force is not specified, make sure not to overwrite a file!
      cerr << "Error opening '" << OutputFilename << "': File exists!\n"
           << "Use -f command line argument to force output\n";
      return 1;
    }
    Out = new std::ofstream(OutputFilename.c_str());

    if (!Out->good()) {
      cerr << "Error opening " << OutputFilename << "!\n";
      return 1;
    }

    // Make sure that the Output file gets unlink'd from the disk if we get a
    // SIGINT
    RemoveFileOnSignal(OutputFilename);
  }

  // Create a PassManager to hold and optimize the collection of passes we are
  // about to build...
  //
  PassManager Passes;

  // Create a new optimization pass for each one specified on the command line
  for (unsigned i = 0; i < OptimizationList.size(); ++i) {
    enum Opts Opt = OptimizationList[i];
    for (unsigned j = 0; j < sizeof(OptTable)/sizeof(OptTable[0]); ++j)
      if (Opt == OptTable[j].OptID) {
        Passes.add(OptTable[j].PassCtor());
        break;
      }

    if (PrintEachXForm)
      Passes.add(new PrintModulePass(&std::cerr));
  }

  // Check that the module is well formed on completion of optimization
  Passes.add(createVerifierPass());

  // Write bytecode out to disk or cout as the last step...
  Passes.add(new WriteBytecodePass(Out, Out != &std::cout));

  // Now that we have all of the passes ready, run them.
  if (Passes.run(M.get()) && !Quiet)
    cerr << "Program modified.\n";

  return 0;
}
