//===-- llc.cpp - Implement the LLVM Compiler -----------------------------===//
//
// This is the llc compiler driver.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bytecode/Reader.h"
#include "llvm/Target/TargetMachineImpls.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Linker.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Bytecode/WriteBytecodePass.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/PassNameParser.h"
#include "Support/CommandLine.h"
#include "Support/Signals.h"
#include <memory>
#include <fstream>
using std::string;
using std::cerr;

//------------------------------------------------------------------------------
// Option declarations for LLC.
//------------------------------------------------------------------------------

// Make all registered optimization passes available to llc.  These passes
// will all be run before the simplification and lowering steps used by the
// back-end code generator, and will be run in the order specified on the
// command line. The OptimizationList is automatically populated with
// registered Passes by the PassNameParser.
//
static cl::list<const PassInfo*, bool,
                FilteredPassNameParser<PassInfo::Optimization> >
OptimizationList(cl::desc("Optimizations available:"));


// General options for llc.  Other pass-specific options are specified
// within the corresponding llc passes, and target-specific options
// and back-end code generation options are specified with the target machine.
// 
static cl::opt<string>
InputFilename(cl::Positional, cl::desc("<input bytecode>"), cl::init("-"));

static cl::opt<string>
OutputFilename("o", cl::desc("Output filename"), cl::value_desc("filename"));

static cl::opt<bool> Force("f", cl::desc("Overwrite output files"));

static cl::opt<bool>
DumpAsm("d", cl::desc("Print bytecode before native code generation"),
        cl::Hidden);

static cl::opt<string>
TraceLibPath("tracelibpath", cl::desc("Path to libinstr for trace code"),
             cl::value_desc("directory"), cl::Hidden);


// flags set from -tracem and -trace options to control tracing
static bool TraceFunctions   = false;
static bool TraceBasicBlocks = false;


// GetFileNameRoot - Helper function to get the basename of a filename...
static inline string
GetFileNameRoot(const string &InputFilename)
{
  string IFN = InputFilename;
  string outputFilename;
  int Len = IFN.length();
  if (IFN[Len-3] == '.' && IFN[Len-2] == 'b' && IFN[Len-1] == 'c') {
    outputFilename = string(IFN.begin(), IFN.end()-3); // s/.bc/.s/
  } else {
    outputFilename = IFN;
  }
  return outputFilename;
}

static bool
insertTraceCodeFor(Module &M)
{
  PassManager Passes;

  // Insert trace code in all functions in the module
  if (TraceBasicBlocks)
    Passes.add(createTraceValuesPassForBasicBlocks());
  else if (TraceFunctions)
    Passes.add(createTraceValuesPassForFunction());
  else
    return false;

  // Eliminate duplication in constant pool
  Passes.add(createConstantMergePass());

  // Run passes to insert and clean up trace code...
  Passes.run(M);

  std::string ErrorMessage;

  // Load the module that contains the runtime helper routines neccesary for
  // pointer hashing and stuff...  link this module into the program if possible
  //
  Module *TraceModule = ParseBytecodeFile(TraceLibPath+"libinstr.bc");

  // Check if the TraceLibPath contains a valid module.  If not, try to load
  // the module from the current LLVM-GCC install directory.  This is kindof
  // a hack, but allows people to not HAVE to have built the library.
  //
  if (TraceModule == 0)
    TraceModule = ParseBytecodeFile("/home/vadve/lattner/cvs/gcc_install/lib/"
                                    "gcc-lib/llvm/3.1/libinstr.bc");

  // If we still didn't get it, cancel trying to link it in...
  if (TraceModule == 0)
    cerr << "Warning, could not load trace routines to link into program!\n";
  else
    {
      // Link in the trace routines... if this fails, don't panic, because the
      // compile should still succeed, but the native linker will probably fail.
      //
      std::auto_ptr<Module> TraceRoutines(TraceModule);
      if (LinkModules(&M, TraceRoutines.get(), &ErrorMessage))
        cerr << "Warning: Error linking in trace routines: "
             << ErrorMessage << "\n";
    }

  // Write out the module with tracing code just before code generation
  assert (InputFilename != "-"
          && "Cannot write out traced bytecode when reading input from stdin");
  string TraceFilename = GetFileNameRoot(InputFilename) + ".trace.bc";

  std::ofstream Out(TraceFilename.c_str());
  if (!Out.good())
    cerr << "Error opening '" << TraceFilename
         << "'!: Skipping output of trace code as bytecode\n";
  else
    {
      cerr << "Emitting trace code to '" << TraceFilename
           << "' for comparison...\n";
      WriteBytecodeToFile(&M, Out);
    }

  return true;
}

// Making tracing a module pass so the entire module with tracing
// can be written out before continuing.
struct InsertTracingCodePass: public Pass {
  virtual bool run(Module &M) {
    return insertTraceCodeFor(M); 
  }
};


//===---------------------------------------------------------------------===//
// Function main()
// 
// Entry point for the llc compiler.
//===---------------------------------------------------------------------===//

int
main(int argc, char **argv)
{
  cl::ParseCommandLineOptions(argc, argv, " llvm system compiler\n");
  
  // Allocate a target... in the future this will be controllable on the
  // command line.
  std::auto_ptr<TargetMachine> target(allocateSparcTargetMachine());
  assert(target.get() && "Could not allocate target machine!");

  TargetMachine &Target = *target.get();

  // Load the module to be compiled...
  std::auto_ptr<Module> M(ParseBytecodeFile(InputFilename));
  if (M.get() == 0)
    {
      cerr << argv[0] << ": bytecode didn't read correctly.\n";
      return 1;
    }

  // Build up all of the passes that we want to do to the module...
  PassManager Passes;

  // Create a new optimization pass for each one specified on the command line
  // Deal specially with tracing passes, which must be run differently than opt.
  // 
  for (unsigned i = 0; i < OptimizationList.size(); ++i)
    {
      const PassInfo *Opt = OptimizationList[i];
      
      if (std::string(Opt->getPassArgument()) == "trace")
        TraceFunctions = !(TraceBasicBlocks = true);
      else if (std::string(Opt->getPassArgument()) == "tracem")
        TraceFunctions = !(TraceBasicBlocks = false);
      else
        { // handle other passes as normal optimization passes
          if (Opt->getNormalCtor())
            Passes.add(Opt->getNormalCtor()());
          else if (Opt->getDataCtor())
            Passes.add(Opt->getDataCtor()(Target.getTargetData()));
          else if (Opt->getTargetCtor())
            Passes.add(Opt->getTargetCtor()(Target));
          else
            cerr << argv[0] << ": cannot create pass: "
                 << Opt->getPassName() << "\n";
        }
    }

  // Run tracing passes after other optimization passes and before llc passes.
  if (TraceFunctions || TraceBasicBlocks)
    Passes.add(new InsertTracingCodePass);

  // Decompose multi-dimensional refs into a sequence of 1D refs
  Passes.add(createDecomposeMultiDimRefsPass());

  // Replace malloc and free instructions with library calls.
  // Do this after tracing until lli implements these lib calls.
  // For now, it will emulate malloc and free internally.
  Passes.add(createLowerAllocationsPass());

  // If LLVM dumping after transformations is requested, add it to the pipeline
  if (DumpAsm)
    Passes.add(new PrintFunctionPass("Code after xformations: \n", &cerr));

  // Strip all of the symbols from the bytecode so that it will be smaller...
  Passes.add(createSymbolStrippingPass());

  // Figure out where we are going to send the output...
  std::ostream *Out = 0;
  if (OutputFilename != "")
    {   // Specified an output filename?
      if (!Force && std::ifstream(OutputFilename.c_str())) {
        // If force is not specified, make sure not to overwrite a file!
        cerr << argv[0] << ": error opening '" << OutputFilename
             << "': file exists!\n"
             << "Use -f command line argument to force output\n";
        return 1;
      }
      Out = new std::ofstream(OutputFilename.c_str());

      // Make sure that the Out file gets unlink'd from the disk if we get a
      // SIGINT
      RemoveFileOnSignal(OutputFilename);
    }
  else
    {
      if (InputFilename == "-")
        {
          OutputFilename = "-";
          Out = &std::cout;
        }
      else
        {
          string OutputFilename = GetFileNameRoot(InputFilename); 
          OutputFilename += ".s";

          if (!Force && std::ifstream(OutputFilename.c_str()))
            {
              // If force is not specified, make sure not to overwrite a file!
              cerr << argv[0] << ": error opening '" << OutputFilename
                   << "': file exists!\n"
                   << "Use -f command line argument to force output\n";
              return 1;
            }

          Out = new std::ofstream(OutputFilename.c_str());
          if (!Out->good())
            {
              cerr << argv[0] << ": error opening " << OutputFilename << "!\n";
              delete Out;
              return 1;
            }

          // Make sure that the Out file gets unlink'd from the disk if we get a
          // SIGINT
          RemoveFileOnSignal(OutputFilename);
        }
    }

  // Ask the target to add backend passes as neccesary
  if (Target.addPassesToEmitAssembly(Passes, *Out)) {
    cerr << argv[0] << ": target '" << Target.getName()
         << " does not support static compilation!\n";
  } else {
    // Run our queue of passes all at once now, efficiently.
    Passes.run(*M.get());
  }

  // Delete the ostream if it's not a stdout stream
  if (Out != &std::cout) delete Out;

  return 0;
}
