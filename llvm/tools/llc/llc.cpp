//===-- llc.cpp - Implement the LLVM Compiler -----------------------------===//
//
// This is the llc compiler driver.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bytecode/Reader.h"
#include "llvm/Target/TargetMachineImpls.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/PassNameParser.h"
#include "Support/CommandLine.h"
#include "Support/Signals.h"
#include <memory>
#include <fstream>

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
static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bytecode>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Output filename"), cl::value_desc("filename"));

static cl::opt<bool> Force("f", cl::desc("Overwrite output files"));

static cl::opt<bool>
DisableStrip("disable-strip",
          cl::desc("Do not strip the LLVM bytecode included in the executable"));

static cl::opt<bool>
DumpAsm("d", cl::desc("Print bytecode before native code generation"),
        cl::Hidden);

// GetFileNameRoot - Helper function to get the basename of a filename...
static inline std::string
GetFileNameRoot(const std::string &InputFilename)
{
  std::string IFN = InputFilename;
  std::string outputFilename;
  int Len = IFN.length();
  if (IFN[Len-3] == '.' && IFN[Len-2] == 'b' && IFN[Len-1] == 'c') {
    outputFilename = std::string(IFN.begin(), IFN.end()-3); // s/.bc/.s/
  } else {
    outputFilename = IFN;
  }
  return outputFilename;
}


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
  const TargetData &TD = Target.getTargetData();

  // Load the module to be compiled...
  std::auto_ptr<Module> M(ParseBytecodeFile(InputFilename));
  if (M.get() == 0)
    {
      std::cerr << argv[0] << ": bytecode didn't read correctly.\n";
      return 1;
    }

  // Build up all of the passes that we want to do to the module...
  PassManager Passes;

  Passes.add(new TargetData("llc", TD.isLittleEndian(), TD.getPointerSize(),
                            TD.getPointerAlignment(), TD.getDoubleAlignment()));

  // Create a new optimization pass for each one specified on the command line
  // Deal specially with tracing passes, which must be run differently than opt.
  // 
  for (unsigned i = 0; i < OptimizationList.size(); ++i) {
    const PassInfo *Opt = OptimizationList[i];
    
    // handle other passes as normal optimization passes
    if (Opt->getNormalCtor())
      Passes.add(Opt->getNormalCtor()());
    else if (Opt->getTargetCtor())
      Passes.add(Opt->getTargetCtor()(Target));
    else
      std::cerr << argv[0] << ": cannot create pass: "
                << Opt->getPassName() << "\n";
  }

  // Replace malloc and free instructions with library calls.
  // Do this after tracing until lli implements these lib calls.
  // For now, it will emulate malloc and free internally.
  // FIXME: This is sparc specific!
  Passes.add(createLowerAllocationsPass());

  // If LLVM dumping after transformations is requested, add it to the pipeline
  if (DumpAsm)
    Passes.add(new PrintFunctionPass("Code after xformations: \n", &std::cerr));

  // Strip all of the symbols from the bytecode so that it will be smaller...
  if (!DisableStrip)
    Passes.add(createSymbolStrippingPass());

  // Figure out where we are going to send the output...
  std::ostream *Out = 0;
  if (OutputFilename != "")
    {   // Specified an output filename?
      if (!Force && std::ifstream(OutputFilename.c_str())) {
        // If force is not specified, make sure not to overwrite a file!
        std::cerr << argv[0] << ": error opening '" << OutputFilename
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
          std::string OutputFilename = GetFileNameRoot(InputFilename); 
          OutputFilename += ".s";

          if (!Force && std::ifstream(OutputFilename.c_str()))
            {
              // If force is not specified, make sure not to overwrite a file!
              std::cerr << argv[0] << ": error opening '" << OutputFilename
                        << "': file exists!\n"
                        << "Use -f command line argument to force output\n";
              return 1;
            }

          Out = new std::ofstream(OutputFilename.c_str());
          if (!Out->good())
            {
              std::cerr << argv[0] << ": error opening " << OutputFilename
                        << "!\n";
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
    std::cerr << argv[0] << ": target '" << Target.getName()
              << " does not support static compilation!\n";
  } else {
    // Run our queue of passes all at once now, efficiently.
    Passes.run(*M.get());
  }

  // Delete the ostream if it's not a stdout stream
  if (Out != &std::cout) delete Out;

  return 0;
}
