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
#include "llvm/Target/TargetData.h"
#include "Support/CommandLine.h"
#include "Support/Signals.h"
#include <fstream>
#include <memory>
#include <algorithm>

using std::cerr;
using std::string;

//===----------------------------------------------------------------------===//
// PassNameParser class - Make use of the pass registration mechanism to
// automatically add a command line argument to opt for each pass.
//
namespace {  // anonymous namespace for local class...
class PassNameParser : public PassRegistrationListener, 
                       public cl::parser<const PassInfo*> {
  cl::Option *Opt;
public:
  PassNameParser() : Opt(0) {}
  
  void initialize(cl::Option &O) {
    Opt = &O;
    cl::parser<const PassInfo*>::initialize(O);

    // Add all of the passes to the map that got initialized before 'this' did.
    enumeratePasses();
  }

  static inline bool ignorablePass(const PassInfo *P) {
    // Ignore non-selectable and non-constructible passes!
    return P->getPassArgument() == 0 ||
          (P->getNormalCtor() == 0 && P->getDataCtor() == 0);
  }

  // Implement the PassRegistrationListener callbacks used to populate our map
  //
  virtual void passRegistered(const PassInfo *P) {
    if (ignorablePass(P) || !Opt) return;
    assert(findOption(P->getPassArgument()) == getNumOptions() &&
           "Two passes with the same argument attempted to be registered!");
    addLiteralOption(P->getPassArgument(), P, P->getPassName());
    Opt->addArgument(P->getPassArgument());
  }
  virtual void passEnumerate(const PassInfo *P) { passRegistered(P); }

  virtual void passUnregistered(const PassInfo *P) {
    if (ignorablePass(P) || !Opt) return;
    assert(findOption(P->getPassArgument()) != getNumOptions() &&
           "Registered Pass not in the pass map!");
    removeLiteralOption(P->getPassArgument());
    Opt->removeArgument(P->getPassArgument());
  }

  // ValLessThan - Provide a sorting comparator for Values elements...
  typedef std::pair<const char*,
                    std::pair<const PassInfo*, const char*> > ValType;
  static bool ValLessThan(const ValType &VT1, const ValType &VT2) {
    return std::string(VT1.first) < std::string(VT2.first);
  }

  // printOptionInfo - Print out information about this option.  Override the
  // default implementation to sort the table before we print...
  virtual void printOptionInfo(const cl::Option &O, unsigned GlobalWidth) const{
    PassNameParser *PNP = const_cast<PassNameParser*>(this);
    std::sort(PNP->Values.begin(), PNP->Values.end(), ValLessThan);
    cl::parser<const PassInfo*>::printOptionInfo(O, GlobalWidth);
  }
};
} // end anonymous namespace


// The OptimizationList is automatically populated with registered Passes by the
// PassNameParser.
//
static cl::list<const PassInfo*, bool, PassNameParser>
OptimizationList(cl::desc("Optimizations available:"));


// Other command line options...
//
static cl::opt<string>
InputFilename(cl::Positional, cl::desc("<input bytecode>"), cl::init("-"));

static cl::opt<string>
OutputFilename("o", cl::desc("Override output filename"),
               cl::value_desc("filename"));

static cl::opt<bool>
Force("f", cl::desc("Overwrite output files"));

static cl::opt<bool>
PrintEachXForm("p", cl::desc("Print module after each transformation"));

static cl::opt<bool>
Quiet("q", cl::desc("Don't print modifying pass names"));

static cl::alias
QuietA("quiet", cl::desc("Alias for -q"), cl::aliasopt(Quiet));


//===----------------------------------------------------------------------===//
// main for opt
//
int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv,
			      " llvm .bc -> .bc modular optimizer\n");

  // FIXME: This should be parameterizable eventually for different target
  // types...
  TargetData TD("opt target");

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
    const PassInfo *Opt = OptimizationList[i];
    
    if (Opt->getNormalCtor())
      Passes.add(Opt->getNormalCtor()());
    else if (Opt->getDataCtor())
      Passes.add(Opt->getDataCtor()(TD));  // Pass dummy target data...
    else
      cerr << "Cannot create pass: " << Opt->getPassName() << "\n";

    if (PrintEachXForm)
      Passes.add(new PrintModulePass(&cerr));
  }

  // Check that the module is well formed on completion of optimization
  Passes.add(createVerifierPass());

  // Write bytecode out to disk or cout as the last step...
  Passes.add(new WriteBytecodePass(Out, Out != &std::cout));

  // Now that we have all of the passes ready, run them.
  if (Passes.run(*M.get()) && !Quiet)
    cerr << "Program modified.\n";

  return 0;
}
