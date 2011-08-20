//===- Cloog.cpp - Cloog interface ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Cloog[1] interface.
//
// The Cloog interface takes a Scop and generates a Cloog AST (clast). This
// clast can either be returned directly or it can be pretty printed to stdout.
//
// A typical clast output looks like this:
//
// for (c2 = max(0, ceild(n + m, 2); c2 <= min(511, floord(5 * n, 3)); c2++) {
//   bb2(c2);
// }
//
// [1] http://www.cloog.org/ - The Chunky Loop Generator
//
//===----------------------------------------------------------------------===//

#include "polly/Cloog.h"
#include "polly/LinkAllPasses.h"
#include "polly/ScopInfo.h"

#include "llvm/Assembly/Writer.h"
#include "llvm/Module.h"

#include "cloog/isl/domain.h"
#include "cloog/isl/cloog.h"

using namespace llvm;
using namespace polly;

namespace polly {
class Cloog {
  Scop *S;
  CloogOptions *Options;
  CloogState *State;
  clast_stmt *ClastRoot;

  void buildCloogOptions();
  CloogUnionDomain *buildCloogUnionDomain();
  CloogInput *buildCloogInput();

public:
  Cloog(Scop *Scop);

  ~Cloog();

  /// Write a .cloog input file
  void dump(FILE *F);

  /// Print a source code representation of the program.
  void pprint(llvm::raw_ostream &OS);

  /// Create the Cloog AST from this program.
  struct clast_root *getClast();
};

Cloog::Cloog(Scop *Scop) : S(Scop) {
  State = cloog_isl_state_malloc(Scop->getCtx());
  buildCloogOptions();
  ClastRoot = cloog_clast_create_from_input(buildCloogInput(), Options);
}

Cloog::~Cloog() {
  cloog_clast_free(ClastRoot);
  cloog_options_free(Options);
  cloog_state_free(State);
}

// Create a FILE* write stream and get the output to it written
// to a std::string.
class FileToString {
  int FD[2];
  FILE *input;
  static const int BUFFERSIZE = 20;

  char buf[BUFFERSIZE + 1];


public:
  FileToString() {
    pipe(FD);
    input = fdopen(FD[1], "w");
  }
  ~FileToString() {
    close(FD[0]);
    //close(FD[1]);
  }

  FILE *getInputFile() {
    return input;
  }

  void closeInput() {
    fclose(input);
    close(FD[1]);
  }

  std::string getOutput() {
    std::string output;
    int readSize;

    while (true) {
      readSize = read(FD[0], &buf, BUFFERSIZE);

      if (readSize <= 0)
        break;

      output += std::string(buf, readSize);
    }


    return output;
  }

};

/// Write .cloog input file.
void Cloog::dump(FILE *F) {
  CloogInput *Input = buildCloogInput();
  cloog_input_dump_cloog(F, Input, Options);
  cloog_input_free(Input);
}

/// Print a source code representation of the program.
void Cloog::pprint(raw_ostream &OS) {
  FileToString *Output = new FileToString();
  clast_pprint(Output->getInputFile(), ClastRoot, 0, Options);
  Output->closeInput();
  OS << Output->getOutput();
  delete (Output);
}

/// Create the Cloog AST from this program.
struct clast_root *Cloog::getClast() {
  return (clast_root*)ClastRoot;
}

void Cloog::buildCloogOptions() {
  Options = cloog_options_malloc(State);
  Options->quiet = 1;
  Options->strides = 1;
  Options->save_domains = 1;
  Options->noscalars = 1;
}

CloogUnionDomain *Cloog::buildCloogUnionDomain() {
  CloogUnionDomain *DU = cloog_union_domain_alloc(S->getNumParams());

  for (Scop::iterator SI = S->begin(), SE = S->end(); SI != SE; ++SI) {
    ScopStmt *Stmt = *SI;

    if (Stmt->isFinalRead())
      continue;

    CloogScattering *Scattering=
      cloog_scattering_from_isl_map(isl_map_copy(Stmt->getScattering()));
    CloogDomain *Domain =
      cloog_domain_from_isl_set(Stmt->getDomain());

    std::string entryName = Stmt->getBaseName();

    DU = cloog_union_domain_add_domain(DU, entryName.c_str(), Domain,
                                       Scattering, Stmt);
  }

  return DU;
}

CloogInput *Cloog::buildCloogInput() {
  CloogDomain *Context =
    cloog_domain_from_isl_set(isl_set_copy(S->getContext()));
  CloogUnionDomain *Statements = buildCloogUnionDomain();
  CloogInput *Input = cloog_input_alloc (Context, Statements);
  return Input;
}
} // End namespace polly.

namespace {

struct CloogExporter : public ScopPass {
  static char ID;
  Scop *S;
  explicit CloogExporter() : ScopPass(ID) {}

  std::string getFileName(Region *R) const;
  virtual bool runOnScop(Scop &S);
  void getAnalysisUsage(AnalysisUsage &AU) const;
};

}
std::string CloogExporter::getFileName(Region *R) const {
  std::string FunctionName = R->getEntry()->getParent()->getNameStr();
  std::string ExitName, EntryName;

  raw_string_ostream ExitStr(ExitName);
  raw_string_ostream EntryStr(EntryName);

  WriteAsOperand(EntryStr, R->getEntry(), false);
  EntryStr.str();

  if (R->getExit()) {
    WriteAsOperand(ExitStr, R->getExit(), false);
    ExitStr.str();
  } else
    ExitName = "FunctionExit";

  std::string RegionName = EntryName + "---" + ExitName;
  std::string FileName = FunctionName + "___" + RegionName + ".cloog";

  return FileName;
}

char CloogExporter::ID = 0;
bool CloogExporter::runOnScop(Scop &S) {
  Region &R = S.getRegion();
  CloogInfo &C = getAnalysis<CloogInfo>();

  std::string FunctionName = R.getEntry()->getParent()->getNameStr();
  std::string Filename = getFileName(&R);

  errs() << "Writing Scop '" << R.getNameStr() << "' in function '"
    << FunctionName << "' to '" << Filename << "'...\n";

  FILE *F = fopen(Filename.c_str(), "w");
  C.dump(F);
  fclose(F);

  return false;
}

void CloogExporter::getAnalysisUsage(AnalysisUsage &AU) const {
  // Get the Common analysis usage of ScopPasses.
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<CloogInfo>();
}

static RegisterPass<CloogExporter> A("polly-export-cloog",
                                    "Polly - Export the Cloog input file"
                                    " (Writes a .cloog file for each Scop)"
                                    );

llvm::Pass* polly::createCloogExporterPass() {
  return new CloogExporter();
}

/// Write a .cloog input file
void CloogInfo::dump(FILE *F) {
  C->dump(F);
}

/// Print a source code representation of the program.
void CloogInfo::pprint(llvm::raw_ostream &OS) {
  C->pprint(OS);
}

/// Create the Cloog AST from this program.
const struct clast_root *CloogInfo::getClast() {
  return C->getClast();
}

void CloogInfo::releaseMemory() {
  if (C) {
    delete C;
    C = 0;
  }
}

bool CloogInfo::runOnScop(Scop &S) {
  if (C)
    delete C;

  scop = &S;

  C = new Cloog(&S);

  return false;
}

void CloogInfo::printScop(raw_ostream &OS) const {
  Function *function = scop->getRegion().getEntry()->getParent();

  OS << function->getNameStr() << "():\n";

  C->pprint(OS);
}

void CloogInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  // Get the Common analysis usage of ScopPasses.
  ScopPass::getAnalysisUsage(AU);
}
char CloogInfo::ID = 0;


static RegisterPass<CloogInfo> B("polly-cloog",
                                 "Execute Cloog code generation");

Pass* polly::createCloogInfoPass() {
  return new CloogInfo();
}
