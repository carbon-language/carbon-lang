//===-- GraphWriter.cpp - Implements GraphWriter support routines ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements misc. GraphWriter support routines.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/GraphWriter.h"
#include "llvm/Config/config.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
using namespace llvm;

static cl::opt<bool> ViewBackground("view-background", cl::Hidden,
  cl::desc("Execute graph viewer in the background. Creates tmp file litter."));

std::string llvm::DOT::EscapeString(const std::string &Label) {
  std::string Str(Label);
  for (unsigned i = 0; i != Str.length(); ++i)
  switch (Str[i]) {
    case '\n':
      Str.insert(Str.begin()+i, '\\');  // Escape character...
      ++i;
      Str[i] = 'n';
      break;
    case '\t':
      Str.insert(Str.begin()+i, ' ');  // Convert to two spaces
      ++i;
      Str[i] = ' ';
      break;
    case '\\':
      if (i+1 != Str.length())
        switch (Str[i+1]) {
          case 'l': continue; // don't disturb \l
          case '|': case '{': case '}':
            Str.erase(Str.begin()+i); continue;
          default: break;
        }
    case '{': case '}':
    case '<': case '>':
    case '|': case '"':
      Str.insert(Str.begin()+i, '\\');  // Escape character...
      ++i;  // don't infinite loop
      break;
  }
  return Str;
}

/// \brief Get a color string for this node number. Simply round-robin selects
/// from a reasonable number of colors.
StringRef llvm::DOT::getColorString(unsigned ColorNumber) {
  static const int NumColors = 20;
  static const char* Colors[NumColors] = {
    "aaaaaa", "aa0000", "00aa00", "aa5500", "0055ff", "aa00aa", "00aaaa",
    "555555", "ff5555", "55ff55", "ffff55", "5555ff", "ff55ff", "55ffff",
    "ffaaaa", "aaffaa", "ffffaa", "aaaaff", "ffaaff", "aaffff"};
  return Colors[ColorNumber % NumColors];
}

std::string llvm::createGraphFilename(const Twine &Name, int &FD) {
  FD = -1;
  SmallString<128> Filename;
  error_code EC = sys::fs::createTemporaryFile(Name, "dot", FD, Filename);
  if (EC) {
    errs() << "Error: " << EC.message() << "\n";
    return "";
  }

  errs() << "Writing '" << Filename << "'... ";
  return Filename.str();
}

// Execute the graph viewer. Return true if successful.
static bool LLVM_ATTRIBUTE_UNUSED
ExecGraphViewer(StringRef ExecPath, std::vector<const char*> &args,
                StringRef Filename, bool wait, std::string &ErrMsg) {
  if (wait) {
    if (sys::ExecuteAndWait(ExecPath, &args[0],0,0,0,0,&ErrMsg)) {
      errs() << "Error: " << ErrMsg << "\n";
      return false;
    }
    sys::fs::remove(Filename);
    errs() << " done. \n";
  }
  else {
    sys::ExecuteNoWait(ExecPath, &args[0],0,0,0,&ErrMsg);
    errs() << "Remember to erase graph file: " << Filename.str() << "\n";
  }
  return true;
}

void llvm::DisplayGraph(StringRef FilenameRef, bool wait,
                        GraphProgram::Name program) {
  std::string Filename = FilenameRef;
  wait &= !ViewBackground;
  std::string ErrMsg;
#if HAVE_GRAPHVIZ
  std::string Graphviz(LLVM_PATH_GRAPHVIZ);

  std::vector<const char*> args;
  args.push_back(Graphviz.c_str());
  args.push_back(Filename.c_str());
  args.push_back(0);

  errs() << "Running 'Graphviz' program... ";
  if (!ExecGraphViewer(Graphviz, args, Filename, wait, ErrMsg))
    return;

#elif HAVE_XDOT
  std::vector<const char*> args;
  args.push_back(LLVM_PATH_XDOT);
  args.push_back(Filename.c_str());

  switch (program) {
  case GraphProgram::DOT:   args.push_back("-f"); args.push_back("dot"); break;
  case GraphProgram::FDP:   args.push_back("-f"); args.push_back("fdp"); break;
  case GraphProgram::NEATO: args.push_back("-f"); args.push_back("neato");break;
  case GraphProgram::TWOPI: args.push_back("-f"); args.push_back("twopi");break;
  case GraphProgram::CIRCO: args.push_back("-f"); args.push_back("circo");break;
  }

  args.push_back(0);

  errs() << "Running 'xdot.py' program... ";
  if (!ExecGraphViewer(LLVM_PATH_XDOT, args, Filename, wait, ErrMsg))
    return;

#elif (HAVE_GV && (HAVE_DOT || HAVE_FDP || HAVE_NEATO || \
                   HAVE_TWOPI || HAVE_CIRCO))
  std::string PSFilename = Filename + ".ps";
  std::string prog;

  // Set default grapher
#if HAVE_CIRCO
  prog = LLVM_PATH_CIRCO;
#endif
#if HAVE_TWOPI
  prog = LLVM_PATH_TWOPI;
#endif
#if HAVE_NEATO
  prog = LLVM_PATH_NEATO;
#endif
#if HAVE_FDP
  prog = LLVM_PATH_FDP;
#endif
#if HAVE_DOT
  prog = LLVM_PATH_DOT;
#endif

  // Find which program the user wants
#if HAVE_DOT
  if (program == GraphProgram::DOT)
    prog = LLVM_PATH_DOT;
#endif
#if (HAVE_FDP)
  if (program == GraphProgram::FDP)
    prog = LLVM_PATH_FDP;
#endif
#if (HAVE_NEATO)
  if (program == GraphProgram::NEATO)
    prog = LLVM_PATH_NEATO;
#endif
#if (HAVE_TWOPI)
  if (program == GraphProgram::TWOPI)
    prog = LLVM_PATH_TWOPI;
#endif
#if (HAVE_CIRCO)
  if (program == GraphProgram::CIRCO)
    prog = LLVM_PATH_CIRCO;
#endif

  std::vector<const char*> args;
  args.push_back(prog.c_str());
  args.push_back("-Tps");
  args.push_back("-Nfontname=Courier");
  args.push_back("-Gsize=7.5,10");
  args.push_back(Filename.c_str());
  args.push_back("-o");
  args.push_back(PSFilename.c_str());
  args.push_back(0);

  errs() << "Running '" << prog << "' program... ";

  if (!ExecGraphViewer(prog, args, Filename, wait, ErrMsg))
    return;

  std::string gv(LLVM_PATH_GV);
  args.clear();
  args.push_back(gv.c_str());
  args.push_back(PSFilename.c_str());
  args.push_back("--spartan");
  args.push_back(0);

  ErrMsg.clear();
  if (!ExecGraphViewer(gv, args, PSFilename, wait, ErrMsg))
    return;

#elif HAVE_DOTTY
  std::string dotty(LLVM_PATH_DOTTY);

  std::vector<const char*> args;
  args.push_back(dotty.c_str());
  args.push_back(Filename.c_str());
  args.push_back(0);

// Dotty spawns another app and doesn't wait until it returns
#if defined (__MINGW32__) || defined (_WINDOWS)
  wait = false;
#endif
  errs() << "Running 'dotty' program... ";
  if (!ExecGraphViewer(dotty, args, Filename, wait, ErrMsg))
    return;
#else
  (void)Filename;
  (void)ErrMsg;
#endif
}
