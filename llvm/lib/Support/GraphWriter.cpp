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
#include "llvm/System/Path.h"
#include "llvm/System/Program.h"
#include "llvm/Config/config.h"
using namespace llvm;

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



void llvm::DisplayGraph(const sys::Path &Filename, bool wait,
                        GraphProgram::Name program) {
  std::string ErrMsg;
#if HAVE_GRAPHVIZ
  sys::Path Graphviz(LLVM_PATH_GRAPHVIZ);

  std::vector<const char*> args;
  args.push_back(Graphviz.c_str());
  args.push_back(Filename.c_str());
  args.push_back(0);
  
  errs() << "Running 'Graphviz' program... ";
  if (sys::Program::ExecuteAndWait(Graphviz, &args[0],0,0,0,0,&ErrMsg))
    errs() << "Error viewing graph " << Filename.str() << ": " << ErrMsg
           << "\n";
  else
    Filename.eraseFromDisk();

#elif (HAVE_GV && (HAVE_DOT || HAVE_FDP || HAVE_NEATO || \
                   HAVE_TWOPI || HAVE_CIRCO))
  sys::Path PSFilename = Filename;
  PSFilename.appendSuffix("ps");

  sys::Path prog;

  // Set default grapher
#if HAVE_CIRCO
  prog = sys::Path(LLVM_PATH_CIRCO);
#endif
#if HAVE_TWOPI
  prog = sys::Path(LLVM_PATH_TWOPI);
#endif
#if HAVE_NEATO
  prog = sys::Path(LLVM_PATH_NEATO);
#endif
#if HAVE_FDP
  prog = sys::Path(LLVM_PATH_FDP);
#endif
#if HAVE_DOT
  prog = sys::Path(LLVM_PATH_DOT);
#endif

  // Find which program the user wants
#if HAVE_DOT
  if (program == GraphProgram::DOT)
    prog = sys::Path(LLVM_PATH_DOT);
#endif
#if (HAVE_FDP)
  if (program == GraphProgram::FDP)
    prog = sys::Path(LLVM_PATH_FDP);
#endif
#if (HAVE_NEATO)
  if (program == GraphProgram::NEATO)
    prog = sys::Path(LLVM_PATH_NEATO);
#endif
#if (HAVE_TWOPI)
  if (program == GraphProgram::TWOPI)
    prog = sys::Path(LLVM_PATH_TWOPI);
#endif
#if (HAVE_CIRCO)
  if (program == GraphProgram::CIRCO)
    prog = sys::Path(LLVM_PATH_CIRCO);
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

  if (sys::Program::ExecuteAndWait(prog, &args[0], 0, 0, 0, 0, &ErrMsg)) {
     errs() << "Error viewing graph " << Filename.str() << ": '"
            << ErrMsg << "\n";
  } else {
    errs() << " done. \n";

    sys::Path gv(LLVM_PATH_GV);
    args.clear();
    args.push_back(gv.c_str());
    args.push_back(PSFilename.c_str());
    args.push_back("-spartan");
    args.push_back(0);
    
    ErrMsg.clear();
    if (wait) {
       if (sys::Program::ExecuteAndWait(gv, &args[0],0,0,0,0,&ErrMsg))
          errs() << "Error viewing graph: " << ErrMsg << "\n";
       Filename.eraseFromDisk();
       PSFilename.eraseFromDisk();
    }
    else {
       sys::Program::ExecuteNoWait(gv, &args[0],0,0,0,&ErrMsg);
       errs() << "Remember to erase graph files: " << Filename.str() << " "
              << PSFilename << "\n";
    }
  }
#elif HAVE_DOTTY
  sys::Path dotty(LLVM_PATH_DOTTY);

  std::vector<const char*> args;
  args.push_back(dotty.c_str());
  args.push_back(Filename.c_str());
  args.push_back(0);
  
  errs() << "Running 'dotty' program... ";
  if (sys::Program::ExecuteAndWait(dotty, &args[0],0,0,0,0,&ErrMsg)) {
     errs() << "Error viewing graph " << Filename.str() << ": "
            << ErrMsg << "\n";
  } else {
#ifdef __MINGW32__ // Dotty spawns another app and doesn't wait until it returns
    return;
#endif
    Filename.eraseFromDisk();
  }
#endif
}
