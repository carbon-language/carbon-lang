//===-- GraphWriter.cpp - Implements GraphWriter support routines ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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
#include <iostream>
using namespace llvm;

void llvm::DisplayGraph(const sys::Path &Filename) {
#if HAVE_GRAPHVIZ
  sys::Path Graphviz(LLVM_PATH_GRAPHVIZ);

  std::vector<const char*> args;
  args.push_back(Graphviz.c_str());
  args.push_back(Filename.c_str());
  args.push_back(0);
  
  std::cerr << "Running 'Graphviz' program... " << std::flush;
  if (sys::Program::ExecuteAndWait(Graphviz, &args[0])) {
    std::cerr << "Error viewing graph: 'Graphviz' not in path?\n";
  }
#elif (HAVE_GV && HAVE_DOT)
  sys::Path PSFilename = Filename;
  PSFilename.appendSuffix("ps");
  
  sys::Path dot(LLVM_PATH_DOT);

  std::vector<const char*> args;
  args.push_back(dot.c_str());
  args.push_back("-Tps");
  args.push_back("-Nfontname=Courier");
  args.push_back("-Gsize=7.5,10");
  args.push_back(Filename.c_str());
  args.push_back("-o");
  args.push_back(PSFilename.c_str());
  args.push_back(0);
  
  std::cerr << "Running 'dot' program... " << std::flush;
  if (sys::Program::ExecuteAndWait(dot, &args[0])) {
    std::cerr << "Error viewing graph: 'dot' not in path?\n";
  } else {
    std::cerr << " done. \n";

    sys::Path gv(LLVM_PATH_GV);
    args.clear();
    args.push_back(gv.c_str());
    args.push_back(PSFilename.c_str());
    args.push_back(0);
    
    sys::Program::ExecuteAndWait(gv, &args[0]);
  }
  PSFilename.eraseFromDisk();
#elif HAVE_DOTTY
  sys::Path dotty(LLVM_PATH_DOTTY);

  std::vector<const char*> args;
  args.push_back(Filename.c_str());
  args.push_back(0);
  
  std::cerr << "Running 'dotty' program... " << std::flush;
  if (sys::Program::ExecuteAndWait(dotty, &args[0])) {
    std::cerr << "Error viewing graph: 'dotty' not in path?\n";
  } else {
#ifdef __MINGW32__ // Dotty spawns another app and doesn't wait until it returns.
    return;
#endif
  }
#endif
  
  Filename.eraseFromDisk();
}
