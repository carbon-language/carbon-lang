//===- Hello.cpp - Example code from "Writing an LLVMC Plugin" ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Test plugin for LLVMC. Shows how to write plugins without using TableGen.
//
//===----------------------------------------------------------------------===//

#include "llvm/CompilerDriver/CompilationGraph.h"
#include "llvm/CompilerDriver/Plugin.h"
#include "llvm/Support/raw_ostream.h"

namespace {
struct MyPlugin : public llvmc::BasePlugin {

  void PreprocessOptions() const
  {}

  void PopulateLanguageMap(llvmc::LanguageMap&) const
  { outs() << "Hello!\n"; }

  void PopulateCompilationGraph(llvmc::CompilationGraph&) const
  {}
};

static llvmc::RegisterPlugin<MyPlugin> RP("Hello", "Hello World plugin");

}
