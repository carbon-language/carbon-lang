//===-- PluginLoader.cpp - Implement -load command line option ------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the -load <plugin> command line option processor.  When
// linked into a program, this new command line option is available that allows
// users to load shared objects into the running program.
//
// Note that there are no symbols exported by the .o file generated for this
// .cpp file.  Because of this, a program must link against support.o instead of
// support.a: otherwise this translation unit will not be included.
//
//===----------------------------------------------------------------------===//

#include "Support/DynamicLinker.h"
#include "Support/CommandLine.h"
#include "Config/dlfcn.h"
#include "Config/link.h"
#include <iostream>
using namespace llvm;

namespace {
  struct PluginLoader {
    void operator=(const std::string &Filename) {
      std::string ErrorMessage;
      if (LinkDynamicObject (Filename.c_str (), &ErrorMessage))
        std::cerr << "Error opening '" << Filename << "': " << ErrorMessage
                  << "\n  -load request ignored.\n";	
    }
  };
}

// This causes operator= above to be invoked for every -load option.
static cl::opt<PluginLoader, false, cl::parser<std::string> >
LoadOpt("load", cl::ZeroOrMore, cl::value_desc("plugin.so"),
        cl::desc("Load the specified plugin"));
