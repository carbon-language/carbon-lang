//===-- PluginLoader.cpp - Implement -load command line option ------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the -load <plugin> command line option handler.
//
//===----------------------------------------------------------------------===//

#define DONT_GET_PLUGIN_LOADER_OPTION
#include "Support/PluginLoader.h"
#include "Support/DynamicLinker.h"
#include <iostream>
using namespace llvm;

void PluginLoader::operator=(const std::string &Filename) {
  std::string ErrorMessage;
  if (LinkDynamicObject(Filename.c_str(), &ErrorMessage))
    std::cerr << "Error opening '" << Filename << "': " << ErrorMessage
              << "\n  -load request ignored.\n";	
}
