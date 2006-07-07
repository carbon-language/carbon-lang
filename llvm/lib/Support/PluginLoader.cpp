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
#include "llvm/Support/PluginLoader.h"
#include "llvm/System/DynamicLibrary.h"
#include <iostream>
#include <vector>
using namespace llvm;

static std::vector<std::string> *Plugins;

void PluginLoader::operator=(const std::string &Filename) {
  if (!Plugins)
    Plugins = new std::vector<std::string>();

  std::string Error;
  if (sys::DynamicLibrary::LoadLibraryPermanently(Filename.c_str(), &Error)) {
    std::cerr << "Error opening '" << Filename << "': " << Error
              << "\n  -load request ignored.\n";
  } else {
    Plugins->push_back(Filename);
  }
}

unsigned PluginLoader::getNumPlugins() {
  return Plugins ? Plugins->size() : 0;
}

std::string &PluginLoader::getPlugin(unsigned num) {
  assert(Plugins && num < Plugins->size() && "Asking for an out of bounds plugin");
  return (*Plugins)[num];
}
