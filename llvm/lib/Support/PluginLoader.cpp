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

std::vector<std::string> plugins;

void PluginLoader::operator=(const std::string &Filename) {
  std::string ErrorMessage;
  try {
    sys::DynamicLibrary::LoadLibraryPermanently(Filename.c_str());
    plugins.push_back(Filename);
  } catch (const std::string& errmsg) {
    if (errmsg.empty()) {
      ErrorMessage = "Unknown";
    } else {
      ErrorMessage = errmsg;
    }
  }
  if (!ErrorMessage.empty())
    std::cerr << "Error opening '" << Filename << "': " << ErrorMessage
              << "\n  -load request ignored.\n";
}

unsigned PluginLoader::getNumPlugins()
{
  return plugins.size();
}

std::string& PluginLoader::getPlugin(unsigned num)
{
  assert(num < plugins.size() && "Asking for an out of bounds plugin");
  return plugins[num];
}
