//===--- Plugin.cpp - The LLVM Compiler Driver ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Plugin support for llvmc2.
//
//===----------------------------------------------------------------------===//

#include "Plugin.h"

#include <vector>

namespace {
  typedef std::vector<llvmc::BasePlugin*> PluginRegistry;
  static PluginRegistry GlobalPluginRegistry;
}

namespace llvmc {

  RegisterPluginImpl::RegisterPluginImpl(BasePlugin* plugin) {
    GlobalPluginRegistry.push_back(plugin);
  }

  void PopulateLanguageMap(LanguageMap& langMap) {
    for (PluginRegistry::const_iterator B = GlobalPluginRegistry.begin(),
           E = GlobalPluginRegistry.end(); B != E; ++B)
      (*B)->PopulateLanguageMap(langMap);
  }

  void PopulateCompilationGraph(CompilationGraph& graph) {
    for (PluginRegistry::const_iterator B = GlobalPluginRegistry.begin(),
           E = GlobalPluginRegistry.end(); B != E; ++B)
      (*B)->PopulateCompilationGraph(graph);
  }

}
