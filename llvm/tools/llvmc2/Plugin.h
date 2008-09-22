//===--- Plugin.h - The LLVM Compiler Driver --------------------*- C++ -*-===//
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

#ifndef LLVM_TOOLS_LLVMC2_PLUGIN_H
#define LLVM_TOOLS_LLVMC2_PLUGIN_H

namespace llvmc {

  class LanguageMap;
  class CompilationGraph;

  /// BasePlugin - An abstract base class for all LLVMC plugins.
  struct BasePlugin {

    /// PopulateLanguageMap - The auto-generated function that fills in
    /// the language map (map from file extensions to language names).
    virtual void PopulateLanguageMap(LanguageMap&) const = 0;

    /// PopulateCompilationGraph - The auto-generated function that
    /// populates the compilation graph with nodes and edges.
    virtual void PopulateCompilationGraph(CompilationGraph&) const = 0;
  };

  // Helper class for RegisterPlugin.
  class RegisterPluginImpl {
  protected:
    RegisterPluginImpl(BasePlugin*);
  };

  /// RegisterPlugin<T> template - Used to register LLVMC plugins.
  template <class T>
  struct RegisterPlugin : RegisterPluginImpl {
    RegisterPlugin() : RegisterPluginImpl (new T()) {}
  };

  /// PopulateLanguageMap - Fills in the language map by calling
  /// PopulateLanguageMap methods of all plugins.
  void PopulateLanguageMap(LanguageMap& langMap);

  /// PopulateCompilationGraph - Populates the compilation graph by
  /// calling PopulateCompilationGraph methods of all plugins.
  void PopulateCompilationGraph(CompilationGraph& tools);

}

#endif // LLVM_TOOLS_LLVMC2_PLUGIN_H
