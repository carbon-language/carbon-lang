//===-- llvm/LinkTimeOptimizer.h - Public Interface  ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Devang Patel and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This header provides public interface to use LLVM link time optimization
// library. This is intended to be used by linker to do link time optimization.
//
//===----------------------------------------------------------------------===//

#ifndef __LTO_H__
#define __LTO_H__

#include <string>
#include <vector>
#include <set>
#include <llvm/ADT/hash_map>

namespace llvm {

  class Module;
  class GlobalValue;

  enum LTOStatus {
    LTO_UNKNOWN,
    LTO_OPT_SUCCESS,
    LTO_READ_SUCCESS,
    LTO_READ_FAILURE,
    LTO_WRITE_FAILURE,
    LTO_NO_TARGET,
    LTO_NO_WORK,
    LTO_MODULE_MERGE_FAILURE,
    LTO_ASM_FAILURE
  };
 
  enum LTOLinkageTypes {
    LTOExternalLinkage, // Externally visible function
    LTOLinkOnceLinkage, // Keep one copy of named function when linking (inline)
    LTOWeakLinkage,     // Keep one copy of named function when linking (weak)
    LTOInternalLinkage  // Rename collisions when linking (static functions)
  };

  /// This class represents LLVM symbol information without exposing details
  /// of LLVM global values. It encapsulates symbol linkage information. This
  /// is typically used in hash_map where associated name identifies the 
  /// the symbol name.
  class LLVMSymbol {

  public:

    LTOLinkageTypes getLinkage() const { return linkage; }
    void mayBeNotUsed();

    LLVMSymbol (enum LTOLinkageTypes lt, GlobalValue *g, const std::string &n, 
		const std::string &m) : linkage(lt), gv(g), name(n), 
					mangledName(m) {}

    const char *getName() { return name.c_str(); }
    const char *getMangledName() { return mangledName.c_str(); }

  private:
    enum LTOLinkageTypes linkage;
    GlobalValue *gv;
    std::string name;
    std::string mangledName;
  };

  class string_compare {
  public:
    bool operator()(const char* left, const char* right) const { 
      return (strcmp(left, right) == 0); 
    }
  };
  
  /// This is the main link time optimization class. It exposes simple API
  /// to perform link time optimization using LLVM intermodular optimizer.
  class LinkTimeOptimizer {

  public:
    typedef hash_map<const char*, LLVMSymbol*, hash<const char*>, 
		     string_compare> NameToSymbolMap;

    enum LTOStatus readLLVMObjectFile(const std::string &InputFilename,
				      NameToSymbolMap &symbols,
				      std::set<std::string> &references);
    enum LTOStatus optimizeModules(const std::string &OutputFilename,
				   std::vector<const char*> &exportList);

  private:
    std::vector<Module *> modules;
    NameToSymbolMap allSymbols;
  };

} // End llvm namespace

/// This provides C interface to initialize link time optimizer. This allows
/// linker to use dlopen() interface to dynamically load LinkTimeOptimizer.
/// extern "C" helps, because dlopen() interface uses name to find the symbol.
extern "C"
llvm::LinkTimeOptimizer *createLLVMOptimizer();

#endif
