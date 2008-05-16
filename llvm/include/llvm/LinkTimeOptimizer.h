//===-- llvm/LinkTimeOptimizer.h - Public Interface  ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include <cstring>

#define LLVM_LTO_VERSION 2

namespace llvm {

  class Module;
  class GlobalValue;
  class TargetMachine;

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
    LTOInternalLinkage, // Rename collisions when linking (static functions)
    LTOCommonLinkage    // tentative definitions (usually equivalent to weak)
  };

  enum LTOVisibilityTypes {
    LTODefaultVisibility = 0,  ///< The GV is visible
    LTOHiddenVisibility,       ///< The GV is hidden
    LTOProtectedVisibility     ///< The GV is protected
  };


  enum LTOCodeGenModel {
    LTO_CGM_Static,
    LTO_CGM_Dynamic,
    LTO_CGM_DynamicNoPIC
  };

  /// This class represents LLVM symbol information without exposing details
  /// of LLVM global values. It encapsulates symbol linkage information. This
  /// is typically used in hash_map where associated name identifies the 
  /// the symbol name.
  class LLVMSymbol {

  public:

    LTOLinkageTypes getLinkage() const { return linkage; }
    LTOVisibilityTypes getVisibility() const { return visibility; }
    void mayBeNotUsed();

    LLVMSymbol (enum LTOLinkageTypes lt, enum LTOVisibilityTypes vis, 
                GlobalValue *g, const std::string &n, 
                const std::string &m, int a) : linkage(lt), visibility(vis),
                                               gv(g), name(n), 
                                               mangledName(m), alignment(a) {}

    const char *getName() { return name.c_str(); }
    const char *getMangledName() { return mangledName.c_str(); }
    int getAlignment() { return alignment; }

  private:
    enum LTOLinkageTypes linkage;
    enum LTOVisibilityTypes visibility;
    GlobalValue *gv;
    std::string name;
    std::string mangledName;
    int alignment;
  };

  class string_compare {
  public:
    bool operator()(const char* left, const char* right) const { 
      return (strcmp(left, right) == 0); 
    }
  };

  /// This is abstract class to facilitate dlopen() interface.
  /// See LTO below for more info.
  class LinkTimeOptimizer {
  public:
    typedef hash_map<const char*, LLVMSymbol*, hash<const char*>, 
                     string_compare> NameToSymbolMap;
    typedef hash_map<const char*, Module*, hash<const char*>, 
                     string_compare> NameToModuleMap;
    virtual enum LTOStatus readLLVMObjectFile(const std::string &,
                                              NameToSymbolMap &,
                                              std::set<std::string> &) = 0;
    virtual enum LTOStatus optimizeModules(const std::string &,
                                           std::vector<const char*> &exportList,
                                           std::string &targetTriple,
                                           bool saveTemps, const char *) = 0;
    virtual void getTargetTriple(const std::string &, std::string &) = 0;
    virtual void removeModule (const std::string &InputFilename) = 0;
    virtual void setCodeGenModel(LTOCodeGenModel CGM) = 0;
    virtual void printVersion () = 0;
    virtual ~LinkTimeOptimizer() = 0;
  };

  /// This is the main link time optimization class. It exposes simple API
  /// to perform link time optimization using LLVM intermodular optimizer.
  class LTO : public LinkTimeOptimizer {

  public:
    typedef hash_map<const char*, LLVMSymbol*, hash<const char*>, 
                     string_compare> NameToSymbolMap;
    typedef hash_map<const char*, Module*, hash<const char*>, 
                     string_compare> NameToModuleMap;

    enum LTOStatus readLLVMObjectFile(const std::string &InputFilename,
                                      NameToSymbolMap &symbols,
                                      std::set<std::string> &references);
    enum LTOStatus optimizeModules(const std::string &OutputFilename,
                                   std::vector<const char*> &exportList,
                                   std::string &targetTriple, 
                                   bool saveTemps,  const char *);
    void getTargetTriple(const std::string &InputFilename, 
                         std::string &targetTriple);
    void removeModule (const std::string &InputFilename);
    void printVersion();

    void setCodeGenModel(LTOCodeGenModel CGM) {
      CGModel = CGM;
    }

    // Constructors and destructors
    LTO() : Target(NULL), CGModel(LTO_CGM_Dynamic) {
      /// TODO: Use Target info, it is available at this time.
    }
    ~LTO();

  private:
    Module *getModule (const std::string &InputFilename);
    enum LTOStatus optimize(Module *, std::ostream &, 
                            std::vector<const char *> &);
    void getTarget(Module *);

  private:
    std::vector<Module *> modules;
    NameToSymbolMap allSymbols;
    NameToModuleMap allModules;
    TargetMachine *Target;
    LTOCodeGenModel CGModel;
  };

} // End llvm namespace

/// This provides C interface to initialize link time optimizer. This allows
/// linker to use dlopen() interface to dynamically load LinkTimeOptimizer.
/// extern "C" helps, because dlopen() interface uses name to find the symbol.
extern "C"
llvm::LinkTimeOptimizer *createLLVMOptimizer(unsigned VERSION = LLVM_LTO_VERSION);

#endif
