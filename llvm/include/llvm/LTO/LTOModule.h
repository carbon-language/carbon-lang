//===-LTOModule.h - LLVM Link Time Optimizer ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the LTOModule class.
//
//===----------------------------------------------------------------------===//

#ifndef LTO_MODULE_H
#define LTO_MODULE_H

#include "llvm-c/lto.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/Target/TargetMachine.h"
#include <string>
#include <vector>

// Forward references to llvm classes.
namespace llvm {
  class Function;
  class GlobalValue;
  class MemoryBuffer;
  class TargetOptions;
  class Value;
}

//===----------------------------------------------------------------------===//
/// C++ class which implements the opaque lto_module_t type.
///
struct LTOModule {
private:
  typedef llvm::StringMap<uint8_t> StringSet;

  struct NameAndAttributes {
    const char        *name;
    uint32_t           attributes;
    bool               isFunction;
    const llvm::GlobalValue *symbol;
  };

  std::unique_ptr<llvm::Module>           _module;
  std::unique_ptr<llvm::TargetMachine>    _target;
  llvm::MCObjectFileInfo ObjFileInfo;
  StringSet                               _linkeropt_strings;
  std::vector<const char *>               _deplibs;
  std::vector<const char *>               _linkeropts;
  std::vector<NameAndAttributes>          _symbols;

  // _defines and _undefines only needed to disambiguate tentative definitions
  StringSet                               _defines;
  llvm::StringMap<NameAndAttributes>      _undefines;
  std::vector<const char*>                _asm_undefines;
  llvm::MCContext                         _context;

  // Use mangler to add GlobalPrefix to names to match linker names.
  llvm::Mangler                           _mangler;

  LTOModule(llvm::Module *m, llvm::TargetMachine *t);
public:
  /// Returns 'true' if the file or memory contents is LLVM bitcode.
  static bool isBitcodeFile(const void *mem, size_t length);
  static bool isBitcodeFile(const char *path);

  /// Returns 'true' if the file or memory contents is LLVM bitcode for the
  /// specified triple.
  static bool isBitcodeFileForTarget(const void *mem,
                                     size_t length,
                                     const char *triplePrefix);
  static bool isBitcodeFileForTarget(const char *path,
                                     const char *triplePrefix);

  /// Create an LTOModule. N.B. These methods take ownership of the buffer. The
  /// caller must have initialized the Targets, the TargetMCs, the AsmPrinters,
  /// and the AsmParsers by calling:
  ///
  /// InitializeAllTargets();
  /// InitializeAllTargetMCs();
  /// InitializeAllAsmPrinters();
  /// InitializeAllAsmParsers();
  static LTOModule *makeLTOModule(const char* path,
                                  llvm::TargetOptions options,
                                  std::string &errMsg);
  static LTOModule *makeLTOModule(int fd, const char *path,
                                  size_t size, llvm::TargetOptions options,
                                  std::string &errMsg);
  static LTOModule *makeLTOModule(int fd, const char *path,
                                  size_t map_size,
                                  off_t offset, llvm::TargetOptions options,
                                  std::string& errMsg);
  static LTOModule *makeLTOModule(const void *mem, size_t length,
                                  llvm::TargetOptions options,
                                  std::string &errMsg,
                                  llvm::StringRef path = "");

  /// Return the Module's target triple.
  const char *getTargetTriple() {
    return _module->getTargetTriple().c_str();
  }

  /// Set the Module's target triple.
  void setTargetTriple(const char *triple) {
    _module->setTargetTriple(triple);
  }

  /// Get the number of symbols
  uint32_t getSymbolCount() {
    return _symbols.size();
  }

  /// Get the attributes for a symbol at the specified index.
  lto_symbol_attributes getSymbolAttributes(uint32_t index) {
    if (index < _symbols.size())
      return lto_symbol_attributes(_symbols[index].attributes);
    return lto_symbol_attributes(0);
  }

  /// Get the name of the symbol at the specified index.
  const char *getSymbolName(uint32_t index) {
    if (index < _symbols.size())
      return _symbols[index].name;
    return nullptr;
  }

  /// Get the number of dependent libraries
  uint32_t getDependentLibraryCount() {
    return _deplibs.size();
  }

  /// Get the dependent library at the specified index.
  const char *getDependentLibrary(uint32_t index) {
    if (index < _deplibs.size())
      return _deplibs[index];
    return nullptr;
  }

  /// Get the number of linker options
  uint32_t getLinkerOptCount() {
    return _linkeropts.size();
  }

  /// Get the linker option at the specified index.
  const char *getLinkerOpt(uint32_t index) {
    if (index < _linkeropts.size())
      return _linkeropts[index];
    return nullptr;
  }

  /// Return the Module.
  llvm::Module *getLLVVMModule() { return _module.get(); }

  const std::vector<const char*> &getAsmUndefinedRefs() {
    return _asm_undefines;
  }

private:
  /// Parse metadata from the module
  // FIXME: it only parses "Linker Options" metadata at the moment
  void parseMetadata();

  /// Parse the symbols from the module and model-level ASM and add them to
  /// either the defined or undefined lists.
  bool parseSymbols(std::string &errMsg);

  /// Add a symbol which isn't defined just yet to a list to be resolved later.
  void addPotentialUndefinedSymbol(const llvm::GlobalValue *dcl, bool isFunc);

  /// Add a defined symbol to the list.
  void addDefinedSymbol(const llvm::GlobalValue *def, bool isFunction);

  /// Add a function symbol as defined to the list.
  void addDefinedFunctionSymbol(const llvm::Function *f);

  /// Add a data symbol as defined to the list.
  void addDefinedDataSymbol(const llvm::GlobalValue *v);

  /// Add global symbols from module-level ASM to the defined or undefined
  /// lists.
  bool addAsmGlobalSymbols(std::string &errMsg);

  /// Add a global symbol from module-level ASM to the defined list.
  void addAsmGlobalSymbol(const char *, lto_symbol_attributes scope);

  /// Add a global symbol from module-level ASM to the undefined list.
  void addAsmGlobalSymbolUndef(const char *);

  /// Parse i386/ppc ObjC class data structure.
  void addObjCClass(const llvm::GlobalVariable *clgv);

  /// Parse i386/ppc ObjC category data structure.
  void addObjCCategory(const llvm::GlobalVariable *clgv);

  /// Parse i386/ppc ObjC class list data structure.
  void addObjCClassRef(const llvm::GlobalVariable *clgv);

  /// Get string that the data pointer points to.
  bool objcClassNameFromExpression(const llvm::Constant* c, std::string &name);

  /// Returns 'true' if the memory buffer is for the specified target triple.
  static bool isTargetMatch(llvm::MemoryBuffer *memBuffer,
                            const char *triplePrefix);

  /// Create an LTOModule (private version). N.B. This method takes ownership of
  /// the buffer.
  static LTOModule *makeLTOModule(llvm::MemoryBuffer *buffer,
                                  llvm::TargetOptions options,
                                  std::string &errMsg);

  /// Create a MemoryBuffer from a memory range with an optional name.
  static llvm::MemoryBuffer *makeBuffer(const void *mem, size_t length,
                                        llvm::StringRef name = "");
};

#endif // LTO_MODULE_H
