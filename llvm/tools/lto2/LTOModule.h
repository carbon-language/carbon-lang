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

#include "llvm/Module.h"
#include "llvm/GlobalValue.h"
#include "llvm/Constants.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/StringMap.h"

#include "llvm-c/lto.h"

#include <vector>


//
// C++ class which implements the opaque lto_module_t
//
class LTOModule {
public:

    static bool              isBitcodeFile(const void* mem, size_t length);
    static bool              isBitcodeFile(const char* path);

    static bool              isBitcodeFileForTarget(const void* mem, 
                                    size_t length, const char* triplePrefix);

    static bool              isBitcodeFileForTarget(const char* path, 
                                                    const char* triplePrefix);

    static LTOModule*        makeLTOModule(const char* path, std::string& errMsg);
    static LTOModule*        makeLTOModule(const void* mem, size_t length,
                                                            std::string& errMsg);
                            ~LTOModule();

    const char*              getTargetTriple();
    uint32_t                 getSymbolCount();
    lto_symbol_attributes    getSymbolAttributes(uint32_t index);
    const char*              getSymbolName(uint32_t index);
    
    llvm::Module *           getLLVVMModule() { return _module; }
    bool                     targetSupported() { return (_target !=  NULL); }

private:
                            LTOModule(llvm::Module* m, llvm::TargetMachine* t);

    void                    addDefinedSymbol(llvm::GlobalValue* def, 
                                                    llvm::Mangler& mangler, 
                                                    bool isFunction);
    void                    addUndefinedSymbol(const char* name);
    void                    findExternalRefs(llvm::Value* value, 
                                                llvm::Mangler& mangler);
    
    typedef llvm::StringMap<uint8_t> StringSet;
    
    struct NameAndAttributes { 
        const char*            name; 
        lto_symbol_attributes  attributes; 
    };

    llvm::Module *                   _module;
    llvm::TargetMachine *            _target;
    bool                             _symbolsParsed;
    std::vector<NameAndAttributes>   _symbols;
    StringSet                        _defines;    // only needed to disambiguate tentative definitions
    StringSet                        _undefines;  // only needed to disambiguate tentative definitions
};

#endif // LTO_MODULE_H

