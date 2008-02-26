//===-LTOModule.cpp - LLVM Link Time Optimizer ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Link Time Optimization library. This library is 
// intended to be used by linker to optimize code at link time.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Linker.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/ModuleProvider.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/System/Program.h"
#include "llvm/System/Path.h"
#include "llvm/System/Signals.h"
#include "llvm/Target/SubtargetFeature.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Analysis/LoadValueNumbering.h"
#include "llvm/Support/MathExtras.h"

#include "LTOModule.h"

#include <fstream>

using namespace llvm;

bool LTOModule::isBitcodeFile(const void* mem, size_t length)
{
    return ( llvm::sys::IdentifyFileType((char*)mem, length) 
                                            == llvm::sys::Bitcode_FileType );
}

bool LTOModule::isBitcodeFile(const char* path)
{
    return llvm::sys::Path(path).isBitcodeFile();
}

bool LTOModule::isBitcodeFileForTarget(const void* mem, 
                                    size_t length, const char* triplePrefix) 
{
    bool result = false;
    MemoryBuffer* buffer;
    buffer = MemoryBuffer::getMemBuffer((char*)mem, (char*)mem+length);
    if ( buffer != NULL ) {
        ModuleProvider* mp = getBitcodeModuleProvider(buffer);
        if ( mp != NULL ) {
            std::string actualTarget = mp->getModule()->getTargetTriple();
            if  ( strncmp(actualTarget.c_str(), triplePrefix, 
                                    strlen(triplePrefix)) == 0) {
                result = true;
            }
            //  mp destructor will delete buffer
            delete mp;
        }
        else {
            // if getBitcodeModuleProvider failed, we need to delete buffer
            delete buffer;
        }
    }
    return result;
}

bool LTOModule::isBitcodeFileForTarget(const char* path,
                                                const char* triplePrefix) 
{
    bool result = false;
    MemoryBuffer* buffer;
    buffer = MemoryBuffer::getFile(path, strlen(path));
    if ( buffer != NULL ) {
        ModuleProvider* mp = getBitcodeModuleProvider(buffer);
        if ( mp != NULL ) {
            std::string actualTarget = mp->getModule()->getTargetTriple();
            if  ( strncmp(actualTarget.c_str(), triplePrefix, 
                                    strlen(triplePrefix)) == 0) {
                result = true;
            }
            //  mp destructor will delete buffer
            delete mp;
        }
        else {
            // if getBitcodeModuleProvider failed, we need to delete buffer
            delete buffer;
        }
    }
    return result;
}


LTOModule::LTOModule(Module* m, TargetMachine* t) 
 : _module(m), _target(t), _symbolsParsed(false)
{
}

LTOModule::~LTOModule()
{
    delete _module;
    if ( _target != NULL )
        delete _target;
}


LTOModule* LTOModule::makeLTOModule(const char* path, std::string& errMsg)
{
    MemoryBuffer* buffer = MemoryBuffer::getFile(path, strlen(path));
    if ( buffer != NULL ) {
        Module* m = ParseBitcodeFile(buffer, &errMsg);
        delete buffer;
        if ( m != NULL ) {
            const TargetMachineRegistry::entry* march = 
              TargetMachineRegistry::getClosestStaticTargetForModule(*m, errMsg);
            if ( march != NULL ) {
                std::string features;
                TargetMachine*    target = march->CtorFn(*m, features);
                return new LTOModule(m, target);
            }
        }
    }
    return NULL;
}

LTOModule* LTOModule::makeLTOModule(const void* mem, size_t length, 
                                                        std::string& errMsg)
{
    MemoryBuffer* buffer;
    buffer = MemoryBuffer::getMemBuffer((char*)mem, (char*)mem+length);
    if ( buffer != NULL ) {
        Module* m = ParseBitcodeFile(buffer, &errMsg);
        delete buffer;
        if ( m != NULL ) {
            const TargetMachineRegistry::entry* march = 
             TargetMachineRegistry::getClosestStaticTargetForModule(*m, errMsg);
            if ( march != NULL ) {
                std::string features;
                TargetMachine*    target = march->CtorFn(*m, features);
                return new LTOModule(m, target);
            }
        }
    }
    return NULL;
}


const char* LTOModule::getTargetTriple()
{
    return _module->getTargetTriple().c_str();
}

void LTOModule::addDefinedSymbol(GlobalValue* def, Mangler &mangler, 
                                bool isFunction)
{    
    const char* symbolName = ::strdup(mangler.getValueName(def).c_str());
    
    // set alignment part log2() can have rounding errors
    uint32_t align = def->getAlignment();
    uint32_t attr = align ? __builtin_ctz(def->getAlignment()) : 0;
    
    // set permissions part
    if ( isFunction )
        attr |= LTO_SYMBOL_PERMISSIONS_CODE;
    else {
        GlobalVariable* gv = dyn_cast<GlobalVariable>(def);
        if ( (gv != NULL) && gv->isConstant() )
            attr |= LTO_SYMBOL_PERMISSIONS_RODATA;
        else
            attr |= LTO_SYMBOL_PERMISSIONS_DATA;
    }
    
    // set definition part 
    if ( def->hasWeakLinkage() || def->hasLinkOnceLinkage() ) {
        // lvm bitcode does not differenciate between weak def data 
        // and tentative definitions!
        // HACK HACK HACK
        // C++ does not use tentative definitions, but does use weak symbols
        // so guess that anything that looks like a C++ symbol is weak and others
        // are tentative definitions
        if ( (strncmp(symbolName, "__Z", 3) == 0) )
            attr |= LTO_SYMBOL_DEFINITION_WEAK;
        else {
            attr |= LTO_SYMBOL_DEFINITION_TENTATIVE;
        }
    }
    else { 
        attr |= LTO_SYMBOL_DEFINITION_REGULAR;
    }
    
    // set scope part
    if ( def->hasHiddenVisibility() )
        attr |= LTO_SYMBOL_SCOPE_HIDDEN;
    else if ( def->hasExternalLinkage() || def->hasWeakLinkage() )
        attr |= LTO_SYMBOL_SCOPE_DEFAULT;
    else
        attr |= LTO_SYMBOL_SCOPE_INTERNAL;

    // add to table of symbols
    NameAndAttributes info;
    info.name = symbolName;
    info.attributes = (lto_symbol_attributes)attr;
    _symbols.push_back(info);
    _defines[info.name] = 1;
}


void LTOModule::addUndefinedSymbol(const char* name)
{    
    // ignore all llvm.* symbols
    if ( strncmp(name, "llvm.", 5) != 0 ) {
        _undefines[name] = 1;
    }
}



// Find exeternal symbols referenced by VALUE. This is a recursive function.
void LTOModule::findExternalRefs(Value* value, Mangler &mangler) {

    if (GlobalValue* gv = dyn_cast<GlobalValue>(value)) {
        if ( !gv->hasExternalLinkage() )
            addUndefinedSymbol(mangler.getValueName(gv).c_str());
    }

    // GlobalValue, even with InternalLinkage type, may have operands with 
    // ExternalLinkage type. Do not ignore these operands.
    if (Constant* c = dyn_cast<Constant>(value)) {
        // Handle ConstantExpr, ConstantStruct, ConstantArry etc..
        for (unsigned i = 0, e = c->getNumOperands(); i != e; ++i)
            findExternalRefs(c->getOperand(i), mangler);
    }
}


uint32_t LTOModule::getSymbolCount()
{
    if ( !_symbolsParsed ) {
        _symbolsParsed = true;
        
        // Use mangler to add GlobalPrefix to names to match linker names.
        Mangler mangler(*_module, _target->getTargetAsmInfo()->getGlobalPrefix());

        // add functions
        for (Module::iterator f = _module->begin(); f != _module->end(); ++f) {
            if ( f->isDeclaration() ) {
                addUndefinedSymbol(mangler.getValueName(f).c_str());
            }
            else {
                addDefinedSymbol(f, mangler, true);
                // add external symbols referenced by this function.
                for (Function::iterator b = f->begin(); b != f->end(); ++b) {
                    for (BasicBlock::iterator i = b->begin(); 
                                                        i != b->end(); ++i) {
                        for (unsigned count = 0, total = i->getNumOperands(); 
                                                    count != total; ++count) {
                            findExternalRefs(i->getOperand(count), mangler);
                        }
                    }
                }
            }
        }
        
        // add data 
        for (Module::global_iterator v = _module->global_begin(), 
                                    e = _module->global_end(); v !=  e; ++v) {
            if ( v->isDeclaration() ) {
                addUndefinedSymbol(mangler.getValueName(v).c_str());
            }
            else {
                addDefinedSymbol(v, mangler, false);
                // add external symbols referenced by this data
                for (unsigned count = 0, total = v->getNumOperands(); 
                                                count != total; ++count) {
                    findExternalRefs(v->getOperand(count), mangler);
                }
            }
        }

        // make symbols for all undefines
        for (StringSet::iterator it=_undefines.begin(); 
                                                it != _undefines.end(); ++it) {
            // if this symbol also has a definition, then don't make an undefine
            // because it is a tentative definition
            if ( _defines.find(it->getKeyData(), it->getKeyData()+it->getKeyLength()) == _defines.end() ) {
                NameAndAttributes info;
                info.name = it->getKeyData();
                info.attributes = LTO_SYMBOL_DEFINITION_UNDEFINED;
                _symbols.push_back(info);
            }
        }
        
    }
    
    return _symbols.size();
}


lto_symbol_attributes LTOModule::getSymbolAttributes(uint32_t index)
{
    if ( index < _symbols.size() )
        return _symbols[index].attributes;
    else
        return lto_symbol_attributes(0);
}

const char* LTOModule::getSymbolName(uint32_t index)
{
    if ( index < _symbols.size() )
        return _symbols[index].name;
    else
        return NULL;
}

