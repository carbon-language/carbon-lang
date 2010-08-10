//===-- LTOModule.cpp - LLVM Link Time Optimizer --------------------------===//
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

#include "LTOModule.h"

#include "llvm/Constants.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/System/Host.h"
#include "llvm/System/Path.h"
#include "llvm/System/Process.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Target/SubtargetFeature.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetSelect.h"

using namespace llvm;

bool LTOModule::isBitcodeFile(const void *mem, size_t length) {
  return llvm::sys::IdentifyFileType((char*)mem, length)
    == llvm::sys::Bitcode_FileType;
}

bool LTOModule::isBitcodeFile(const char *path) {
  return llvm::sys::Path(path).isBitcodeFile();
}

bool LTOModule::isBitcodeFileForTarget(const void *mem, size_t length,
                                       const char *triplePrefix) {
  MemoryBuffer *buffer = makeBuffer(mem, length);
  if (!buffer)
    return false;
  return isTargetMatch(buffer, triplePrefix);
}


bool LTOModule::isBitcodeFileForTarget(const char *path,
                                       const char *triplePrefix) {
  MemoryBuffer *buffer = MemoryBuffer::getFile(path);
  if (buffer == NULL)
    return false;
  return isTargetMatch(buffer, triplePrefix);
}

// Takes ownership of buffer.
bool LTOModule::isTargetMatch(MemoryBuffer *buffer, const char *triplePrefix) {
  OwningPtr<Module> m(getLazyBitcodeModule(buffer, getGlobalContext()));
  // On success, m owns buffer and both are deleted at end of this method.
  if (!m) {
    delete buffer;
    return false;
  }
  std::string actualTarget = m->getTargetTriple();
  return (strncmp(actualTarget.c_str(), triplePrefix,
                  strlen(triplePrefix)) == 0);
}


LTOModule::LTOModule(Module *m, TargetMachine *t)
  : _module(m), _target(t), _symbolsParsed(false)
{
}

LTOModule *LTOModule::makeLTOModule(const char *path,
                                    std::string &errMsg) {
  OwningPtr<MemoryBuffer> buffer(MemoryBuffer::getFile(path, &errMsg));
  if (!buffer)
    return NULL;
  return makeLTOModule(buffer.get(), errMsg);
}

/// makeBuffer - Create a MemoryBuffer from a memory range.  MemoryBuffer
/// requires the byte past end of the buffer to be a zero.  We might get lucky
/// and already be that way, otherwise make a copy.  Also if next byte is on a
/// different page, don't assume it is readable.
MemoryBuffer *LTOModule::makeBuffer(const void *mem, size_t length) {
  const char *startPtr = (char*)mem;
  const char *endPtr = startPtr+length;
  if (((uintptr_t)endPtr & (sys::Process::GetPageSize()-1)) == 0 ||
      *endPtr != 0)
    return MemoryBuffer::getMemBufferCopy(StringRef(startPtr, length));

  return MemoryBuffer::getMemBuffer(StringRef(startPtr, length));
}


LTOModule *LTOModule::makeLTOModule(const void *mem, size_t length,
                                    std::string &errMsg) {
  OwningPtr<MemoryBuffer> buffer(makeBuffer(mem, length));
  if (!buffer)
    return NULL;
  return makeLTOModule(buffer.get(), errMsg);
}

LTOModule *LTOModule::makeLTOModule(MemoryBuffer *buffer,
                                    std::string &errMsg) {
  InitializeAllTargets();

  // parse bitcode buffer
  OwningPtr<Module> m(ParseBitcodeFile(buffer, getGlobalContext(), &errMsg));
  if (!m)
    return NULL;

  std::string Triple = m->getTargetTriple();
  if (Triple.empty())
    Triple = sys::getHostTriple();

  // find machine architecture for this module
  const Target *march = TargetRegistry::lookupTarget(Triple, errMsg);
  if (!march)
    return NULL;

  // construct LTModule, hand over ownership of module and target
  SubtargetFeatures Features;
  Features.getDefaultSubtargetFeatures("" /* cpu */, llvm::Triple(Triple));
  std::string FeatureStr = Features.getString();
  TargetMachine *target = march->createTargetMachine(Triple, FeatureStr);
  return new LTOModule(m.take(), target);
}


const char *LTOModule::getTargetTriple() {
  return _module->getTargetTriple().c_str();
}

void LTOModule::setTargetTriple(const char *triple) {
  _module->setTargetTriple(triple);
}

void LTOModule::addDefinedFunctionSymbol(Function *f, Mangler &mangler) {
  // add to list of defined symbols
  addDefinedSymbol(f, mangler, true);

  // add external symbols referenced by this function.
  for (Function::iterator b = f->begin(); b != f->end(); ++b) {
    for (BasicBlock::iterator i = b->begin(); i != b->end(); ++i) {
      for (unsigned count = 0, total = i->getNumOperands();
           count != total; ++count) {
        findExternalRefs(i->getOperand(count), mangler);
      }
    }
  }
}

// Get string that data pointer points to.
bool LTOModule::objcClassNameFromExpression(Constant *c, std::string &name) {
  if (ConstantExpr *ce = dyn_cast<ConstantExpr>(c)) {
    Constant *op = ce->getOperand(0);
    if (GlobalVariable *gvn = dyn_cast<GlobalVariable>(op)) {
      Constant *cn = gvn->getInitializer();
      if (ConstantArray *ca = dyn_cast<ConstantArray>(cn)) {
        if (ca->isCString()) {
          name = ".objc_class_name_" + ca->getAsString();
          return true;
        }
      }
    }
  }
  return false;
}

// Parse i386/ppc ObjC class data structure.
void LTOModule::addObjCClass(GlobalVariable *clgv) {
  if (ConstantStruct *c = dyn_cast<ConstantStruct>(clgv->getInitializer())) {
    // second slot in __OBJC,__class is pointer to superclass name
    std::string superclassName;
    if (objcClassNameFromExpression(c->getOperand(1), superclassName)) {
      NameAndAttributes info;
      if (_undefines.find(superclassName.c_str()) == _undefines.end()) {
        const char *symbolName = ::strdup(superclassName.c_str());
        info.name = ::strdup(symbolName);
        info.attributes = LTO_SYMBOL_DEFINITION_UNDEFINED;
        // string is owned by _undefines
        _undefines[info.name] = info;
      }
    }
    // third slot in __OBJC,__class is pointer to class name
    std::string className;
    if (objcClassNameFromExpression(c->getOperand(2), className)) {
      const char *symbolName = ::strdup(className.c_str());
      NameAndAttributes info;
      info.name = symbolName;
      info.attributes = (lto_symbol_attributes)
        (LTO_SYMBOL_PERMISSIONS_DATA |
         LTO_SYMBOL_DEFINITION_REGULAR |
         LTO_SYMBOL_SCOPE_DEFAULT);
      _symbols.push_back(info);
      _defines[info.name] = 1;
    }
  }
}


// Parse i386/ppc ObjC category data structure.
void LTOModule::addObjCCategory(GlobalVariable *clgv) {
  if (ConstantStruct *c = dyn_cast<ConstantStruct>(clgv->getInitializer())) {
    // second slot in __OBJC,__category is pointer to target class name
    std::string targetclassName;
    if (objcClassNameFromExpression(c->getOperand(1), targetclassName)) {
      NameAndAttributes info;
      if (_undefines.find(targetclassName.c_str()) == _undefines.end()) {
        const char *symbolName = ::strdup(targetclassName.c_str());
        info.name = ::strdup(symbolName);
        info.attributes = LTO_SYMBOL_DEFINITION_UNDEFINED;
        // string is owned by _undefines
        _undefines[info.name] = info;
      }
    }
  }
}


// Parse i386/ppc ObjC class list data structure.
void LTOModule::addObjCClassRef(GlobalVariable *clgv) {
  std::string targetclassName;
  if (objcClassNameFromExpression(clgv->getInitializer(), targetclassName)) {
    NameAndAttributes info;
    if (_undefines.find(targetclassName.c_str()) == _undefines.end()) {
      const char *symbolName = ::strdup(targetclassName.c_str());
      info.name = ::strdup(symbolName);
      info.attributes = LTO_SYMBOL_DEFINITION_UNDEFINED;
      // string is owned by _undefines
      _undefines[info.name] = info;
    }
  }
}


void LTOModule::addDefinedDataSymbol(GlobalValue *v, Mangler &mangler) {
  // Add to list of defined symbols.
  addDefinedSymbol(v, mangler, false);

  // Special case i386/ppc ObjC data structures in magic sections:
  // The issue is that the old ObjC object format did some strange
  // contortions to avoid real linker symbols.  For instance, the
  // ObjC class data structure is allocated statically in the executable
  // that defines that class.  That data structures contains a pointer to
  // its superclass.  But instead of just initializing that part of the
  // struct to the address of its superclass, and letting the static and
  // dynamic linkers do the rest, the runtime works by having that field
  // instead point to a C-string that is the name of the superclass.
  // At runtime the objc initialization updates that pointer and sets
  // it to point to the actual super class.  As far as the linker
  // knows it is just a pointer to a string.  But then someone wanted the
  // linker to issue errors at build time if the superclass was not found.
  // So they figured out a way in mach-o object format to use an absolute
  // symbols (.objc_class_name_Foo = 0) and a floating reference
  // (.reference .objc_class_name_Bar) to cause the linker into erroring when
  // a class was missing.
  // The following synthesizes the implicit .objc_* symbols for the linker
  // from the ObjC data structures generated by the front end.
  if (v->hasSection() /* && isTargetDarwin */) {
    // special case if this data blob is an ObjC class definition
    if (v->getSection().compare(0, 15, "__OBJC,__class,") == 0) {
      if (GlobalVariable *gv = dyn_cast<GlobalVariable>(v)) {
        addObjCClass(gv);
      }
    }

    // special case if this data blob is an ObjC category definition
    else if (v->getSection().compare(0, 18, "__OBJC,__category,") == 0) {
      if (GlobalVariable *gv = dyn_cast<GlobalVariable>(v)) {
        addObjCCategory(gv);
      }
    }

    // special case if this data blob is the list of referenced classes
    else if (v->getSection().compare(0, 18, "__OBJC,__cls_refs,") == 0) {
      if (GlobalVariable *gv = dyn_cast<GlobalVariable>(v)) {
        addObjCClassRef(gv);
      }
    }
  }

  // add external symbols referenced by this data.
  for (unsigned count = 0, total = v->getNumOperands();
       count != total; ++count) {
    findExternalRefs(v->getOperand(count), mangler);
  }
}


void LTOModule::addDefinedSymbol(GlobalValue *def, Mangler &mangler,
                                 bool isFunction) {
  // ignore all llvm.* symbols
  if (def->getName().startswith("llvm."))
    return;

  // string is owned by _defines
  const char *symbolName = ::strdup(mangler.getNameWithPrefix(def).c_str());

  // set alignment part log2() can have rounding errors
  uint32_t align = def->getAlignment();
  uint32_t attr = align ? CountTrailingZeros_32(def->getAlignment()) : 0;

  // set permissions part
  if (isFunction)
    attr |= LTO_SYMBOL_PERMISSIONS_CODE;
  else {
    GlobalVariable *gv = dyn_cast<GlobalVariable>(def);
    if (gv && gv->isConstant())
      attr |= LTO_SYMBOL_PERMISSIONS_RODATA;
    else
      attr |= LTO_SYMBOL_PERMISSIONS_DATA;
  }

  // set definition part
  if (def->hasWeakLinkage() || def->hasLinkOnceLinkage()) {
    attr |= LTO_SYMBOL_DEFINITION_WEAK;
  }
  else if (def->hasCommonLinkage()) {
    attr |= LTO_SYMBOL_DEFINITION_TENTATIVE;
  }
  else {
    attr |= LTO_SYMBOL_DEFINITION_REGULAR;
  }

  // set scope part
  if (def->hasHiddenVisibility())
    attr |= LTO_SYMBOL_SCOPE_HIDDEN;
  else if (def->hasProtectedVisibility())
    attr |= LTO_SYMBOL_SCOPE_PROTECTED;
  else if (def->hasExternalLinkage() || def->hasWeakLinkage()
           || def->hasLinkOnceLinkage() || def->hasCommonLinkage())
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

void LTOModule::addAsmGlobalSymbol(const char *name) {
  // only add new define if not already defined
  if (_defines.count(name) == 0)
    return;

  // string is owned by _defines
  const char *symbolName = ::strdup(name);
  uint32_t attr = LTO_SYMBOL_DEFINITION_REGULAR;
  attr |= LTO_SYMBOL_SCOPE_DEFAULT;
  NameAndAttributes info;
  info.name = symbolName;
  info.attributes = (lto_symbol_attributes)attr;
  _symbols.push_back(info);
  _defines[info.name] = 1;
}

void LTOModule::addPotentialUndefinedSymbol(GlobalValue *decl,
                                            Mangler &mangler) {
  // ignore all llvm.* symbols
  if (decl->getName().startswith("llvm."))
    return;

  // ignore all aliases
  if (isa<GlobalAlias>(decl))
    return;

  std::string name = mangler.getNameWithPrefix(decl);

  // we already have the symbol
  if (_undefines.find(name) != _undefines.end())
    return;

  NameAndAttributes info;
  // string is owned by _undefines
  info.name = ::strdup(name.c_str());
  if (decl->hasExternalWeakLinkage())
    info.attributes = LTO_SYMBOL_DEFINITION_WEAKUNDEF;
  else
    info.attributes = LTO_SYMBOL_DEFINITION_UNDEFINED;
  _undefines[name] = info;
}



// Find external symbols referenced by VALUE. This is a recursive function.
void LTOModule::findExternalRefs(Value *value, Mangler &mangler) {
  if (GlobalValue *gv = dyn_cast<GlobalValue>(value)) {
    if (!gv->hasExternalLinkage())
      addPotentialUndefinedSymbol(gv, mangler);
    // If this is a variable definition, do not recursively process
    // initializer.  It might contain a reference to this variable
    // and cause an infinite loop.  The initializer will be
    // processed in addDefinedDataSymbol().
    return;
  }

  // GlobalValue, even with InternalLinkage type, may have operands with
  // ExternalLinkage type. Do not ignore these operands.
  if (Constant *c = dyn_cast<Constant>(value)) {
    // Handle ConstantExpr, ConstantStruct, ConstantArry etc.
    for (unsigned i = 0, e = c->getNumOperands(); i != e; ++i)
      findExternalRefs(c->getOperand(i), mangler);
  }
}

void LTOModule::lazyParseSymbols() {
  if (_symbolsParsed)
    return;

  _symbolsParsed = true;

  // Use mangler to add GlobalPrefix to names to match linker names.
  MCContext Context(*_target->getMCAsmInfo());
  Mangler mangler(Context, *_target->getTargetData());

  // add functions
  for (Module::iterator f = _module->begin(); f != _module->end(); ++f) {
    if (f->isDeclaration())
      addPotentialUndefinedSymbol(f, mangler);
    else
      addDefinedFunctionSymbol(f, mangler);
  }

  // add data
  for (Module::global_iterator v = _module->global_begin(),
         e = _module->global_end(); v !=  e; ++v) {
    if (v->isDeclaration())
      addPotentialUndefinedSymbol(v, mangler);
    else
      addDefinedDataSymbol(v, mangler);
  }

  // add asm globals
  const std::string &inlineAsm = _module->getModuleInlineAsm();
  const std::string glbl = ".globl";
  std::string asmSymbolName;
  std::string::size_type pos = inlineAsm.find(glbl, 0);
  while (pos != std::string::npos) {
    // eat .globl
    pos = pos + 6;

    // skip white space between .globl and symbol name
    std::string::size_type pbegin = inlineAsm.find_first_not_of(' ', pos);
    if (pbegin == std::string::npos)
      break;

    // find end-of-line
    std::string::size_type pend = inlineAsm.find_first_of('\n', pbegin);
    if (pend == std::string::npos)
      break;

    asmSymbolName.assign(inlineAsm, pbegin, pend - pbegin);
    addAsmGlobalSymbol(asmSymbolName.c_str());

    // search next .globl
    pos = inlineAsm.find(glbl, pend);
  }

  // make symbols for all undefines
  for (StringMap<NameAndAttributes>::iterator it=_undefines.begin();
       it != _undefines.end(); ++it) {
    // if this symbol also has a definition, then don't make an undefine
    // because it is a tentative definition
    if (_defines.count(it->getKey()) == 0) {
      NameAndAttributes info = it->getValue();
      _symbols.push_back(info);
    }
  }
}


uint32_t LTOModule::getSymbolCount() {
  lazyParseSymbols();
  return _symbols.size();
}


lto_symbol_attributes LTOModule::getSymbolAttributes(uint32_t index) {
  lazyParseSymbols();
  if (index < _symbols.size())
    return _symbols[index].attributes;
  else
    return lto_symbol_attributes(0);
}

const char *LTOModule::getSymbolName(uint32_t index) {
  lazyParseSymbols();
  if (index < _symbols.size())
    return _symbols[index].name;
  else
    return NULL;
}
