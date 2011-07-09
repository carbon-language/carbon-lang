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
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/system_error.h"
#include "llvm/Target/Mangler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Target/TargetAsmParser.h"
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
  OwningPtr<MemoryBuffer> buffer;
  if (MemoryBuffer::getFile(path, buffer))
    return false;
  return isTargetMatch(buffer.take(), triplePrefix);
}

// Takes ownership of buffer.
bool LTOModule::isTargetMatch(MemoryBuffer *buffer, const char *triplePrefix) {
  std::string Triple = getBitcodeTargetTriple(buffer, getGlobalContext());
  delete buffer;
  return (strncmp(Triple.c_str(), triplePrefix,
 		  strlen(triplePrefix)) == 0);
}


LTOModule::LTOModule(Module *m, TargetMachine *t)
  : _module(m), _target(t)
{
}

LTOModule *LTOModule::makeLTOModule(const char *path,
                                    std::string &errMsg) {
  OwningPtr<MemoryBuffer> buffer;
  if (error_code ec = MemoryBuffer::getFile(path, buffer)) {
    errMsg = ec.message();
    return NULL;
  }
  return makeLTOModule(buffer.take(), errMsg);
}

LTOModule *LTOModule::makeLTOModule(int fd, const char *path,
                                    size_t size,
                                    std::string &errMsg) {
  return makeLTOModule(fd, path, size, size, 0, errMsg);
}

LTOModule *LTOModule::makeLTOModule(int fd, const char *path,
                                    size_t file_size,
                                    size_t map_size,
                                    off_t offset,
                                    std::string &errMsg) {
  OwningPtr<MemoryBuffer> buffer;
  if (error_code ec = MemoryBuffer::getOpenFile(fd, path, buffer, file_size,
                                                map_size, offset, false)) {
    errMsg = ec.message();
    return NULL;
  }
  return makeLTOModule(buffer.take(), errMsg);
}

/// makeBuffer - Create a MemoryBuffer from a memory range.
MemoryBuffer *LTOModule::makeBuffer(const void *mem, size_t length) {
  const char *startPtr = (char*)mem;
  return MemoryBuffer::getMemBuffer(StringRef(startPtr, length), "", false);
}


LTOModule *LTOModule::makeLTOModule(const void *mem, size_t length,
                                    std::string &errMsg) {
  OwningPtr<MemoryBuffer> buffer(makeBuffer(mem, length));
  if (!buffer)
    return NULL;
  return makeLTOModule(buffer.take(), errMsg);
}

LTOModule *LTOModule::makeLTOModule(MemoryBuffer *buffer,
                                    std::string &errMsg) {
  static bool Initialized = false;
  if (!Initialized) {
    InitializeAllTargets();
    InitializeAllAsmParsers();
    Initialized = true;
  }

  // parse bitcode buffer
  OwningPtr<Module> m(getLazyBitcodeModule(buffer, getGlobalContext(),
                                           &errMsg));
  if (!m) {
    delete buffer;
    return NULL;
  }

  std::string Triple = m->getTargetTriple();
  if (Triple.empty())
    Triple = sys::getHostTriple();

  // find machine architecture for this module
  const Target *march = TargetRegistry::lookupTarget(Triple, errMsg);
  if (!march)
    return NULL;

  // construct LTOModule, hand over ownership of module and target
  SubtargetFeatures Features;
  Features.getDefaultSubtargetFeatures(llvm::Triple(Triple));
  std::string FeatureStr = Features.getString();
  std::string CPU;
  TargetMachine *target = march->createTargetMachine(Triple, CPU, FeatureStr);
  LTOModule *Ret = new LTOModule(m.take(), target);
  bool Err = Ret->ParseSymbols();
  if (Err) {
    delete Ret;
    return NULL;
  }
  return Ret;
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
}

// Get string that data pointer points to.
bool LTOModule::objcClassNameFromExpression(Constant *c, std::string &name) {
  if (ConstantExpr *ce = dyn_cast<ConstantExpr>(c)) {
    Constant *op = ce->getOperand(0);
    if (GlobalVariable *gvn = dyn_cast<GlobalVariable>(op)) {
      Constant *cn = gvn->getInitializer();
      if (ConstantArray *ca = dyn_cast<ConstantArray>(cn)) {
        if (ca->isCString()) {
          name = ".objc_class_name_" + ca->getAsCString();
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
      StringMap<NameAndAttributes>::value_type &entry =
        _undefines.GetOrCreateValue(superclassName);
      if (!entry.getValue().name) {
        const char *symbolName = entry.getKey().data();
        info.name = symbolName;
        info.attributes = LTO_SYMBOL_DEFINITION_UNDEFINED;
        entry.setValue(info);
      }
    }
    // third slot in __OBJC,__class is pointer to class name
    std::string className;
    if (objcClassNameFromExpression(c->getOperand(2), className)) {
      StringSet::value_type &entry =
        _defines.GetOrCreateValue(className);
      entry.setValue(1);
      NameAndAttributes info;
      info.name = entry.getKey().data();
      info.attributes = (lto_symbol_attributes)
        (LTO_SYMBOL_PERMISSIONS_DATA |
         LTO_SYMBOL_DEFINITION_REGULAR |
         LTO_SYMBOL_SCOPE_DEFAULT);
      _symbols.push_back(info);
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

      StringMap<NameAndAttributes>::value_type &entry =
        _undefines.GetOrCreateValue(targetclassName);

      if (entry.getValue().name)
        return;

      const char *symbolName = entry.getKey().data();
      info.name = symbolName;
      info.attributes = LTO_SYMBOL_DEFINITION_UNDEFINED;
      entry.setValue(info);
    }
  }
}


// Parse i386/ppc ObjC class list data structure.
void LTOModule::addObjCClassRef(GlobalVariable *clgv) {
  std::string targetclassName;
  if (objcClassNameFromExpression(clgv->getInitializer(), targetclassName)) {
    NameAndAttributes info;

    StringMap<NameAndAttributes>::value_type &entry =
      _undefines.GetOrCreateValue(targetclassName);
    if (entry.getValue().name)
      return;

    const char *symbolName = entry.getKey().data();
    info.name = symbolName;
    info.attributes = LTO_SYMBOL_DEFINITION_UNDEFINED;
    entry.setValue(info);
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
}


void LTOModule::addDefinedSymbol(GlobalValue *def, Mangler &mangler,
                                 bool isFunction) {
  // ignore all llvm.* symbols
  if (def->getName().startswith("llvm."))
    return;

  // string is owned by _defines
  SmallString<64> Buffer;
  mangler.getNameWithPrefix(Buffer, def, false);

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
  if (def->hasWeakLinkage() || def->hasLinkOnceLinkage() ||
      def->hasLinkerPrivateWeakLinkage() ||
      def->hasLinkerPrivateWeakDefAutoLinkage())
    attr |= LTO_SYMBOL_DEFINITION_WEAK;
  else if (def->hasCommonLinkage())
    attr |= LTO_SYMBOL_DEFINITION_TENTATIVE;
  else
    attr |= LTO_SYMBOL_DEFINITION_REGULAR;

  // set scope part
  if (def->hasHiddenVisibility())
    attr |= LTO_SYMBOL_SCOPE_HIDDEN;
  else if (def->hasProtectedVisibility())
    attr |= LTO_SYMBOL_SCOPE_PROTECTED;
  else if (def->hasExternalLinkage() || def->hasWeakLinkage() ||
           def->hasLinkOnceLinkage() || def->hasCommonLinkage() ||
           def->hasLinkerPrivateWeakLinkage())
    attr |= LTO_SYMBOL_SCOPE_DEFAULT;
  else if (def->hasLinkerPrivateWeakDefAutoLinkage())
    attr |= LTO_SYMBOL_SCOPE_DEFAULT_CAN_BE_HIDDEN;
  else
    attr |= LTO_SYMBOL_SCOPE_INTERNAL;

  // add to table of symbols
  NameAndAttributes info;
  StringSet::value_type &entry = _defines.GetOrCreateValue(Buffer);
  entry.setValue(1);

  StringRef Name = entry.getKey();
  info.name = Name.data();
  assert(info.name[Name.size()] == '\0');
  info.attributes = (lto_symbol_attributes)attr;
  _symbols.push_back(info);
}

void LTOModule::addAsmGlobalSymbol(const char *name,
                                   lto_symbol_attributes scope) {
  StringSet::value_type &entry = _defines.GetOrCreateValue(name);

  // only add new define if not already defined
  if (entry.getValue())
    return;

  entry.setValue(1);
  const char *symbolName = entry.getKey().data();
  uint32_t attr = LTO_SYMBOL_DEFINITION_REGULAR;
  attr |= scope;
  NameAndAttributes info;
  info.name = symbolName;
  info.attributes = (lto_symbol_attributes)attr;
  _symbols.push_back(info);
}

void LTOModule::addAsmGlobalSymbolUndef(const char *name) {
  StringMap<NameAndAttributes>::value_type &entry =
    _undefines.GetOrCreateValue(name);

  _asm_undefines.push_back(entry.getKey().data());

  // we already have the symbol
  if (entry.getValue().name)
    return;

  uint32_t attr = LTO_SYMBOL_DEFINITION_UNDEFINED;;
  attr |= LTO_SYMBOL_SCOPE_DEFAULT;
  NameAndAttributes info;
  info.name = entry.getKey().data();
  info.attributes = (lto_symbol_attributes)attr;

  entry.setValue(info);
}

void LTOModule::addPotentialUndefinedSymbol(GlobalValue *decl,
                                            Mangler &mangler) {
  // ignore all llvm.* symbols
  if (decl->getName().startswith("llvm."))
    return;

  // ignore all aliases
  if (isa<GlobalAlias>(decl))
    return;

  SmallString<64> name;
  mangler.getNameWithPrefix(name, decl, false);

  StringMap<NameAndAttributes>::value_type &entry =
    _undefines.GetOrCreateValue(name);

  // we already have the symbol
  if (entry.getValue().name)
    return;

  NameAndAttributes info;

  info.name = entry.getKey().data();
  if (decl->hasExternalWeakLinkage())
    info.attributes = LTO_SYMBOL_DEFINITION_WEAKUNDEF;
  else
    info.attributes = LTO_SYMBOL_DEFINITION_UNDEFINED;

  entry.setValue(info);
}


namespace {
  class RecordStreamer : public MCStreamer {
  public:
    enum State { NeverSeen, Global, Defined, DefinedGlobal, Used};

  private:
    StringMap<State> Symbols;

    void markDefined(const MCSymbol &Symbol) {
      State &S = Symbols[Symbol.getName()];
      switch (S) {
      case DefinedGlobal:
      case Global:
        S = DefinedGlobal;
        break;
      case NeverSeen:
      case Defined:
      case Used:
        S = Defined;
        break;
      }
    }
    void markGlobal(const MCSymbol &Symbol) {
      State &S = Symbols[Symbol.getName()];
      switch (S) {
      case DefinedGlobal:
      case Defined:
        S = DefinedGlobal;
        break;

      case NeverSeen:
      case Global:
      case Used:
        S = Global;
        break;
      }
    }
    void markUsed(const MCSymbol &Symbol) {
      State &S = Symbols[Symbol.getName()];
      switch (S) {
      case DefinedGlobal:
      case Defined:
      case Global:
        break;

      case NeverSeen:
      case Used:
        S = Used;
        break;
      }
    }

    // FIXME: mostly copied for the obj streamer.
    void AddValueSymbols(const MCExpr *Value) {
      switch (Value->getKind()) {
      case MCExpr::Target:
        // FIXME: What should we do in here?
        break;

      case MCExpr::Constant:
        break;

      case MCExpr::Binary: {
        const MCBinaryExpr *BE = cast<MCBinaryExpr>(Value);
        AddValueSymbols(BE->getLHS());
        AddValueSymbols(BE->getRHS());
        break;
      }

      case MCExpr::SymbolRef:
        markUsed(cast<MCSymbolRefExpr>(Value)->getSymbol());
        break;

      case MCExpr::Unary:
        AddValueSymbols(cast<MCUnaryExpr>(Value)->getSubExpr());
        break;
      }
    }

  public:
    typedef StringMap<State>::const_iterator const_iterator;

    const_iterator begin() {
      return Symbols.begin();
    }

    const_iterator end() {
      return Symbols.end();
    }

    RecordStreamer(MCContext &Context) : MCStreamer(Context) {}

    virtual void ChangeSection(const MCSection *Section) {}
    virtual void InitSections() {}
    virtual void EmitLabel(MCSymbol *Symbol) {
      Symbol->setSection(*getCurrentSection());
      markDefined(*Symbol);
    }
    virtual void EmitAssemblerFlag(MCAssemblerFlag Flag) {}
    virtual void EmitThumbFunc(MCSymbol *Func) {}
    virtual void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) {
      // FIXME: should we handle aliases?
      markDefined(*Symbol);
    }
    virtual void EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) {
      if (Attribute == MCSA_Global)
        markGlobal(*Symbol);
    }
    virtual void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {}
    virtual void EmitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol) {}
    virtual void BeginCOFFSymbolDef(const MCSymbol *Symbol) {}
    virtual void EmitCOFFSymbolStorageClass(int StorageClass) {}
    virtual void EmitZerofill(const MCSection *Section, MCSymbol *Symbol,
                              unsigned Size , unsigned ByteAlignment) {
      markDefined(*Symbol);
    }
    virtual void EmitCOFFSymbolType(int Type) {}
    virtual void EndCOFFSymbolDef() {}
    virtual void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                  unsigned ByteAlignment) {
      markDefined(*Symbol);
    }
    virtual void EmitELFSize(MCSymbol *Symbol, const MCExpr *Value) {}
    virtual void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size) {}
    virtual void EmitTBSSSymbol(const MCSection *Section, MCSymbol *Symbol,
                                uint64_t Size, unsigned ByteAlignment) {}
    virtual void EmitBytes(StringRef Data, unsigned AddrSpace) {}
    virtual void EmitValueImpl(const MCExpr *Value, unsigned Size,
                               unsigned AddrSpace) {}
    virtual void EmitULEB128Value(const MCExpr *Value) {}
    virtual void EmitSLEB128Value(const MCExpr *Value) {}
    virtual void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value,
                                      unsigned ValueSize,
                                      unsigned MaxBytesToEmit) {}
    virtual void EmitCodeAlignment(unsigned ByteAlignment,
                                   unsigned MaxBytesToEmit) {}
    virtual void EmitValueToOffset(const MCExpr *Offset,
                                   unsigned char Value ) {}
    virtual void EmitFileDirective(StringRef Filename) {}
    virtual void EmitDwarfAdvanceLineAddr(int64_t LineDelta,
                                          const MCSymbol *LastLabel,
                                        const MCSymbol *Label) {}

    virtual void EmitInstruction(const MCInst &Inst) {
      // Scan for values.
      for (unsigned i = Inst.getNumOperands(); i--; )
        if (Inst.getOperand(i).isExpr())
          AddValueSymbols(Inst.getOperand(i).getExpr());
    }
    virtual void Finish() {}
  };
}

bool LTOModule::addAsmGlobalSymbols(MCContext &Context) {
  const std::string &inlineAsm = _module->getModuleInlineAsm();

  OwningPtr<RecordStreamer> Streamer(new RecordStreamer(Context));
  MemoryBuffer *Buffer = MemoryBuffer::getMemBuffer(inlineAsm);
  SourceMgr SrcMgr;
  SrcMgr.AddNewSourceBuffer(Buffer, SMLoc());
  OwningPtr<MCAsmParser> Parser(createMCAsmParser(_target->getTarget(), SrcMgr,
                                                  Context, *Streamer,
                                                  *_target->getMCAsmInfo()));
  OwningPtr<MCSubtargetInfo> STI(_target->getTarget().
                      createMCSubtargetInfo(_target->getTargetTriple(),
                                            _target->getTargetCPU(),
                                            _target->getTargetFeatureString()));
  OwningPtr<TargetAsmParser>
    TAP(_target->getTarget().createAsmParser(*STI, *Parser.get()));
  Parser->setTargetParser(*TAP);
  int Res = Parser->Run(false);
  if (Res)
    return true;

  for (RecordStreamer::const_iterator i = Streamer->begin(),
         e = Streamer->end(); i != e; ++i) {
    StringRef Key = i->first();
    RecordStreamer::State Value = i->second;
    if (Value == RecordStreamer::DefinedGlobal)
      addAsmGlobalSymbol(Key.data(), LTO_SYMBOL_SCOPE_DEFAULT);
    else if (Value == RecordStreamer::Defined)
      addAsmGlobalSymbol(Key.data(), LTO_SYMBOL_SCOPE_INTERNAL);
    else if (Value == RecordStreamer::Global ||
             Value == RecordStreamer::Used)
      addAsmGlobalSymbolUndef(Key.data());
  }
  return false;
}

static bool isDeclaration(const GlobalValue &V) {
  if (V.hasAvailableExternallyLinkage())
    return true;
  if (V.isMaterializable())
    return false;
  return V.isDeclaration();
}

static bool isAliasToDeclaration(const GlobalAlias &V) {
  return isDeclaration(*V.getAliasedGlobal());
}

bool LTOModule::ParseSymbols() {
  // Use mangler to add GlobalPrefix to names to match linker names.
  MCContext Context(*_target->getMCAsmInfo(), NULL);
  Mangler mangler(Context, *_target->getTargetData());

  // add functions
  for (Module::iterator f = _module->begin(); f != _module->end(); ++f) {
    if (isDeclaration(*f))
      addPotentialUndefinedSymbol(f, mangler);
    else
      addDefinedFunctionSymbol(f, mangler);
  }

  // add data
  for (Module::global_iterator v = _module->global_begin(),
         e = _module->global_end(); v !=  e; ++v) {
    if (isDeclaration(*v))
      addPotentialUndefinedSymbol(v, mangler);
    else
      addDefinedDataSymbol(v, mangler);
  }

  // add asm globals
  if (addAsmGlobalSymbols(Context))
    return true;

  // add aliases
  for (Module::alias_iterator i = _module->alias_begin(),
         e = _module->alias_end(); i != e; ++i) {
    if (isAliasToDeclaration(*i))
      addPotentialUndefinedSymbol(i, mangler);
    else
      addDefinedDataSymbol(i, mangler);
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
  return false;
}


uint32_t LTOModule::getSymbolCount() {
  return _symbols.size();
}


lto_symbol_attributes LTOModule::getSymbolAttributes(uint32_t index) {
  if (index < _symbols.size())
    return _symbols[index].attributes;
  else
    return lto_symbol_attributes(0);
}

const char *LTOModule::getSymbolName(uint32_t index) {
  if (index < _symbols.size())
    return _symbols[index].name;
  else
    return NULL;
}
