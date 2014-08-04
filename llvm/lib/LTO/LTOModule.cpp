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

#include "llvm/LTO/LTOModule.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Transforms/Utils/GlobalStatus.h"
#include <system_error>
using namespace llvm;

LTOModule::LTOModule(std::unique_ptr<object::IRObjectFile> Obj,
                     llvm::TargetMachine *TM)
    : IRFile(std::move(Obj)), _target(TM) {}

/// isBitcodeFile - Returns 'true' if the file (or memory contents) is LLVM
/// bitcode.
bool LTOModule::isBitcodeFile(const void *mem, size_t length) {
  return sys::fs::identify_magic(StringRef((const char *)mem, length)) ==
         sys::fs::file_magic::bitcode;
}

bool LTOModule::isBitcodeFile(const char *path) {
  sys::fs::file_magic type;
  if (sys::fs::identify_magic(path, type))
    return false;
  return type == sys::fs::file_magic::bitcode;
}

bool LTOModule::isBitcodeForTarget(MemoryBuffer *buffer,
                                   StringRef triplePrefix) {
  std::string Triple = getBitcodeTargetTriple(buffer, getGlobalContext());
  return StringRef(Triple).startswith(triplePrefix);
}

LTOModule *LTOModule::createFromFile(const char *path, TargetOptions options,
                                     std::string &errMsg) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(path);
  if (std::error_code EC = BufferOrErr.getError()) {
    errMsg = EC.message();
    return nullptr;
  }
  return makeLTOModule(std::move(BufferOrErr.get()), options, errMsg);
}

LTOModule *LTOModule::createFromOpenFile(int fd, const char *path, size_t size,
                                         TargetOptions options,
                                         std::string &errMsg) {
  return createFromOpenFileSlice(fd, path, size, 0, options, errMsg);
}

LTOModule *LTOModule::createFromOpenFileSlice(int fd, const char *path,
                                              size_t map_size, off_t offset,
                                              TargetOptions options,
                                              std::string &errMsg) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getOpenFileSlice(fd, path, map_size, offset);
  if (std::error_code EC = BufferOrErr.getError()) {
    errMsg = EC.message();
    return nullptr;
  }
  return makeLTOModule(std::move(BufferOrErr.get()), options, errMsg);
}

LTOModule *LTOModule::createFromBuffer(const void *mem, size_t length,
                                       TargetOptions options,
                                       std::string &errMsg, StringRef path) {
  std::unique_ptr<MemoryBuffer> buffer(makeBuffer(mem, length, path));
  if (!buffer)
    return nullptr;
  return makeLTOModule(std::move(buffer), options, errMsg);
}

LTOModule *LTOModule::makeLTOModule(std::unique_ptr<MemoryBuffer> Buffer,
                                    TargetOptions options,
                                    std::string &errMsg) {
  ErrorOr<Module *> MOrErr =
      getLazyBitcodeModule(Buffer.get(), getGlobalContext());
  if (std::error_code EC = MOrErr.getError()) {
    errMsg = EC.message();
    return nullptr;
  }
  std::unique_ptr<Module> M(MOrErr.get());

  std::string TripleStr = M->getTargetTriple();
  if (TripleStr.empty())
    TripleStr = sys::getDefaultTargetTriple();
  llvm::Triple Triple(TripleStr);

  // find machine architecture for this module
  const Target *march = TargetRegistry::lookupTarget(TripleStr, errMsg);
  if (!march)
    return nullptr;

  // construct LTOModule, hand over ownership of module and target
  SubtargetFeatures Features;
  Features.getDefaultSubtargetFeatures(Triple);
  std::string FeatureStr = Features.getString();
  // Set a default CPU for Darwin triples.
  std::string CPU;
  if (Triple.isOSDarwin()) {
    if (Triple.getArch() == llvm::Triple::x86_64)
      CPU = "core2";
    else if (Triple.getArch() == llvm::Triple::x86)
      CPU = "yonah";
    else if (Triple.getArch() == llvm::Triple::aarch64)
      CPU = "cyclone";
  }

  TargetMachine *target = march->createTargetMachine(TripleStr, CPU, FeatureStr,
                                                     options);
  M->materializeAllPermanently(true);
  M->setDataLayout(target->getSubtargetImpl()->getDataLayout());

  std::unique_ptr<object::IRObjectFile> IRObj(
      new object::IRObjectFile(std::move(Buffer), std::move(M)));

  LTOModule *Ret = new LTOModule(std::move(IRObj), target);

  if (Ret->parseSymbols(errMsg)) {
    delete Ret;
    return nullptr;
  }

  Ret->parseMetadata();

  return Ret;
}

/// Create a MemoryBuffer from a memory range with an optional name.
MemoryBuffer *LTOModule::makeBuffer(const void *mem, size_t length,
                                    StringRef name) {
  const char *startPtr = (const char*)mem;
  return MemoryBuffer::getMemBuffer(StringRef(startPtr, length), name, false);
}

/// objcClassNameFromExpression - Get string that the data pointer points to.
bool
LTOModule::objcClassNameFromExpression(const Constant *c, std::string &name) {
  if (const ConstantExpr *ce = dyn_cast<ConstantExpr>(c)) {
    Constant *op = ce->getOperand(0);
    if (GlobalVariable *gvn = dyn_cast<GlobalVariable>(op)) {
      Constant *cn = gvn->getInitializer();
      if (ConstantDataArray *ca = dyn_cast<ConstantDataArray>(cn)) {
        if (ca->isCString()) {
          name = ".objc_class_name_" + ca->getAsCString().str();
          return true;
        }
      }
    }
  }
  return false;
}

/// addObjCClass - Parse i386/ppc ObjC class data structure.
void LTOModule::addObjCClass(const GlobalVariable *clgv) {
  const ConstantStruct *c = dyn_cast<ConstantStruct>(clgv->getInitializer());
  if (!c) return;

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
      info.isFunction = false;
      info.symbol = clgv;
      entry.setValue(info);
    }
  }

  // third slot in __OBJC,__class is pointer to class name
  std::string className;
  if (objcClassNameFromExpression(c->getOperand(2), className)) {
    StringSet::value_type &entry = _defines.GetOrCreateValue(className);
    entry.setValue(1);

    NameAndAttributes info;
    info.name = entry.getKey().data();
    info.attributes = LTO_SYMBOL_PERMISSIONS_DATA |
      LTO_SYMBOL_DEFINITION_REGULAR | LTO_SYMBOL_SCOPE_DEFAULT;
    info.isFunction = false;
    info.symbol = clgv;
    _symbols.push_back(info);
  }
}

/// addObjCCategory - Parse i386/ppc ObjC category data structure.
void LTOModule::addObjCCategory(const GlobalVariable *clgv) {
  const ConstantStruct *c = dyn_cast<ConstantStruct>(clgv->getInitializer());
  if (!c) return;

  // second slot in __OBJC,__category is pointer to target class name
  std::string targetclassName;
  if (!objcClassNameFromExpression(c->getOperand(1), targetclassName))
    return;

  NameAndAttributes info;
  StringMap<NameAndAttributes>::value_type &entry =
    _undefines.GetOrCreateValue(targetclassName);

  if (entry.getValue().name)
    return;

  const char *symbolName = entry.getKey().data();
  info.name = symbolName;
  info.attributes = LTO_SYMBOL_DEFINITION_UNDEFINED;
  info.isFunction = false;
  info.symbol = clgv;
  entry.setValue(info);
}

/// addObjCClassRef - Parse i386/ppc ObjC class list data structure.
void LTOModule::addObjCClassRef(const GlobalVariable *clgv) {
  std::string targetclassName;
  if (!objcClassNameFromExpression(clgv->getInitializer(), targetclassName))
    return;

  NameAndAttributes info;
  StringMap<NameAndAttributes>::value_type &entry =
    _undefines.GetOrCreateValue(targetclassName);
  if (entry.getValue().name)
    return;

  const char *symbolName = entry.getKey().data();
  info.name = symbolName;
  info.attributes = LTO_SYMBOL_DEFINITION_UNDEFINED;
  info.isFunction = false;
  info.symbol = clgv;
  entry.setValue(info);
}

void LTOModule::addDefinedDataSymbol(const object::BasicSymbolRef &Sym) {
  SmallString<64> Buffer;
  {
    raw_svector_ostream OS(Buffer);
    Sym.printName(OS);
  }

  const GlobalValue *V = IRFile->getSymbolGV(Sym.getRawDataRefImpl());
  addDefinedDataSymbol(Buffer.c_str(), V);
}

void LTOModule::addDefinedDataSymbol(const char *Name, const GlobalValue *v) {
  // Add to list of defined symbols.
  addDefinedSymbol(Name, v, false);

  if (!v->hasSection() /* || !isTargetDarwin */)
    return;

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

  // special case if this data blob is an ObjC class definition
  std::string Section = v->getSection();
  if (Section.compare(0, 15, "__OBJC,__class,") == 0) {
    if (const GlobalVariable *gv = dyn_cast<GlobalVariable>(v)) {
      addObjCClass(gv);
    }
  }

  // special case if this data blob is an ObjC category definition
  else if (Section.compare(0, 18, "__OBJC,__category,") == 0) {
    if (const GlobalVariable *gv = dyn_cast<GlobalVariable>(v)) {
      addObjCCategory(gv);
    }
  }

  // special case if this data blob is the list of referenced classes
  else if (Section.compare(0, 18, "__OBJC,__cls_refs,") == 0) {
    if (const GlobalVariable *gv = dyn_cast<GlobalVariable>(v)) {
      addObjCClassRef(gv);
    }
  }
}

void LTOModule::addDefinedFunctionSymbol(const object::BasicSymbolRef &Sym) {
  SmallString<64> Buffer;
  {
    raw_svector_ostream OS(Buffer);
    Sym.printName(OS);
  }

  const Function *F =
      cast<Function>(IRFile->getSymbolGV(Sym.getRawDataRefImpl()));
  addDefinedFunctionSymbol(Buffer.c_str(), F);
}

void LTOModule::addDefinedFunctionSymbol(const char *Name, const Function *F) {
  // add to list of defined symbols
  addDefinedSymbol(Name, F, true);
}

void LTOModule::addDefinedSymbol(const char *Name, const GlobalValue *def,
                                 bool isFunction) {
  // set alignment part log2() can have rounding errors
  uint32_t align = def->getAlignment();
  uint32_t attr = align ? countTrailingZeros(align) : 0;

  // set permissions part
  if (isFunction) {
    attr |= LTO_SYMBOL_PERMISSIONS_CODE;
  } else {
    const GlobalVariable *gv = dyn_cast<GlobalVariable>(def);
    if (gv && gv->isConstant())
      attr |= LTO_SYMBOL_PERMISSIONS_RODATA;
    else
      attr |= LTO_SYMBOL_PERMISSIONS_DATA;
  }

  // set definition part
  if (def->hasWeakLinkage() || def->hasLinkOnceLinkage())
    attr |= LTO_SYMBOL_DEFINITION_WEAK;
  else if (def->hasCommonLinkage())
    attr |= LTO_SYMBOL_DEFINITION_TENTATIVE;
  else
    attr |= LTO_SYMBOL_DEFINITION_REGULAR;

  // set scope part
  if (def->hasLocalLinkage())
    // Ignore visibility if linkage is local.
    attr |= LTO_SYMBOL_SCOPE_INTERNAL;
  else if (def->hasHiddenVisibility())
    attr |= LTO_SYMBOL_SCOPE_HIDDEN;
  else if (def->hasProtectedVisibility())
    attr |= LTO_SYMBOL_SCOPE_PROTECTED;
  else if (canBeOmittedFromSymbolTable(def))
    attr |= LTO_SYMBOL_SCOPE_DEFAULT_CAN_BE_HIDDEN;
  else
    attr |= LTO_SYMBOL_SCOPE_DEFAULT;

  StringSet::value_type &entry = _defines.GetOrCreateValue(Name);
  entry.setValue(1);

  // fill information structure
  NameAndAttributes info;
  StringRef NameRef = entry.getKey();
  info.name = NameRef.data();
  assert(info.name[NameRef.size()] == '\0');
  info.attributes = attr;
  info.isFunction = isFunction;
  info.symbol = def;

  // add to table of symbols
  _symbols.push_back(info);
}

/// addAsmGlobalSymbol - Add a global symbol from module-level ASM to the
/// defined list.
void LTOModule::addAsmGlobalSymbol(const char *name,
                                   lto_symbol_attributes scope) {
  StringSet::value_type &entry = _defines.GetOrCreateValue(name);

  // only add new define if not already defined
  if (entry.getValue())
    return;

  entry.setValue(1);

  NameAndAttributes &info = _undefines[entry.getKey().data()];

  if (info.symbol == nullptr) {
    // FIXME: This is trying to take care of module ASM like this:
    //
    //   module asm ".zerofill __FOO, __foo, _bar_baz_qux, 0"
    //
    // but is gross and its mother dresses it funny. Have the ASM parser give us
    // more details for this type of situation so that we're not guessing so
    // much.

    // fill information structure
    info.name = entry.getKey().data();
    info.attributes =
      LTO_SYMBOL_PERMISSIONS_DATA | LTO_SYMBOL_DEFINITION_REGULAR | scope;
    info.isFunction = false;
    info.symbol = nullptr;

    // add to table of symbols
    _symbols.push_back(info);
    return;
  }

  if (info.isFunction)
    addDefinedFunctionSymbol(info.name, cast<Function>(info.symbol));
  else
    addDefinedDataSymbol(info.name, info.symbol);

  _symbols.back().attributes &= ~LTO_SYMBOL_SCOPE_MASK;
  _symbols.back().attributes |= scope;
}

/// addAsmGlobalSymbolUndef - Add a global symbol from module-level ASM to the
/// undefined list.
void LTOModule::addAsmGlobalSymbolUndef(const char *name) {
  StringMap<NameAndAttributes>::value_type &entry =
    _undefines.GetOrCreateValue(name);

  _asm_undefines.push_back(entry.getKey().data());

  // we already have the symbol
  if (entry.getValue().name)
    return;

  uint32_t attr = LTO_SYMBOL_DEFINITION_UNDEFINED;
  attr |= LTO_SYMBOL_SCOPE_DEFAULT;
  NameAndAttributes info;
  info.name = entry.getKey().data();
  info.attributes = attr;
  info.isFunction = false;
  info.symbol = nullptr;

  entry.setValue(info);
}

/// Add a symbol which isn't defined just yet to a list to be resolved later.
void LTOModule::addPotentialUndefinedSymbol(const object::BasicSymbolRef &Sym,
                                            bool isFunc) {
  SmallString<64> name;
  {
    raw_svector_ostream OS(name);
    Sym.printName(OS);
  }

  StringMap<NameAndAttributes>::value_type &entry =
    _undefines.GetOrCreateValue(name);

  // we already have the symbol
  if (entry.getValue().name)
    return;

  NameAndAttributes info;

  info.name = entry.getKey().data();

  const GlobalValue *decl = IRFile->getSymbolGV(Sym.getRawDataRefImpl());

  if (decl->hasExternalWeakLinkage())
    info.attributes = LTO_SYMBOL_DEFINITION_WEAKUNDEF;
  else
    info.attributes = LTO_SYMBOL_DEFINITION_UNDEFINED;

  info.isFunction = isFunc;
  info.symbol = decl;

  entry.setValue(info);
}

/// parseSymbols - Parse the symbols from the module and model-level ASM and add
/// them to either the defined or undefined lists.
bool LTOModule::parseSymbols(std::string &errMsg) {
  for (auto &Sym : IRFile->symbols()) {
    const GlobalValue *GV = IRFile->getSymbolGV(Sym.getRawDataRefImpl());
    uint32_t Flags = Sym.getFlags();
    if (Flags & object::BasicSymbolRef::SF_FormatSpecific)
      continue;

    bool IsUndefined = Flags & object::BasicSymbolRef::SF_Undefined;

    if (!GV) {
      SmallString<64> Buffer;
      {
        raw_svector_ostream OS(Buffer);
        Sym.printName(OS);
      }
      const char *Name = Buffer.c_str();

      if (IsUndefined)
        addAsmGlobalSymbolUndef(Name);
      else if (Flags & object::BasicSymbolRef::SF_Global)
        addAsmGlobalSymbol(Name, LTO_SYMBOL_SCOPE_DEFAULT);
      else
        addAsmGlobalSymbol(Name, LTO_SYMBOL_SCOPE_INTERNAL);
      continue;
    }

    auto *F = dyn_cast<Function>(GV);
    if (IsUndefined) {
      addPotentialUndefinedSymbol(Sym, F != nullptr);
      continue;
    }

    if (F) {
      addDefinedFunctionSymbol(Sym);
      continue;
    }

    if (isa<GlobalVariable>(GV)) {
      addDefinedDataSymbol(Sym);
      continue;
    }

    assert(isa<GlobalAlias>(GV));
    addDefinedDataSymbol(Sym);
  }

  // make symbols for all undefines
  for (StringMap<NameAndAttributes>::iterator u =_undefines.begin(),
         e = _undefines.end(); u != e; ++u) {
    // If this symbol also has a definition, then don't make an undefine because
    // it is a tentative definition.
    if (_defines.count(u->getKey())) continue;
    NameAndAttributes info = u->getValue();
    _symbols.push_back(info);
  }

  return false;
}

/// parseMetadata - Parse metadata from the module
void LTOModule::parseMetadata() {
  // Linker Options
  if (Value *Val = getModule().getModuleFlag("Linker Options")) {
    MDNode *LinkerOptions = cast<MDNode>(Val);
    for (unsigned i = 0, e = LinkerOptions->getNumOperands(); i != e; ++i) {
      MDNode *MDOptions = cast<MDNode>(LinkerOptions->getOperand(i));
      for (unsigned ii = 0, ie = MDOptions->getNumOperands(); ii != ie; ++ii) {
        MDString *MDOption = cast<MDString>(MDOptions->getOperand(ii));
        StringRef Op = _linkeropt_strings.
            GetOrCreateValue(MDOption->getString()).getKey();
        StringRef DepLibName = _target->getSubtargetImpl()
                                   ->getTargetLowering()
                                   ->getObjFileLowering()
                                   .getDepLibFromLinkerOpt(Op);
        if (!DepLibName.empty())
          _deplibs.push_back(DepLibName.data());
        else if (!Op.empty())
          _linkeropts.push_back(Op.data());
      }
    }
  }

  // Add other interesting metadata here.
}
