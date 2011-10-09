//===-- PTXAsmPrinter.cpp - PTX LLVM assembly writer ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to PTX assembly language.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ptx-asm-printer"

#include "PTX.h"
#include "PTXAsmPrinter.h"
#include "PTXMachineFunctionInfo.h"
#include "PTXParamManager.h"
#include "PTXRegisterInfo.h"
#include "PTXTargetMachine.h"
#include "llvm/Argument.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static const char PARAM_PREFIX[] = "__param_";
static const char RETURN_PREFIX[] = "__ret_";

static const char *getRegisterTypeName(unsigned RegNo,
                                       const MachineRegisterInfo& MRI) {
  const TargetRegisterClass *TRC = MRI.getRegClass(RegNo);

#define TEST_REGCLS(cls, clsstr) \
  if (PTX::cls ## RegisterClass == TRC) return # clsstr;

  TEST_REGCLS(RegPred, pred);
  TEST_REGCLS(RegI16, b16);
  TEST_REGCLS(RegI32, b32);
  TEST_REGCLS(RegI64, b64);
  TEST_REGCLS(RegF32, b32);
  TEST_REGCLS(RegF64, b64);
#undef TEST_REGCLS

  llvm_unreachable("Not in any register class!");
  return NULL;
}

static const char *getStateSpaceName(unsigned addressSpace) {
  switch (addressSpace) {
  default: llvm_unreachable("Unknown state space");
  case PTXStateSpace::Global:    return "global";
  case PTXStateSpace::Constant:  return "const";
  case PTXStateSpace::Local:     return "local";
  case PTXStateSpace::Parameter: return "param";
  case PTXStateSpace::Shared:    return "shared";
  }
  return NULL;
}

static const char *getTypeName(Type* type) {
  while (true) {
    switch (type->getTypeID()) {
      default: llvm_unreachable("Unknown type");
      case Type::FloatTyID: return ".f32";
      case Type::DoubleTyID: return ".f64";
      case Type::IntegerTyID:
        switch (type->getPrimitiveSizeInBits()) {
          default: llvm_unreachable("Unknown integer bit-width");
          case 16: return ".u16";
          case 32: return ".u32";
          case 64: return ".u64";
        }
      case Type::ArrayTyID:
      case Type::PointerTyID:
        type = dyn_cast<SequentialType>(type)->getElementType();
        break;
    }
  }
  return NULL;
}

bool PTXAsmPrinter::doFinalization(Module &M) {
  // XXX Temproarily remove global variables so that doFinalization() will not
  // emit them again (global variables are emitted at beginning).

  Module::GlobalListType &global_list = M.getGlobalList();
  int i, n = global_list.size();
  GlobalVariable **gv_array = new GlobalVariable* [n];

  // first, back-up GlobalVariable in gv_array
  i = 0;
  for (Module::global_iterator I = global_list.begin(), E = global_list.end();
       I != E; ++I)
    gv_array[i++] = &*I;

  // second, empty global_list
  while (!global_list.empty())
    global_list.remove(global_list.begin());

  // call doFinalization
  bool ret = AsmPrinter::doFinalization(M);

  // now we restore global variables
  for (i = 0; i < n; i ++)
    global_list.insert(global_list.end(), gv_array[i]);

  delete[] gv_array;
  return ret;
}

void PTXAsmPrinter::EmitStartOfAsmFile(Module &M)
{
  const PTXSubtarget& ST = TM.getSubtarget<PTXSubtarget>();

  // Emit the PTX .version and .target attributes
  OutStreamer.EmitRawText(Twine("\t.version " + ST.getPTXVersionString()));
  OutStreamer.EmitRawText(Twine("\t.target " + ST.getTargetString() +
                                (ST.supportsDouble() ? ""
                                                     : ", map_f64_to_f32")));
  // .address_size directive is optional, but it must immediately follow
  // the .target directive if present within a module
  if (ST.supportsPTX23()) {
    std::string addrSize = ST.is64Bit() ? "64" : "32";
    OutStreamer.EmitRawText(Twine("\t.address_size " + addrSize));
  }

  OutStreamer.AddBlankLine();

  // Define any .file directives
  DebugInfoFinder DbgFinder;
  DbgFinder.processModule(M);

  for (DebugInfoFinder::iterator I = DbgFinder.compile_unit_begin(),
       E = DbgFinder.compile_unit_end(); I != E; ++I) {
    DICompileUnit DIUnit(*I);
    StringRef FN = DIUnit.getFilename();
    StringRef Dir = DIUnit.getDirectory();
    GetOrCreateSourceID(FN, Dir);
  }

  OutStreamer.AddBlankLine();

  // declare global variables
  for (Module::const_global_iterator i = M.global_begin(), e = M.global_end();
       i != e; ++i)
    EmitVariableDeclaration(i);
}

void PTXAsmPrinter::EmitFunctionBodyStart() {
  OutStreamer.EmitRawText(Twine("{"));

  const PTXMachineFunctionInfo *MFI = MF->getInfo<PTXMachineFunctionInfo>();
  const PTXParamManager &PM = MFI->getParamManager();

  // Print register definitions
  std::string regDefs;
  unsigned numRegs;

  // pred
  numRegs = MFI->getNumRegistersForClass(PTX::RegPredRegisterClass);
  if(numRegs > 0) {
    regDefs += "\t.reg .pred %p<";
    regDefs += utostr(numRegs);
    regDefs += ">;\n";
  }

  // i16
  numRegs = MFI->getNumRegistersForClass(PTX::RegI16RegisterClass);
  if(numRegs > 0) {
    regDefs += "\t.reg .b16 %rh<";
    regDefs += utostr(numRegs);
    regDefs += ">;\n";
  }

  // i32
  numRegs = MFI->getNumRegistersForClass(PTX::RegI32RegisterClass);
  if(numRegs > 0) {
    regDefs += "\t.reg .b32 %r<";
    regDefs += utostr(numRegs);
    regDefs += ">;\n";
  }

  // i64
  numRegs = MFI->getNumRegistersForClass(PTX::RegI64RegisterClass);
  if(numRegs > 0) {
    regDefs += "\t.reg .b64 %rd<";
    regDefs += utostr(numRegs);
    regDefs += ">;\n";
  }

  // f32
  numRegs = MFI->getNumRegistersForClass(PTX::RegF32RegisterClass);
  if(numRegs > 0) {
    regDefs += "\t.reg .f32 %f<";
    regDefs += utostr(numRegs);
    regDefs += ">;\n";
  }

  // f64
  numRegs = MFI->getNumRegistersForClass(PTX::RegF64RegisterClass);
  if(numRegs > 0) {
    regDefs += "\t.reg .f64 %fd<";
    regDefs += utostr(numRegs);
    regDefs += ">;\n";
  }

  // Local params
  for (PTXParamManager::param_iterator i = PM.local_begin(), e = PM.local_end();
       i != e; ++i) {
    regDefs += "\t.param .b";
    regDefs += utostr(PM.getParamSize(*i));
    regDefs += " ";
    regDefs += PM.getParamName(*i);
    regDefs += ";\n";
  }

  OutStreamer.EmitRawText(Twine(regDefs));


  const MachineFrameInfo* FrameInfo = MF->getFrameInfo();
  DEBUG(dbgs() << "Have " << FrameInfo->getNumObjects()
               << " frame object(s)\n");
  for (unsigned i = 0, e = FrameInfo->getNumObjects(); i != e; ++i) {
    DEBUG(dbgs() << "Size of object: " << FrameInfo->getObjectSize(i) << "\n");
    if (FrameInfo->getObjectSize(i) > 0) {
      std::string def = "\t.local .align ";
      def += utostr(FrameInfo->getObjectAlignment(i));
      def += " .b8";
      def += " __local";
      def += utostr(i);
      def += "[";
      def += utostr(FrameInfo->getObjectSize(i)); // Convert to bits
      def += "]";
      def += ";";
      OutStreamer.EmitRawText(Twine(def));
    }
  }

  //unsigned Index = 1;
  // Print parameter passing params
  //for (PTXMachineFunctionInfo::param_iterator
  //     i = MFI->paramBegin(), e = MFI->paramEnd(); i != e; ++i) {
  //  std::string def = "\t.param .b";
  //  def += utostr(*i);
  //  def += " __ret_";
  //  def += utostr(Index);
  //  Index++;
  //  def += ";";
  //  OutStreamer.EmitRawText(Twine(def));
  //}
}

void PTXAsmPrinter::EmitFunctionBodyEnd() {
  OutStreamer.EmitRawText(Twine("}"));
}

void PTXAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  MCInst TmpInst;
  LowerPTXMachineInstrToMCInst(MI, TmpInst, *this);
  OutStreamer.EmitInstruction(TmpInst);
}

void PTXAsmPrinter::EmitVariableDeclaration(const GlobalVariable *gv) {
  // Check to see if this is a special global used by LLVM, if so, emit it.
  if (EmitSpecialLLVMGlobal(gv))
    return;

  MCSymbol *gvsym = Mang->getSymbol(gv);

  assert(gvsym->isUndefined() && "Cannot define a symbol twice!");

  std::string decl;

  // check if it is defined in some other translation unit
  if (gv->isDeclaration())
    decl += ".extern ";

  // state space: e.g., .global
  decl += ".";
  decl += getStateSpaceName(gv->getType()->getAddressSpace());
  decl += " ";

  // alignment (optional)
  unsigned alignment = gv->getAlignment();
  if (alignment != 0) {
    decl += ".align ";
    decl += utostr(gv->getAlignment());
    decl += " ";
  }


  if (PointerType::classof(gv->getType())) {
    PointerType* pointerTy = dyn_cast<PointerType>(gv->getType());
    Type* elementTy = pointerTy->getElementType();

    decl += ".b8 ";
    decl += gvsym->getName();
    decl += "[";

    if (elementTy->isArrayTy())
    {
      assert(elementTy->isArrayTy() && "Only pointers to arrays are supported");

      ArrayType* arrayTy = dyn_cast<ArrayType>(elementTy);
      elementTy = arrayTy->getElementType();

      unsigned numElements = arrayTy->getNumElements();

      while (elementTy->isArrayTy()) {

        arrayTy = dyn_cast<ArrayType>(elementTy);
        elementTy = arrayTy->getElementType();

        numElements *= arrayTy->getNumElements();
      }

      // FIXME: isPrimitiveType() == false for i16?
      assert(elementTy->isSingleValueType() &&
              "Non-primitive types are not handled");

      // Compute the size of the array, in bytes.
      uint64_t arraySize = (elementTy->getPrimitiveSizeInBits() >> 3)
                        * numElements;

      decl += utostr(arraySize);
    }

    decl += "]";

    // handle string constants (assume ConstantArray means string)

    if (gv->hasInitializer())
    {
      const Constant *C = gv->getInitializer();
      if (const ConstantArray *CA = dyn_cast<ConstantArray>(C))
      {
        decl += " = {";

        for (unsigned i = 0, e = C->getNumOperands(); i != e; ++i)
        {
          if (i > 0)   decl += ",";

          decl += "0x" +
                utohexstr(cast<ConstantInt>(CA->getOperand(i))->getZExtValue());
        }

        decl += "}";
      }
    }
  }
  else {
    // Note: this is currently the fall-through case and most likely generates
    //       incorrect code.
    decl += getTypeName(gv->getType());
    decl += " ";

    decl += gvsym->getName();

    if (ArrayType::classof(gv->getType()) ||
        PointerType::classof(gv->getType()))
      decl += "[]";
  }

  decl += ";";

  OutStreamer.EmitRawText(Twine(decl));

  OutStreamer.AddBlankLine();
}

void PTXAsmPrinter::EmitFunctionEntryLabel() {
  // The function label could have already been emitted if two symbols end up
  // conflicting due to asm renaming.  Detect this and emit an error.
  if (!CurrentFnSym->isUndefined()) {
    report_fatal_error("'" + Twine(CurrentFnSym->getName()) +
                       "' label emitted multiple times to assembly file");
    return;
  }

  const PTXMachineFunctionInfo *MFI = MF->getInfo<PTXMachineFunctionInfo>();
  const PTXParamManager &PM = MFI->getParamManager();
  const bool isKernel = MFI->isKernel();
  const PTXSubtarget& ST = TM.getSubtarget<PTXSubtarget>();
  const MachineRegisterInfo& MRI = MF->getRegInfo();

  std::string decl = isKernel ? ".entry" : ".func";

  if (!isKernel) {
    decl += " (";
    if (ST.useParamSpaceForDeviceArgs()) {
      for (PTXParamManager::param_iterator i = PM.ret_begin(), e = PM.ret_end(),
           b = i; i != e; ++i) {
        if (i != b) {
          decl += ", ";
        }

        decl += ".param .b";
        decl += utostr(PM.getParamSize(*i));
        decl += " ";
        decl += PM.getParamName(*i);
      }
    } else {
      for (PTXMachineFunctionInfo::reg_iterator
           i = MFI->retreg_begin(), e = MFI->retreg_end(), b = i;
           i != e; ++i) {
        if (i != b) {
          decl += ", ";
        }
        decl += ".reg .";
        decl += getRegisterTypeName(*i, MRI);
        decl += " ";
        decl += MFI->getRegisterName(*i);
      }
    }
    decl += ")";
  }

  // Print function name
  decl += " ";
  decl += CurrentFnSym->getName().str();

  decl += " (";

  const Function *F = MF->getFunction();

  // Print parameters
  if (isKernel || ST.useParamSpaceForDeviceArgs()) {
    /*for (PTXParamManager::param_iterator i = PM.arg_begin(), e = PM.arg_end(),
         b = i; i != e; ++i) {
      if (i != b) {
        decl += ", ";
      }

      decl += ".param .b";
      decl += utostr(PM.getParamSize(*i));
      decl += " ";
      decl += PM.getParamName(*i);
    }*/
    int Counter = 1;
    for (Function::const_arg_iterator i = F->arg_begin(), e = F->arg_end(),
         b = i; i != e; ++i) {
      if (i != b)
        decl += ", ";
      const Type *ArgType = (*i).getType();
      decl += ".param .b";
      if (ArgType->isPointerTy()) {
        if (ST.is64Bit())
          decl += "64";
        else
          decl += "32";
      } else {
        decl += utostr(ArgType->getPrimitiveSizeInBits());
      }
      if (ArgType->isPointerTy() && ST.emitPtrAttribute()) {
        const PointerType *PtrType = dyn_cast<const PointerType>(ArgType);
        decl += " .ptr";
        switch (PtrType->getAddressSpace()) {
        default:
          llvm_unreachable("Unknown address space in argument");
        case PTXStateSpace::Global:
          decl += " .global";
          break;
        case PTXStateSpace::Shared:
          decl += " .shared";
          break;
        }
      }
      decl += " __param_";
      decl += utostr(Counter++);
    }
  } else {
    for (PTXMachineFunctionInfo::reg_iterator
         i = MFI->argreg_begin(), e = MFI->argreg_end(), b = i;
         i != e; ++i) {
      if (i != b) {
        decl += ", ";
      }

      decl += ".reg .";
      decl += getRegisterTypeName(*i, MRI);
      decl += " ";
      decl += MFI->getRegisterName(*i);
    }
  }
  decl += ")";

  OutStreamer.EmitRawText(Twine(decl));
}

unsigned PTXAsmPrinter::GetOrCreateSourceID(StringRef FileName,
                                            StringRef DirName) {
  // If FE did not provide a file name, then assume stdin.
  if (FileName.empty())
    return GetOrCreateSourceID("<stdin>", StringRef());

  // MCStream expects full path name as filename.
  if (!DirName.empty() && !sys::path::is_absolute(FileName)) {
    SmallString<128> FullPathName = DirName;
    sys::path::append(FullPathName, FileName);
    // Here FullPathName will be copied into StringMap by GetOrCreateSourceID.
    return GetOrCreateSourceID(StringRef(FullPathName), StringRef());
  }

  StringMapEntry<unsigned> &Entry = SourceIdMap.GetOrCreateValue(FileName);
  if (Entry.getValue())
    return Entry.getValue();

  unsigned SrcId = SourceIdMap.size();
  Entry.setValue(SrcId);

  // Print out a .file directive to specify files for .loc directives.
  OutStreamer.EmitDwarfFileDirective(SrcId, Entry.getKey());

  return SrcId;
}

MCOperand PTXAsmPrinter::GetSymbolRef(const MachineOperand &MO,
                                      const MCSymbol *Symbol) {
  const MCExpr *Expr;
  Expr = MCSymbolRefExpr::Create(Symbol, MCSymbolRefExpr::VK_None, OutContext);
  return MCOperand::CreateExpr(Expr);
}

MCOperand PTXAsmPrinter::lowerOperand(const MachineOperand &MO) {
  MCOperand MCOp;
  const PTXMachineFunctionInfo *MFI = MF->getInfo<PTXMachineFunctionInfo>();
  const MCExpr *Expr;
  const char *RegSymbolName;
  switch (MO.getType()) {
  default:
    llvm_unreachable("Unknown operand type");
  case MachineOperand::MO_Register:
    // We create register operands as symbols, since the PTXInstPrinter class
    // has no way to map virtual registers back to a name without some ugly
    // hacks.
    // FIXME: Figure out a better way to handle virtual register naming.
    RegSymbolName = MFI->getRegisterName(MO.getReg());
    Expr = MCSymbolRefExpr::Create(RegSymbolName, MCSymbolRefExpr::VK_None,
                                   OutContext);
    MCOp = MCOperand::CreateExpr(Expr);
    break;
  case MachineOperand::MO_Immediate:
    MCOp = MCOperand::CreateImm(MO.getImm());
    break;
  case MachineOperand::MO_MachineBasicBlock:
    MCOp = MCOperand::CreateExpr(MCSymbolRefExpr::Create(
                                 MO.getMBB()->getSymbol(), OutContext));
    break;
  case MachineOperand::MO_GlobalAddress:
    MCOp = GetSymbolRef(MO, Mang->getSymbol(MO.getGlobal()));
    break;
  case MachineOperand::MO_ExternalSymbol:
    MCOp = GetSymbolRef(MO, GetExternalSymbolSymbol(MO.getSymbolName()));
    break;
  case MachineOperand::MO_FPImmediate:
    APFloat Val = MO.getFPImm()->getValueAPF();
    bool ignored;
    Val.convert(APFloat::IEEEdouble, APFloat::rmTowardZero, &ignored);
    MCOp = MCOperand::CreateFPImm(Val.convertToDouble());
    break;
  }

  return MCOp;
}

// Force static initialization.
extern "C" void LLVMInitializePTXAsmPrinter() {
  RegisterAsmPrinter<PTXAsmPrinter> X(ThePTX32Target);
  RegisterAsmPrinter<PTXAsmPrinter> Y(ThePTX64Target);
}

