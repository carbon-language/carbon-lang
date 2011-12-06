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
  OutStreamer.EmitRawText(Twine("\t.version ") + ST.getPTXVersionString());
  OutStreamer.EmitRawText(Twine("\t.target ") + ST.getTargetString() +
                                (ST.supportsDouble() ? ""
                                                     : ", map_f64_to_f32"));
  // .address_size directive is optional, but it must immediately follow
  // the .target directive if present within a module
  if (ST.supportsPTX23()) {
    const char *addrSize = ST.is64Bit() ? "64" : "32";
    OutStreamer.EmitRawText(Twine("\t.address_size ") + addrSize);
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

  // declare external functions
  for (Module::const_iterator i = M.begin(), e = M.end();
       i != e; ++i)
    EmitFunctionDeclaration(i);
  
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
  SmallString<128> regDefs;
  raw_svector_ostream os(regDefs);
  unsigned numRegs;

  // pred
  numRegs = MFI->getNumRegistersForClass(PTX::RegPredRegisterClass);
  if(numRegs > 0)
    os << "\t.reg .pred %p<" << numRegs << ">;\n";

  // i16
  numRegs = MFI->getNumRegistersForClass(PTX::RegI16RegisterClass);
  if(numRegs > 0)
    os << "\t.reg .b16 %rh<" << numRegs << ">;\n";

  // i32
  numRegs = MFI->getNumRegistersForClass(PTX::RegI32RegisterClass);
  if(numRegs > 0)
    os << "\t.reg .b32 %r<" << numRegs << ">;\n";

  // i64
  numRegs = MFI->getNumRegistersForClass(PTX::RegI64RegisterClass);
  if(numRegs > 0)
    os << "\t.reg .b64 %rd<" << numRegs << ">;\n";

  // f32
  numRegs = MFI->getNumRegistersForClass(PTX::RegF32RegisterClass);
  if(numRegs > 0)
    os << "\t.reg .f32 %f<" << numRegs << ">;\n";

  // f64
  numRegs = MFI->getNumRegistersForClass(PTX::RegF64RegisterClass);
  if(numRegs > 0)
    os << "\t.reg .f64 %fd<" << numRegs << ">;\n";

  // Local params
  for (PTXParamManager::param_iterator i = PM.local_begin(), e = PM.local_end();
       i != e; ++i)
    os << "\t.param .b" << PM.getParamSize(*i) << ' ' << PM.getParamName(*i)
       << ";\n";

  OutStreamer.EmitRawText(os.str());


  const MachineFrameInfo* FrameInfo = MF->getFrameInfo();
  DEBUG(dbgs() << "Have " << FrameInfo->getNumObjects()
               << " frame object(s)\n");
  for (unsigned i = 0, e = FrameInfo->getNumObjects(); i != e; ++i) {
    DEBUG(dbgs() << "Size of object: " << FrameInfo->getObjectSize(i) << "\n");
    if (FrameInfo->getObjectSize(i) > 0) {
      OutStreamer.EmitRawText("\t.local .align " +
                              Twine(FrameInfo->getObjectAlignment(i)) +
                              " .b8 __local" +
                              Twine(i) +
                              "[" +
                              Twine(FrameInfo->getObjectSize(i)) +
                              "];");
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

  SmallString<128> decl;
  raw_svector_ostream os(decl);

  // check if it is defined in some other translation unit
  if (gv->isDeclaration())
    os << ".extern ";

  // state space: e.g., .global
  os << '.' << getStateSpaceName(gv->getType()->getAddressSpace()) << ' ';

  // alignment (optional)
  unsigned alignment = gv->getAlignment();
  if (alignment != 0)
    os << ".align " << gv->getAlignment() << ' ';


  if (PointerType::classof(gv->getType())) {
    PointerType* pointerTy = dyn_cast<PointerType>(gv->getType());
    Type* elementTy = pointerTy->getElementType();

    if (elementTy->isArrayTy()) {
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

      // Find the size of the element in bits
      unsigned elementSize = elementTy->getPrimitiveSizeInBits();

      os << ".b" << elementSize << ' ' << gvsym->getName()
         << '[' << numElements << ']';
    } else {
      os << ".b8" << gvsym->getName() << "[]";
    }

    // handle string constants (assume ConstantArray means string)
    if (gv->hasInitializer()) {
      const Constant *C = gv->getInitializer();
      if (const ConstantArray *CA = dyn_cast<ConstantArray>(C)) {
        os << " = {";

        for (unsigned i = 0, e = C->getNumOperands(); i != e; ++i) {
          if (i > 0)
            os << ',';

          os << "0x";
          os.write_hex(cast<ConstantInt>(CA->getOperand(i))->getZExtValue());
        }

        os << '}';
      }
    }
  } else {
    // Note: this is currently the fall-through case and most likely generates
    //       incorrect code.
    os << getTypeName(gv->getType()) << ' ' << gvsym->getName();

    if (isa<ArrayType>(gv->getType()) || isa<PointerType>(gv->getType()))
      os << "[]";
  }

  os << ';';

  OutStreamer.EmitRawText(os.str());
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

  SmallString<128> decl;
  raw_svector_ostream os(decl);
  os << (isKernel ? ".entry" : ".func");

  if (!isKernel) {
    os << " (";
    if (ST.useParamSpaceForDeviceArgs()) {
      for (PTXParamManager::param_iterator i = PM.ret_begin(), e = PM.ret_end(),
           b = i; i != e; ++i) {
        if (i != b)
          os << ", ";

        os << ".param .b" << PM.getParamSize(*i) << ' ' << PM.getParamName(*i);
      }
    } else {
      for (PTXMachineFunctionInfo::reg_iterator
           i = MFI->retreg_begin(), e = MFI->retreg_end(), b = i;
           i != e; ++i) {
        if (i != b)
          os << ", ";

        os << ".reg ." << getRegisterTypeName(*i, MRI) << ' '
           << MFI->getRegisterName(*i);
      }
    }
    os << ')';
  }

  // Print function name
  os << ' ' << CurrentFnSym->getName() << " (";

  const Function *F = MF->getFunction();

  // Print parameters
  if (isKernel || ST.useParamSpaceForDeviceArgs()) {
    /*for (PTXParamManager::param_iterator i = PM.arg_begin(), e = PM.arg_end(),
         b = i; i != e; ++i) {
      if (i != b)
        os << ", ";

      os << ".param .b" << PM.getParamSize(*i) << ' ' << PM.getParamName(*i);
    }*/
    int Counter = 1;
    for (Function::const_arg_iterator i = F->arg_begin(), e = F->arg_end(),
         b = i; i != e; ++i) {
      if (i != b)
        os << ", ";
      const Type *ArgType = (*i).getType();
      os << ".param .b";
      if (ArgType->isPointerTy()) {
        if (ST.is64Bit())
          os << "64";
        else
          os << "32";
      } else {
        os << ArgType->getPrimitiveSizeInBits();
      }
      if (ArgType->isPointerTy() && ST.emitPtrAttribute()) {
        const PointerType *PtrType = dyn_cast<const PointerType>(ArgType);
        os << " .ptr";
        switch (PtrType->getAddressSpace()) {
        default:
          llvm_unreachable("Unknown address space in argument");
        case PTXStateSpace::Global:
          os << " .global";
          break;
        case PTXStateSpace::Shared:
          os << " .shared";
          break;
        }
      }
      os << " __param_" << Counter++;
    }
  } else {
    for (PTXMachineFunctionInfo::reg_iterator
         i = MFI->argreg_begin(), e = MFI->argreg_end(), b = i;
         i != e; ++i) {
      if (i != b)
        os << ", ";

      os << ".reg ." << getRegisterTypeName(*i, MRI) << ' '
         << MFI->getRegisterName(*i);
    }
  }
  os << ')';

  OutStreamer.EmitRawText(os.str());
}

void PTXAsmPrinter::EmitFunctionDeclaration(const Function* func)
{
  const PTXSubtarget& ST = TM.getSubtarget<PTXSubtarget>();
	
  std::string decl = "";

  // hard-coded emission of extern vprintf function 
  
  if (func->getName() == "printf" || func->getName() == "puts") {		
    decl += ".extern .func (.param .b32 __param_1) vprintf (.param .b";
    if (ST.is64Bit())	
      decl += "64";
    else				
      decl += "32";
    decl += " __param_2, .param .b";
    if (ST.is64Bit())	
      decl += "64";
    else				
      decl += "32";
    decl += " __param_3)\n";
  }
  
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
  OutStreamer.EmitDwarfFileDirective(SrcId, "", Entry.getKey());

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
  const MachineRegisterInfo& MRI = MF->getRegInfo();
  const TargetRegisterClass* TRC;
  unsigned RegType;
  unsigned RegOffset;
  unsigned EncodedReg;
  switch (MO.getType()) {
  default:
    llvm_unreachable("Unknown operand type");
  case MachineOperand::MO_Register:
    if (MO.getReg() > 0) {
      TRC = MRI.getRegClass(MO.getReg());
      // Determine which PTX register type to use
      if (TRC == PTX::RegPredRegisterClass)
        RegType = PTXRegisterType::Pred;
      else if (TRC == PTX::RegI16RegisterClass)
        RegType = PTXRegisterType::B16;
      else if (TRC == PTX::RegI32RegisterClass)
        RegType = PTXRegisterType::B32;
      else if (TRC == PTX::RegI64RegisterClass)
        RegType = PTXRegisterType::B64;
      else if (TRC == PTX::RegF32RegisterClass)
        RegType = PTXRegisterType::F32;
      else if (TRC == PTX::RegF64RegisterClass)
        RegType = PTXRegisterType::F64;
      // Determine our virtual register offset
      RegOffset = MFI->getOffsetForRegister(TRC, MO.getReg());
      // Encode the register
      EncodedReg = (RegOffset << 4) | RegType;
    } else {
      EncodedReg = 0;
    }
    MCOp = MCOperand::CreateReg(EncodedReg);
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
