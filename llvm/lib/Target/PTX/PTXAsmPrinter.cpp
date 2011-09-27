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
#include "PTXMachineFunctionInfo.h"
#include "PTXParamManager.h"
#include "PTXRegisterInfo.h"
#include "PTXTargetMachine.h"
#include "llvm/DerivedTypes.h"
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

namespace {
class PTXAsmPrinter : public AsmPrinter {
public:
  explicit PTXAsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
    : AsmPrinter(TM, Streamer) {}

  const char *getPassName() const { return "PTX Assembly Printer"; }

  bool doFinalization(Module &M);

  virtual void EmitStartOfAsmFile(Module &M);

  virtual bool runOnMachineFunction(MachineFunction &MF);

  virtual void EmitFunctionBodyStart();
  virtual void EmitFunctionBodyEnd() { OutStreamer.EmitRawText(Twine("}")); }

  virtual void EmitInstruction(const MachineInstr *MI);

  void printOperand(const MachineInstr *MI, int opNum, raw_ostream &OS);
  void printMemOperand(const MachineInstr *MI, int opNum, raw_ostream &OS,
                       const char *Modifier = 0);
  void printReturnOperand(const MachineInstr *MI, int opNum, raw_ostream &OS,
                          const char *Modifier = 0);
  void printPredicateOperand(const MachineInstr *MI, raw_ostream &O);

  void printCall(const MachineInstr *MI, raw_ostream &O);

  unsigned GetOrCreateSourceID(StringRef FileName,
                               StringRef DirName);

  // autogen'd.
  void printInstruction(const MachineInstr *MI, raw_ostream &OS);
  static const char *getRegisterName(unsigned RegNo);

private:
  void EmitVariableDeclaration(const GlobalVariable *gv);
  void EmitFunctionDeclaration();

  StringMap<unsigned> SourceIdMap;
}; // class PTXAsmPrinter
} // namespace

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
  case PTX::GLOBAL:    return "global";
  case PTX::CONSTANT:  return "const";
  case PTX::LOCAL:     return "local";
  case PTX::PARAMETER: return "param";
  case PTX::SHARED:    return "shared";
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

bool PTXAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  SetupMachineFunction(MF);
  EmitFunctionDeclaration();
  EmitFunctionBody();
  return false;
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

void PTXAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  std::string str;
  str.reserve(64);

  raw_string_ostream OS(str);

  DebugLoc DL = MI->getDebugLoc();
  if (!DL.isUnknown()) {

    const MDNode *S = DL.getScope(MF->getFunction()->getContext());

    // This is taken from DwarfDebug.cpp, which is conveniently not a public
    // LLVM class.
    StringRef Fn;
    StringRef Dir;
    unsigned Src = 1;
    if (S) {
      DIDescriptor Scope(S);
      if (Scope.isCompileUnit()) {
        DICompileUnit CU(S);
        Fn = CU.getFilename();
        Dir = CU.getDirectory();
      } else if (Scope.isFile()) {
        DIFile F(S);
        Fn = F.getFilename();
        Dir = F.getDirectory();
      } else if (Scope.isSubprogram()) {
        DISubprogram SP(S);
        Fn = SP.getFilename();
        Dir = SP.getDirectory();
      } else if (Scope.isLexicalBlock()) {
        DILexicalBlock DB(S);
        Fn = DB.getFilename();
        Dir = DB.getDirectory();
      } else
        assert(0 && "Unexpected scope info");

      Src = GetOrCreateSourceID(Fn, Dir);
    }
    OutStreamer.EmitDwarfLocDirective(Src, DL.getLine(), DL.getCol(),
                                     0, 0, 0, Fn);

    const MCDwarfLoc& MDL = OutContext.getCurrentDwarfLoc();

    OS << "\t.loc ";
    OS << utostr(MDL.getFileNum());
    OS << " ";
    OS << utostr(MDL.getLine());
    OS << " ";
    OS << utostr(MDL.getColumn());
    OS << "\n";
  }


  // Emit predicate
  printPredicateOperand(MI, OS);

  // Write instruction to str
  if (MI->getOpcode() == PTX::CALL) {
    printCall(MI, OS);
  } else {
    printInstruction(MI, OS);
  }
  OS << ';';
  OS.flush();

  StringRef strref = StringRef(str);
  OutStreamer.EmitRawText(strref);
}

void PTXAsmPrinter::printOperand(const MachineInstr *MI, int opNum,
                                 raw_ostream &OS) {
  const MachineOperand &MO = MI->getOperand(opNum);
  const PTXMachineFunctionInfo *MFI = MF->getInfo<PTXMachineFunctionInfo>();

  switch (MO.getType()) {
    default:
      llvm_unreachable("<unknown operand type>");
      break;
    case MachineOperand::MO_GlobalAddress:
      OS << *Mang->getSymbol(MO.getGlobal());
      break;
    case MachineOperand::MO_Immediate:
      OS << (long) MO.getImm();
      break;
    case MachineOperand::MO_MachineBasicBlock:
      OS << *MO.getMBB()->getSymbol();
      break;
    case MachineOperand::MO_Register:
      OS << MFI->getRegisterName(MO.getReg());
      break;
    case MachineOperand::MO_ExternalSymbol:
      OS << MO.getSymbolName();
      break;
    case MachineOperand::MO_FPImmediate:
      APInt constFP = MO.getFPImm()->getValueAPF().bitcastToAPInt();
      bool  isFloat = MO.getFPImm()->getType()->getTypeID() == Type::FloatTyID;
      // Emit 0F for 32-bit floats and 0D for 64-bit doubles.
      if (isFloat) {
        OS << "0F";
      }
      else {
        OS << "0D";
      }
      // Emit the encoded floating-point value.
      if (constFP.getZExtValue() > 0) {
        OS << constFP.toString(16, false);
      }
      else {
        OS << "00000000";
        // If We have a double-precision zero, pad to 8-bytes.
        if (!isFloat) {
          OS << "00000000";
        }
      }
      break;
  }
}

void PTXAsmPrinter::printMemOperand(const MachineInstr *MI, int opNum,
                                    raw_ostream &OS, const char *Modifier) {
  printOperand(MI, opNum, OS);

  if (MI->getOperand(opNum+1).isImm() && MI->getOperand(opNum+1).getImm() == 0)
    return; // don't print "+0"

  OS << "+";
  printOperand(MI, opNum+1, OS);
}

void PTXAsmPrinter::printReturnOperand(const MachineInstr *MI, int opNum,
                                       raw_ostream &OS, const char *Modifier) {
  //OS << RETURN_PREFIX << (int) MI->getOperand(opNum).getImm() + 1;
  OS << "__ret";
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
    decl += utostr(Log2_32(gv->getAlignment()));
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

void PTXAsmPrinter::EmitFunctionDeclaration() {
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

  unsigned cnt = 0;

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

  cnt = 0;

  // Print parameters
  if (isKernel || ST.useParamSpaceForDeviceArgs()) {
    for (PTXParamManager::param_iterator i = PM.arg_begin(), e = PM.arg_end(),
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

void PTXAsmPrinter::
printPredicateOperand(const MachineInstr *MI, raw_ostream &O) {
  int i = MI->findFirstPredOperandIdx();
  if (i == -1)
    llvm_unreachable("missing predicate operand");

  unsigned reg = MI->getOperand(i).getReg();
  int predOp = MI->getOperand(i+1).getImm();
  const PTXMachineFunctionInfo *MFI = MF->getInfo<PTXMachineFunctionInfo>();

  DEBUG(dbgs() << "predicate: (" << reg << ", " << predOp << ")\n");

  if (reg != PTX::NoRegister) {
    O << '@';
    if (predOp == PTX::PRED_NEGATE)
      O << '!';
    O << MFI->getRegisterName(reg);
  }
}

void PTXAsmPrinter::
printCall(const MachineInstr *MI, raw_ostream &O) {
  O << "\tcall.uni\t";
  // The first two operands are the predicate slot
  unsigned Index = 2;
  while (!MI->getOperand(Index).isGlobal()) {
    if (Index == 2) {
      O << "(";
    } else {
      O << ", ";
    }
    printOperand(MI, Index, O);
    Index++;
  }

  if (Index != 2) {
    O << "), ";
  }

  assert(MI->getOperand(Index).isGlobal() &&
         "A GlobalAddress must follow the return arguments");

  const GlobalValue *Address = MI->getOperand(Index).getGlobal();
  O << Address->getName() << ", (";
  Index++;

  while (Index < MI->getNumOperands()) {
    printOperand(MI, Index, O);
    if (Index < MI->getNumOperands()-1) {
      O << ", ";
    }
    Index++;
  }

  O << ")";
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

#include "PTXGenAsmWriter.inc"

// Force static initialization.
extern "C" void LLVMInitializePTXAsmPrinter() {
  RegisterAsmPrinter<PTXAsmPrinter> X(ThePTX32Target);
  RegisterAsmPrinter<PTXAsmPrinter> Y(ThePTX64Target);
}
