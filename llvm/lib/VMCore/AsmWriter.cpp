//===-- AsmWriter.cpp - Printing LLVM as an assembly file -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This library implements the functionality defined in llvm/Assembly/Writer.h
//
// Note that these routines must be extremely tolerant of various errors in the
// LLVM code, because it can be used for debugging transformations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Assembly/Writer.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Assembly/AssemblyAnnotationWriter.h"
#include "llvm/LLVMContext.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/InlineAsm.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Operator.h"
#include "llvm/Module.h"
#include "llvm/ValueSymbolTable.h"
#include "llvm/TypeSymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/FormattedStream.h"
#include <algorithm>
#include <cctype>
using namespace llvm;

// Make virtual table appear in this compilation unit.
AssemblyAnnotationWriter::~AssemblyAnnotationWriter() {}

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static const Module *getModuleFromVal(const Value *V) {
  if (const Argument *MA = dyn_cast<Argument>(V))
    return MA->getParent() ? MA->getParent()->getParent() : 0;

  if (const BasicBlock *BB = dyn_cast<BasicBlock>(V))
    return BB->getParent() ? BB->getParent()->getParent() : 0;

  if (const Instruction *I = dyn_cast<Instruction>(V)) {
    const Function *M = I->getParent() ? I->getParent()->getParent() : 0;
    return M ? M->getParent() : 0;
  }
  
  if (const GlobalValue *GV = dyn_cast<GlobalValue>(V))
    return GV->getParent();
  return 0;
}

// PrintEscapedString - Print each character of the specified string, escaping
// it if it is not printable or if it is an escape char.
static void PrintEscapedString(StringRef Name, raw_ostream &Out) {
  for (unsigned i = 0, e = Name.size(); i != e; ++i) {
    unsigned char C = Name[i];
    if (isprint(C) && C != '\\' && C != '"')
      Out << C;
    else
      Out << '\\' << hexdigit(C >> 4) << hexdigit(C & 0x0F);
  }
}

enum PrefixType {
  GlobalPrefix,
  LabelPrefix,
  LocalPrefix,
  NoPrefix
};

/// PrintLLVMName - Turn the specified name into an 'LLVM name', which is either
/// prefixed with % (if the string only contains simple characters) or is
/// surrounded with ""'s (if it has special chars in it).  Print it out.
static void PrintLLVMName(raw_ostream &OS, StringRef Name, PrefixType Prefix) {
  assert(!Name.empty() && "Cannot get empty name!");
  switch (Prefix) {
  default: llvm_unreachable("Bad prefix!");
  case NoPrefix: break;
  case GlobalPrefix: OS << '@'; break;
  case LabelPrefix:  break;
  case LocalPrefix:  OS << '%'; break;
  }

  // Scan the name to see if it needs quotes first.
  bool NeedsQuotes = isdigit(Name[0]);
  if (!NeedsQuotes) {
    for (unsigned i = 0, e = Name.size(); i != e; ++i) {
      char C = Name[i];
      if (!isalnum(C) && C != '-' && C != '.' && C != '_') {
        NeedsQuotes = true;
        break;
      }
    }
  }

  // If we didn't need any quotes, just write out the name in one blast.
  if (!NeedsQuotes) {
    OS << Name;
    return;
  }

  // Okay, we need quotes.  Output the quotes and escape any scary characters as
  // needed.
  OS << '"';
  PrintEscapedString(Name, OS);
  OS << '"';
}

/// PrintLLVMName - Turn the specified name into an 'LLVM name', which is either
/// prefixed with % (if the string only contains simple characters) or is
/// surrounded with ""'s (if it has special chars in it).  Print it out.
static void PrintLLVMName(raw_ostream &OS, const Value *V) {
  PrintLLVMName(OS, V->getName(),
                isa<GlobalValue>(V) ? GlobalPrefix : LocalPrefix);
}

//===----------------------------------------------------------------------===//
// TypePrinting Class: Type printing machinery
//===----------------------------------------------------------------------===//

static DenseMap<const Type *, std::string> &getTypeNamesMap(void *M) {
  return *static_cast<DenseMap<const Type *, std::string>*>(M);
}

void TypePrinting::clear() {
  getTypeNamesMap(TypeNames).clear();
}

bool TypePrinting::hasTypeName(const Type *Ty) const {
  return getTypeNamesMap(TypeNames).count(Ty);
}

void TypePrinting::addTypeName(const Type *Ty, const std::string &N) {
  getTypeNamesMap(TypeNames).insert(std::make_pair(Ty, N));
}


TypePrinting::TypePrinting() {
  TypeNames = new DenseMap<const Type *, std::string>();
}

TypePrinting::~TypePrinting() {
  delete &getTypeNamesMap(TypeNames);
}

/// CalcTypeName - Write the specified type to the specified raw_ostream, making
/// use of type names or up references to shorten the type name where possible.
void TypePrinting::CalcTypeName(const Type *Ty,
                                SmallVectorImpl<const Type *> &TypeStack,
                                raw_ostream &OS, bool IgnoreTopLevelName) {
  // Check to see if the type is named.
  if (!IgnoreTopLevelName) {
    DenseMap<const Type *, std::string> &TM = getTypeNamesMap(TypeNames);
    DenseMap<const Type *, std::string>::iterator I = TM.find(Ty);
    if (I != TM.end()) {
      OS << I->second;
      return;
    }
  }

  // Check to see if the Type is already on the stack...
  unsigned Slot = 0, CurSize = TypeStack.size();
  while (Slot < CurSize && TypeStack[Slot] != Ty) ++Slot; // Scan for type

  // This is another base case for the recursion.  In this case, we know
  // that we have looped back to a type that we have previously visited.
  // Generate the appropriate upreference to handle this.
  if (Slot < CurSize) {
    OS << '\\' << unsigned(CurSize-Slot);     // Here's the upreference
    return;
  }

  TypeStack.push_back(Ty);    // Recursive case: Add us to the stack..

  switch (Ty->getTypeID()) {
  case Type::VoidTyID:      OS << "void"; break;
  case Type::FloatTyID:     OS << "float"; break;
  case Type::DoubleTyID:    OS << "double"; break;
  case Type::X86_FP80TyID:  OS << "x86_fp80"; break;
  case Type::FP128TyID:     OS << "fp128"; break;
  case Type::PPC_FP128TyID: OS << "ppc_fp128"; break;
  case Type::LabelTyID:     OS << "label"; break;
  case Type::MetadataTyID:  OS << "metadata"; break;
  case Type::X86_MMXTyID:   OS << "x86_mmx"; break;
  case Type::IntegerTyID:
    OS << 'i' << cast<IntegerType>(Ty)->getBitWidth();
    break;

  case Type::FunctionTyID: {
    const FunctionType *FTy = cast<FunctionType>(Ty);
    CalcTypeName(FTy->getReturnType(), TypeStack, OS);
    OS << " (";
    for (FunctionType::param_iterator I = FTy->param_begin(),
         E = FTy->param_end(); I != E; ++I) {
      if (I != FTy->param_begin())
        OS << ", ";
      CalcTypeName(*I, TypeStack, OS);
    }
    if (FTy->isVarArg()) {
      if (FTy->getNumParams()) OS << ", ";
      OS << "...";
    }
    OS << ')';
    break;
  }
  case Type::StructTyID: {
    const StructType *STy = cast<StructType>(Ty);
    if (STy->isPacked())
      OS << '<';
    OS << '{';
    for (StructType::element_iterator I = STy->element_begin(),
         E = STy->element_end(); I != E; ++I) {
      OS << ' ';
      CalcTypeName(*I, TypeStack, OS);
      if (llvm::next(I) == STy->element_end())
        OS << ' ';
      else
        OS << ',';
    }
    OS << '}';
    if (STy->isPacked())
      OS << '>';
    break;
  }
  case Type::PointerTyID: {
    const PointerType *PTy = cast<PointerType>(Ty);
    CalcTypeName(PTy->getElementType(), TypeStack, OS);
    if (unsigned AddressSpace = PTy->getAddressSpace())
      OS << " addrspace(" << AddressSpace << ')';
    OS << '*';
    break;
  }
  case Type::ArrayTyID: {
    const ArrayType *ATy = cast<ArrayType>(Ty);
    OS << '[' << ATy->getNumElements() << " x ";
    CalcTypeName(ATy->getElementType(), TypeStack, OS);
    OS << ']';
    break;
  }
  case Type::VectorTyID: {
    const VectorType *PTy = cast<VectorType>(Ty);
    OS << "<" << PTy->getNumElements() << " x ";
    CalcTypeName(PTy->getElementType(), TypeStack, OS);
    OS << '>';
    break;
  }
  case Type::OpaqueTyID:
    OS << "opaque";
    break;
  default:
    OS << "<unrecognized-type>";
    break;
  }

  TypeStack.pop_back();       // Remove self from stack.
}

/// printTypeInt - The internal guts of printing out a type that has a
/// potentially named portion.
///
void TypePrinting::print(const Type *Ty, raw_ostream &OS,
                         bool IgnoreTopLevelName) {
  // Check to see if the type is named.
  DenseMap<const Type*, std::string> &TM = getTypeNamesMap(TypeNames);
  if (!IgnoreTopLevelName) {
    DenseMap<const Type*, std::string>::iterator I = TM.find(Ty);
    if (I != TM.end()) {
      OS << I->second;
      return;
    }
  }

  // Otherwise we have a type that has not been named but is a derived type.
  // Carefully recurse the type hierarchy to print out any contained symbolic
  // names.
  SmallVector<const Type *, 16> TypeStack;
  std::string TypeName;

  raw_string_ostream TypeOS(TypeName);
  CalcTypeName(Ty, TypeStack, TypeOS, IgnoreTopLevelName);
  OS << TypeOS.str();

  // Cache type name for later use.
  if (!IgnoreTopLevelName)
    TM.insert(std::make_pair(Ty, TypeOS.str()));
}

namespace {
  class TypeFinder {
    // To avoid walking constant expressions multiple times and other IR
    // objects, we keep several helper maps.
    DenseSet<const Value*> VisitedConstants;
    DenseSet<const Type*> VisitedTypes;

    TypePrinting &TP;
    std::vector<const Type*> &NumberedTypes;
  public:
    TypeFinder(TypePrinting &tp, std::vector<const Type*> &numberedTypes)
      : TP(tp), NumberedTypes(numberedTypes) {}

    void Run(const Module &M) {
      // Get types from the type symbol table.  This gets opaque types referened
      // only through derived named types.
      const TypeSymbolTable &ST = M.getTypeSymbolTable();
      for (TypeSymbolTable::const_iterator TI = ST.begin(), E = ST.end();
           TI != E; ++TI)
        IncorporateType(TI->second);

      // Get types from global variables.
      for (Module::const_global_iterator I = M.global_begin(),
           E = M.global_end(); I != E; ++I) {
        IncorporateType(I->getType());
        if (I->hasInitializer())
          IncorporateValue(I->getInitializer());
      }

      // Get types from aliases.
      for (Module::const_alias_iterator I = M.alias_begin(),
           E = M.alias_end(); I != E; ++I) {
        IncorporateType(I->getType());
        IncorporateValue(I->getAliasee());
      }

      // Get types from functions.
      for (Module::const_iterator FI = M.begin(), E = M.end(); FI != E; ++FI) {
        IncorporateType(FI->getType());

        for (Function::const_iterator BB = FI->begin(), E = FI->end();
             BB != E;++BB)
          for (BasicBlock::const_iterator II = BB->begin(),
               E = BB->end(); II != E; ++II) {
            const Instruction &I = *II;
            // Incorporate the type of the instruction and all its operands.
            IncorporateType(I.getType());
            for (User::const_op_iterator OI = I.op_begin(), OE = I.op_end();
                 OI != OE; ++OI)
              IncorporateValue(*OI);
          }
      }
    }

  private:
    void IncorporateType(const Type *Ty) {
      // Check to see if we're already visited this type.
      if (!VisitedTypes.insert(Ty).second)
        return;

      // If this is a structure or opaque type, add a name for the type.
      if (((Ty->isStructTy() && cast<StructType>(Ty)->getNumElements())
            || Ty->isOpaqueTy()) && !TP.hasTypeName(Ty)) {
        TP.addTypeName(Ty, "%"+utostr(unsigned(NumberedTypes.size())));
        NumberedTypes.push_back(Ty);
      }

      // Recursively walk all contained types.
      for (Type::subtype_iterator I = Ty->subtype_begin(),
           E = Ty->subtype_end(); I != E; ++I)
        IncorporateType(*I);
    }

    /// IncorporateValue - This method is used to walk operand lists finding
    /// types hiding in constant expressions and other operands that won't be
    /// walked in other ways.  GlobalValues, basic blocks, instructions, and
    /// inst operands are all explicitly enumerated.
    void IncorporateValue(const Value *V) {
      if (V == 0 || !isa<Constant>(V) || isa<GlobalValue>(V)) return;

      // Already visited?
      if (!VisitedConstants.insert(V).second)
        return;

      // Check this type.
      IncorporateType(V->getType());

      // Look in operands for types.
      const Constant *C = cast<Constant>(V);
      for (Constant::const_op_iterator I = C->op_begin(),
           E = C->op_end(); I != E;++I)
        IncorporateValue(*I);
    }
  };
} // end anonymous namespace


/// AddModuleTypesToPrinter - Add all of the symbolic type names for types in
/// the specified module to the TypePrinter and all numbered types to it and the
/// NumberedTypes table.
static void AddModuleTypesToPrinter(TypePrinting &TP,
                                    std::vector<const Type*> &NumberedTypes,
                                    const Module *M) {
  if (M == 0) return;

  // If the module has a symbol table, take all global types and stuff their
  // names into the TypeNames map.
  const TypeSymbolTable &ST = M->getTypeSymbolTable();
  for (TypeSymbolTable::const_iterator TI = ST.begin(), E = ST.end();
       TI != E; ++TI) {
    const Type *Ty = cast<Type>(TI->second);

    // As a heuristic, don't insert pointer to primitive types, because
    // they are used too often to have a single useful name.
    if (const PointerType *PTy = dyn_cast<PointerType>(Ty)) {
      const Type *PETy = PTy->getElementType();
      if ((PETy->isPrimitiveType() || PETy->isIntegerTy()) &&
          !PETy->isOpaqueTy())
        continue;
    }

    // Likewise don't insert primitives either.
    if (Ty->isIntegerTy() || Ty->isPrimitiveType())
      continue;

    // Get the name as a string and insert it into TypeNames.
    std::string NameStr;
    raw_string_ostream NameROS(NameStr);
    formatted_raw_ostream NameOS(NameROS);
    PrintLLVMName(NameOS, TI->first, LocalPrefix);
    NameOS.flush();
    TP.addTypeName(Ty, NameStr);
  }

  // Walk the entire module to find references to unnamed structure and opaque
  // types.  This is required for correctness by opaque types (because multiple
  // uses of an unnamed opaque type needs to be referred to by the same ID) and
  // it shrinks complex recursive structure types substantially in some cases.
  TypeFinder(TP, NumberedTypes).Run(*M);
}


/// WriteTypeSymbolic - This attempts to write the specified type as a symbolic
/// type, iff there is an entry in the modules symbol table for the specified
/// type or one of it's component types.
///
void llvm::WriteTypeSymbolic(raw_ostream &OS, const Type *Ty, const Module *M) {
  TypePrinting Printer;
  std::vector<const Type*> NumberedTypes;
  AddModuleTypesToPrinter(Printer, NumberedTypes, M);
  Printer.print(Ty, OS);
}

//===----------------------------------------------------------------------===//
// SlotTracker Class: Enumerate slot numbers for unnamed values
//===----------------------------------------------------------------------===//

namespace {

/// This class provides computation of slot numbers for LLVM Assembly writing.
///
class SlotTracker {
public:
  /// ValueMap - A mapping of Values to slot numbers.
  typedef DenseMap<const Value*, unsigned> ValueMap;

private:
  /// TheModule - The module for which we are holding slot numbers.
  const Module* TheModule;

  /// TheFunction - The function for which we are holding slot numbers.
  const Function* TheFunction;
  bool FunctionProcessed;

  /// mMap - The TypePlanes map for the module level data.
  ValueMap mMap;
  unsigned mNext;

  /// fMap - The TypePlanes map for the function level data.
  ValueMap fMap;
  unsigned fNext;

  /// mdnMap - Map for MDNodes.
  DenseMap<const MDNode*, unsigned> mdnMap;
  unsigned mdnNext;
public:
  /// Construct from a module
  explicit SlotTracker(const Module *M);
  /// Construct from a function, starting out in incorp state.
  explicit SlotTracker(const Function *F);

  /// Return the slot number of the specified value in it's type
  /// plane.  If something is not in the SlotTracker, return -1.
  int getLocalSlot(const Value *V);
  int getGlobalSlot(const GlobalValue *V);
  int getMetadataSlot(const MDNode *N);

  /// If you'd like to deal with a function instead of just a module, use
  /// this method to get its data into the SlotTracker.
  void incorporateFunction(const Function *F) {
    TheFunction = F;
    FunctionProcessed = false;
  }

  /// After calling incorporateFunction, use this method to remove the
  /// most recently incorporated function from the SlotTracker. This
  /// will reset the state of the machine back to just the module contents.
  void purgeFunction();

  /// MDNode map iterators.
  typedef DenseMap<const MDNode*, unsigned>::iterator mdn_iterator;
  mdn_iterator mdn_begin() { return mdnMap.begin(); }
  mdn_iterator mdn_end() { return mdnMap.end(); }
  unsigned mdn_size() const { return mdnMap.size(); }
  bool mdn_empty() const { return mdnMap.empty(); }

  /// This function does the actual initialization.
  inline void initialize();

  // Implementation Details
private:
  /// CreateModuleSlot - Insert the specified GlobalValue* into the slot table.
  void CreateModuleSlot(const GlobalValue *V);

  /// CreateMetadataSlot - Insert the specified MDNode* into the slot table.
  void CreateMetadataSlot(const MDNode *N);

  /// CreateFunctionSlot - Insert the specified Value* into the slot table.
  void CreateFunctionSlot(const Value *V);

  /// Add all of the module level global variables (and their initializers)
  /// and function declarations, but not the contents of those functions.
  void processModule();

  /// Add all of the functions arguments, basic blocks, and instructions.
  void processFunction();

  SlotTracker(const SlotTracker &);  // DO NOT IMPLEMENT
  void operator=(const SlotTracker &);  // DO NOT IMPLEMENT
};

}  // end anonymous namespace


static SlotTracker *createSlotTracker(const Value *V) {
  if (const Argument *FA = dyn_cast<Argument>(V))
    return new SlotTracker(FA->getParent());

  if (const Instruction *I = dyn_cast<Instruction>(V))
    return new SlotTracker(I->getParent()->getParent());

  if (const BasicBlock *BB = dyn_cast<BasicBlock>(V))
    return new SlotTracker(BB->getParent());

  if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(V))
    return new SlotTracker(GV->getParent());

  if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(V))
    return new SlotTracker(GA->getParent());

  if (const Function *Func = dyn_cast<Function>(V))
    return new SlotTracker(Func);

  if (const MDNode *MD = dyn_cast<MDNode>(V)) {
    if (!MD->isFunctionLocal())
      return new SlotTracker(MD->getFunction());

    return new SlotTracker((Function *)0);
  }

  return 0;
}

#if 0
#define ST_DEBUG(X) dbgs() << X
#else
#define ST_DEBUG(X)
#endif

// Module level constructor. Causes the contents of the Module (sans functions)
// to be added to the slot table.
SlotTracker::SlotTracker(const Module *M)
  : TheModule(M), TheFunction(0), FunctionProcessed(false), 
    mNext(0), fNext(0),  mdnNext(0) {
}

// Function level constructor. Causes the contents of the Module and the one
// function provided to be added to the slot table.
SlotTracker::SlotTracker(const Function *F)
  : TheModule(F ? F->getParent() : 0), TheFunction(F), FunctionProcessed(false),
    mNext(0), fNext(0), mdnNext(0) {
}

inline void SlotTracker::initialize() {
  if (TheModule) {
    processModule();
    TheModule = 0; ///< Prevent re-processing next time we're called.
  }

  if (TheFunction && !FunctionProcessed)
    processFunction();
}

// Iterate through all the global variables, functions, and global
// variable initializers and create slots for them.
void SlotTracker::processModule() {
  ST_DEBUG("begin processModule!\n");

  // Add all of the unnamed global variables to the value table.
  for (Module::const_global_iterator I = TheModule->global_begin(),
         E = TheModule->global_end(); I != E; ++I) {
    if (!I->hasName())
      CreateModuleSlot(I);
  }

  // Add metadata used by named metadata.
  for (Module::const_named_metadata_iterator
         I = TheModule->named_metadata_begin(),
         E = TheModule->named_metadata_end(); I != E; ++I) {
    const NamedMDNode *NMD = I;
    for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i)
      CreateMetadataSlot(NMD->getOperand(i));
  }

  // Add all the unnamed functions to the table.
  for (Module::const_iterator I = TheModule->begin(), E = TheModule->end();
       I != E; ++I)
    if (!I->hasName())
      CreateModuleSlot(I);

  ST_DEBUG("end processModule!\n");
}

// Process the arguments, basic blocks, and instructions  of a function.
void SlotTracker::processFunction() {
  ST_DEBUG("begin processFunction!\n");
  fNext = 0;

  // Add all the function arguments with no names.
  for(Function::const_arg_iterator AI = TheFunction->arg_begin(),
      AE = TheFunction->arg_end(); AI != AE; ++AI)
    if (!AI->hasName())
      CreateFunctionSlot(AI);

  ST_DEBUG("Inserting Instructions:\n");

  SmallVector<std::pair<unsigned, MDNode*>, 4> MDForInst;

  // Add all of the basic blocks and instructions with no names.
  for (Function::const_iterator BB = TheFunction->begin(),
       E = TheFunction->end(); BB != E; ++BB) {
    if (!BB->hasName())
      CreateFunctionSlot(BB);
    
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I != E;
         ++I) {
      if (!I->getType()->isVoidTy() && !I->hasName())
        CreateFunctionSlot(I);
      
      // Intrinsics can directly use metadata.  We allow direct calls to any
      // llvm.foo function here, because the target may not be linked into the
      // optimizer.
      if (const CallInst *CI = dyn_cast<CallInst>(I)) {
        if (Function *F = CI->getCalledFunction())
          if (F->getName().startswith("llvm."))
            for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
              if (MDNode *N = dyn_cast_or_null<MDNode>(I->getOperand(i)))
                CreateMetadataSlot(N);
      }

      // Process metadata attached with this instruction.
      I->getAllMetadata(MDForInst);
      for (unsigned i = 0, e = MDForInst.size(); i != e; ++i)
        CreateMetadataSlot(MDForInst[i].second);
      MDForInst.clear();
    }
  }

  FunctionProcessed = true;

  ST_DEBUG("end processFunction!\n");
}

/// Clean up after incorporating a function. This is the only way to get out of
/// the function incorporation state that affects get*Slot/Create*Slot. Function
/// incorporation state is indicated by TheFunction != 0.
void SlotTracker::purgeFunction() {
  ST_DEBUG("begin purgeFunction!\n");
  fMap.clear(); // Simply discard the function level map
  TheFunction = 0;
  FunctionProcessed = false;
  ST_DEBUG("end purgeFunction!\n");
}

/// getGlobalSlot - Get the slot number of a global value.
int SlotTracker::getGlobalSlot(const GlobalValue *V) {
  // Check for uninitialized state and do lazy initialization.
  initialize();

  // Find the type plane in the module map
  ValueMap::iterator MI = mMap.find(V);
  return MI == mMap.end() ? -1 : (int)MI->second;
}

/// getMetadataSlot - Get the slot number of a MDNode.
int SlotTracker::getMetadataSlot(const MDNode *N) {
  // Check for uninitialized state and do lazy initialization.
  initialize();

  // Find the type plane in the module map
  mdn_iterator MI = mdnMap.find(N);
  return MI == mdnMap.end() ? -1 : (int)MI->second;
}


/// getLocalSlot - Get the slot number for a value that is local to a function.
int SlotTracker::getLocalSlot(const Value *V) {
  assert(!isa<Constant>(V) && "Can't get a constant or global slot with this!");

  // Check for uninitialized state and do lazy initialization.
  initialize();

  ValueMap::iterator FI = fMap.find(V);
  return FI == fMap.end() ? -1 : (int)FI->second;
}


/// CreateModuleSlot - Insert the specified GlobalValue* into the slot table.
void SlotTracker::CreateModuleSlot(const GlobalValue *V) {
  assert(V && "Can't insert a null Value into SlotTracker!");
  assert(!V->getType()->isVoidTy() && "Doesn't need a slot!");
  assert(!V->hasName() && "Doesn't need a slot!");

  unsigned DestSlot = mNext++;
  mMap[V] = DestSlot;

  ST_DEBUG("  Inserting value [" << V->getType() << "] = " << V << " slot=" <<
           DestSlot << " [");
  // G = Global, F = Function, A = Alias, o = other
  ST_DEBUG((isa<GlobalVariable>(V) ? 'G' :
            (isa<Function>(V) ? 'F' :
             (isa<GlobalAlias>(V) ? 'A' : 'o'))) << "]\n");
}

/// CreateSlot - Create a new slot for the specified value if it has no name.
void SlotTracker::CreateFunctionSlot(const Value *V) {
  assert(!V->getType()->isVoidTy() && !V->hasName() && "Doesn't need a slot!");

  unsigned DestSlot = fNext++;
  fMap[V] = DestSlot;

  // G = Global, F = Function, o = other
  ST_DEBUG("  Inserting value [" << V->getType() << "] = " << V << " slot=" <<
           DestSlot << " [o]\n");
}

/// CreateModuleSlot - Insert the specified MDNode* into the slot table.
void SlotTracker::CreateMetadataSlot(const MDNode *N) {
  assert(N && "Can't insert a null Value into SlotTracker!");

  // Don't insert if N is a function-local metadata, these are always printed
  // inline.
  if (!N->isFunctionLocal()) {
    mdn_iterator I = mdnMap.find(N);
    if (I != mdnMap.end())
      return;

    unsigned DestSlot = mdnNext++;
    mdnMap[N] = DestSlot;
  }

  // Recursively add any MDNodes referenced by operands.
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
    if (const MDNode *Op = dyn_cast_or_null<MDNode>(N->getOperand(i)))
      CreateMetadataSlot(Op);
}

//===----------------------------------------------------------------------===//
// AsmWriter Implementation
//===----------------------------------------------------------------------===//

static void WriteAsOperandInternal(raw_ostream &Out, const Value *V,
                                   TypePrinting *TypePrinter,
                                   SlotTracker *Machine,
                                   const Module *Context);



static const char *getPredicateText(unsigned predicate) {
  const char * pred = "unknown";
  switch (predicate) {
  case FCmpInst::FCMP_FALSE: pred = "false"; break;
  case FCmpInst::FCMP_OEQ:   pred = "oeq"; break;
  case FCmpInst::FCMP_OGT:   pred = "ogt"; break;
  case FCmpInst::FCMP_OGE:   pred = "oge"; break;
  case FCmpInst::FCMP_OLT:   pred = "olt"; break;
  case FCmpInst::FCMP_OLE:   pred = "ole"; break;
  case FCmpInst::FCMP_ONE:   pred = "one"; break;
  case FCmpInst::FCMP_ORD:   pred = "ord"; break;
  case FCmpInst::FCMP_UNO:   pred = "uno"; break;
  case FCmpInst::FCMP_UEQ:   pred = "ueq"; break;
  case FCmpInst::FCMP_UGT:   pred = "ugt"; break;
  case FCmpInst::FCMP_UGE:   pred = "uge"; break;
  case FCmpInst::FCMP_ULT:   pred = "ult"; break;
  case FCmpInst::FCMP_ULE:   pred = "ule"; break;
  case FCmpInst::FCMP_UNE:   pred = "une"; break;
  case FCmpInst::FCMP_TRUE:  pred = "true"; break;
  case ICmpInst::ICMP_EQ:    pred = "eq"; break;
  case ICmpInst::ICMP_NE:    pred = "ne"; break;
  case ICmpInst::ICMP_SGT:   pred = "sgt"; break;
  case ICmpInst::ICMP_SGE:   pred = "sge"; break;
  case ICmpInst::ICMP_SLT:   pred = "slt"; break;
  case ICmpInst::ICMP_SLE:   pred = "sle"; break;
  case ICmpInst::ICMP_UGT:   pred = "ugt"; break;
  case ICmpInst::ICMP_UGE:   pred = "uge"; break;
  case ICmpInst::ICMP_ULT:   pred = "ult"; break;
  case ICmpInst::ICMP_ULE:   pred = "ule"; break;
  }
  return pred;
}


static void WriteOptimizationInfo(raw_ostream &Out, const User *U) {
  if (const OverflowingBinaryOperator *OBO =
        dyn_cast<OverflowingBinaryOperator>(U)) {
    if (OBO->hasNoUnsignedWrap())
      Out << " nuw";
    if (OBO->hasNoSignedWrap())
      Out << " nsw";
  } else if (const PossiblyExactOperator *Div =
               dyn_cast<PossiblyExactOperator>(U)) {
    if (Div->isExact())
      Out << " exact";
  } else if (const GEPOperator *GEP = dyn_cast<GEPOperator>(U)) {
    if (GEP->isInBounds())
      Out << " inbounds";
  }
}

static void WriteConstantInternal(raw_ostream &Out, const Constant *CV,
                                  TypePrinting &TypePrinter,
                                  SlotTracker *Machine,
                                  const Module *Context) {
  if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV)) {
    if (CI->getType()->isIntegerTy(1)) {
      Out << (CI->getZExtValue() ? "true" : "false");
      return;
    }
    Out << CI->getValue();
    return;
  }

  if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CV)) {
    if (&CFP->getValueAPF().getSemantics() == &APFloat::IEEEdouble ||
        &CFP->getValueAPF().getSemantics() == &APFloat::IEEEsingle) {
      // We would like to output the FP constant value in exponential notation,
      // but we cannot do this if doing so will lose precision.  Check here to
      // make sure that we only output it in exponential format if we can parse
      // the value back and get the same value.
      //
      bool ignored;
      bool isDouble = &CFP->getValueAPF().getSemantics()==&APFloat::IEEEdouble;
      double Val = isDouble ? CFP->getValueAPF().convertToDouble() :
                              CFP->getValueAPF().convertToFloat();
      SmallString<128> StrVal;
      raw_svector_ostream(StrVal) << Val;

      // Check to make sure that the stringized number is not some string like
      // "Inf" or NaN, that atof will accept, but the lexer will not.  Check
      // that the string matches the "[-+]?[0-9]" regex.
      //
      if ((StrVal[0] >= '0' && StrVal[0] <= '9') ||
          ((StrVal[0] == '-' || StrVal[0] == '+') &&
           (StrVal[1] >= '0' && StrVal[1] <= '9'))) {
        // Reparse stringized version!
        if (atof(StrVal.c_str()) == Val) {
          Out << StrVal.str();
          return;
        }
      }
      // Otherwise we could not reparse it to exactly the same value, so we must
      // output the string in hexadecimal format!  Note that loading and storing
      // floating point types changes the bits of NaNs on some hosts, notably
      // x86, so we must not use these types.
      assert(sizeof(double) == sizeof(uint64_t) &&
             "assuming that double is 64 bits!");
      char Buffer[40];
      APFloat apf = CFP->getValueAPF();
      // Floats are represented in ASCII IR as double, convert.
      if (!isDouble)
        apf.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven,
                          &ignored);
      Out << "0x" <<
              utohex_buffer(uint64_t(apf.bitcastToAPInt().getZExtValue()),
                            Buffer+40);
      return;
    }

    // Some form of long double.  These appear as a magic letter identifying
    // the type, then a fixed number of hex digits.
    Out << "0x";
    if (&CFP->getValueAPF().getSemantics() == &APFloat::x87DoubleExtended) {
      Out << 'K';
      // api needed to prevent premature destruction
      APInt api = CFP->getValueAPF().bitcastToAPInt();
      const uint64_t* p = api.getRawData();
      uint64_t word = p[1];
      int shiftcount=12;
      int width = api.getBitWidth();
      for (int j=0; j<width; j+=4, shiftcount-=4) {
        unsigned int nibble = (word>>shiftcount) & 15;
        if (nibble < 10)
          Out << (unsigned char)(nibble + '0');
        else
          Out << (unsigned char)(nibble - 10 + 'A');
        if (shiftcount == 0 && j+4 < width) {
          word = *p;
          shiftcount = 64;
          if (width-j-4 < 64)
            shiftcount = width-j-4;
        }
      }
      return;
    } else if (&CFP->getValueAPF().getSemantics() == &APFloat::IEEEquad)
      Out << 'L';
    else if (&CFP->getValueAPF().getSemantics() == &APFloat::PPCDoubleDouble)
      Out << 'M';
    else
      llvm_unreachable("Unsupported floating point type");
    // api needed to prevent premature destruction
    APInt api = CFP->getValueAPF().bitcastToAPInt();
    const uint64_t* p = api.getRawData();
    uint64_t word = *p;
    int shiftcount=60;
    int width = api.getBitWidth();
    for (int j=0; j<width; j+=4, shiftcount-=4) {
      unsigned int nibble = (word>>shiftcount) & 15;
      if (nibble < 10)
        Out << (unsigned char)(nibble + '0');
      else
        Out << (unsigned char)(nibble - 10 + 'A');
      if (shiftcount == 0 && j+4 < width) {
        word = *(++p);
        shiftcount = 64;
        if (width-j-4 < 64)
          shiftcount = width-j-4;
      }
    }
    return;
  }

  if (isa<ConstantAggregateZero>(CV)) {
    Out << "zeroinitializer";
    return;
  }
  
  if (const BlockAddress *BA = dyn_cast<BlockAddress>(CV)) {
    Out << "blockaddress(";
    WriteAsOperandInternal(Out, BA->getFunction(), &TypePrinter, Machine,
                           Context);
    Out << ", ";
    WriteAsOperandInternal(Out, BA->getBasicBlock(), &TypePrinter, Machine,
                           Context);
    Out << ")";
    return;
  }

  if (const ConstantArray *CA = dyn_cast<ConstantArray>(CV)) {
    // As a special case, print the array as a string if it is an array of
    // i8 with ConstantInt values.
    //
    const Type *ETy = CA->getType()->getElementType();
    if (CA->isString()) {
      Out << "c\"";
      PrintEscapedString(CA->getAsString(), Out);
      Out << '"';
    } else {                // Cannot output in string format...
      Out << '[';
      if (CA->getNumOperands()) {
        TypePrinter.print(ETy, Out);
        Out << ' ';
        WriteAsOperandInternal(Out, CA->getOperand(0),
                               &TypePrinter, Machine,
                               Context);
        for (unsigned i = 1, e = CA->getNumOperands(); i != e; ++i) {
          Out << ", ";
          TypePrinter.print(ETy, Out);
          Out << ' ';
          WriteAsOperandInternal(Out, CA->getOperand(i), &TypePrinter, Machine,
                                 Context);
        }
      }
      Out << ']';
    }
    return;
  }

  if (const ConstantStruct *CS = dyn_cast<ConstantStruct>(CV)) {
    if (CS->getType()->isPacked())
      Out << '<';
    Out << '{';
    unsigned N = CS->getNumOperands();
    if (N) {
      Out << ' ';
      TypePrinter.print(CS->getOperand(0)->getType(), Out);
      Out << ' ';

      WriteAsOperandInternal(Out, CS->getOperand(0), &TypePrinter, Machine,
                             Context);

      for (unsigned i = 1; i < N; i++) {
        Out << ", ";
        TypePrinter.print(CS->getOperand(i)->getType(), Out);
        Out << ' ';

        WriteAsOperandInternal(Out, CS->getOperand(i), &TypePrinter, Machine,
                               Context);
      }
      Out << ' ';
    }

    Out << '}';
    if (CS->getType()->isPacked())
      Out << '>';
    return;
  }

  if (const ConstantVector *CP = dyn_cast<ConstantVector>(CV)) {
    const Type *ETy = CP->getType()->getElementType();
    assert(CP->getNumOperands() > 0 &&
           "Number of operands for a PackedConst must be > 0");
    Out << '<';
    TypePrinter.print(ETy, Out);
    Out << ' ';
    WriteAsOperandInternal(Out, CP->getOperand(0), &TypePrinter, Machine,
                           Context);
    for (unsigned i = 1, e = CP->getNumOperands(); i != e; ++i) {
      Out << ", ";
      TypePrinter.print(ETy, Out);
      Out << ' ';
      WriteAsOperandInternal(Out, CP->getOperand(i), &TypePrinter, Machine,
                             Context);
    }
    Out << '>';
    return;
  }

  if (isa<ConstantPointerNull>(CV)) {
    Out << "null";
    return;
  }

  if (isa<UndefValue>(CV)) {
    Out << "undef";
    return;
  }

  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV)) {
    Out << CE->getOpcodeName();
    WriteOptimizationInfo(Out, CE);
    if (CE->isCompare())
      Out << ' ' << getPredicateText(CE->getPredicate());
    Out << " (";

    for (User::const_op_iterator OI=CE->op_begin(); OI != CE->op_end(); ++OI) {
      TypePrinter.print((*OI)->getType(), Out);
      Out << ' ';
      WriteAsOperandInternal(Out, *OI, &TypePrinter, Machine, Context);
      if (OI+1 != CE->op_end())
        Out << ", ";
    }

    if (CE->hasIndices()) {
      ArrayRef<unsigned> Indices = CE->getIndices();
      for (unsigned i = 0, e = Indices.size(); i != e; ++i)
        Out << ", " << Indices[i];
    }

    if (CE->isCast()) {
      Out << " to ";
      TypePrinter.print(CE->getType(), Out);
    }

    Out << ')';
    return;
  }

  Out << "<placeholder or erroneous Constant>";
}

static void WriteMDNodeBodyInternal(raw_ostream &Out, const MDNode *Node,
                                    TypePrinting *TypePrinter,
                                    SlotTracker *Machine,
                                    const Module *Context) {
  Out << "!{";
  for (unsigned mi = 0, me = Node->getNumOperands(); mi != me; ++mi) {
    const Value *V = Node->getOperand(mi);
    if (V == 0)
      Out << "null";
    else {
      TypePrinter->print(V->getType(), Out);
      Out << ' ';
      WriteAsOperandInternal(Out, Node->getOperand(mi), 
                             TypePrinter, Machine, Context);
    }
    if (mi + 1 != me)
      Out << ", ";
  }
  
  Out << "}";
}


/// WriteAsOperand - Write the name of the specified value out to the specified
/// ostream.  This can be useful when you just want to print int %reg126, not
/// the whole instruction that generated it.
///
static void WriteAsOperandInternal(raw_ostream &Out, const Value *V,
                                   TypePrinting *TypePrinter,
                                   SlotTracker *Machine,
                                   const Module *Context) {
  if (V->hasName()) {
    PrintLLVMName(Out, V);
    return;
  }

  const Constant *CV = dyn_cast<Constant>(V);
  if (CV && !isa<GlobalValue>(CV)) {
    assert(TypePrinter && "Constants require TypePrinting!");
    WriteConstantInternal(Out, CV, *TypePrinter, Machine, Context);
    return;
  }

  if (const InlineAsm *IA = dyn_cast<InlineAsm>(V)) {
    Out << "asm ";
    if (IA->hasSideEffects())
      Out << "sideeffect ";
    if (IA->isAlignStack())
      Out << "alignstack ";
    Out << '"';
    PrintEscapedString(IA->getAsmString(), Out);
    Out << "\", \"";
    PrintEscapedString(IA->getConstraintString(), Out);
    Out << '"';
    return;
  }

  if (const MDNode *N = dyn_cast<MDNode>(V)) {
    if (N->isFunctionLocal()) {
      // Print metadata inline, not via slot reference number.
      WriteMDNodeBodyInternal(Out, N, TypePrinter, Machine, Context);
      return;
    }
  
    if (!Machine) {
      if (N->isFunctionLocal())
        Machine = new SlotTracker(N->getFunction());
      else
        Machine = new SlotTracker(Context);
    }
    int Slot = Machine->getMetadataSlot(N);
    if (Slot == -1)
      Out << "<badref>";
    else
      Out << '!' << Slot;
    return;
  }

  if (const MDString *MDS = dyn_cast<MDString>(V)) {
    Out << "!\"";
    PrintEscapedString(MDS->getString(), Out);
    Out << '"';
    return;
  }

  if (V->getValueID() == Value::PseudoSourceValueVal ||
      V->getValueID() == Value::FixedStackPseudoSourceValueVal) {
    V->print(Out);
    return;
  }

  char Prefix = '%';
  int Slot;
  if (Machine) {
    if (const GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
      Slot = Machine->getGlobalSlot(GV);
      Prefix = '@';
    } else {
      Slot = Machine->getLocalSlot(V);
    }
  } else {
    Machine = createSlotTracker(V);
    if (Machine) {
      if (const GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
        Slot = Machine->getGlobalSlot(GV);
        Prefix = '@';
      } else {
        Slot = Machine->getLocalSlot(V);
      }
      delete Machine;
    } else {
      Slot = -1;
    }
  }

  if (Slot != -1)
    Out << Prefix << Slot;
  else
    Out << "<badref>";
}

void llvm::WriteAsOperand(raw_ostream &Out, const Value *V,
                          bool PrintType, const Module *Context) {

  // Fast path: Don't construct and populate a TypePrinting object if we
  // won't be needing any types printed.
  if (!PrintType &&
      ((!isa<Constant>(V) && !isa<MDNode>(V)) ||
       V->hasName() || isa<GlobalValue>(V))) {
    WriteAsOperandInternal(Out, V, 0, 0, Context);
    return;
  }

  if (Context == 0) Context = getModuleFromVal(V);

  TypePrinting TypePrinter;
  std::vector<const Type*> NumberedTypes;
  AddModuleTypesToPrinter(TypePrinter, NumberedTypes, Context);
  if (PrintType) {
    TypePrinter.print(V->getType(), Out);
    Out << ' ';
  }

  WriteAsOperandInternal(Out, V, &TypePrinter, 0, Context);
}

namespace {

class AssemblyWriter {
  formatted_raw_ostream &Out;
  SlotTracker &Machine;
  const Module *TheModule;
  TypePrinting TypePrinter;
  AssemblyAnnotationWriter *AnnotationWriter;
  std::vector<const Type*> NumberedTypes;
  
public:
  inline AssemblyWriter(formatted_raw_ostream &o, SlotTracker &Mac,
                        const Module *M,
                        AssemblyAnnotationWriter *AAW)
    : Out(o), Machine(Mac), TheModule(M), AnnotationWriter(AAW) {
    AddModuleTypesToPrinter(TypePrinter, NumberedTypes, M);
  }

  void printMDNodeBody(const MDNode *MD);
  void printNamedMDNode(const NamedMDNode *NMD);
  
  void printModule(const Module *M);

  void writeOperand(const Value *Op, bool PrintType);
  void writeParamOperand(const Value *Operand, Attributes Attrs);

  void writeAllMDNodes();

  void printTypeSymbolTable(const TypeSymbolTable &ST);
  void printGlobal(const GlobalVariable *GV);
  void printAlias(const GlobalAlias *GV);
  void printFunction(const Function *F);
  void printArgument(const Argument *FA, Attributes Attrs);
  void printBasicBlock(const BasicBlock *BB);
  void printInstruction(const Instruction &I);

private:
  // printInfoComment - Print a little comment after the instruction indicating
  // which slot it occupies.
  void printInfoComment(const Value &V);
};
}  // end of anonymous namespace

void AssemblyWriter::writeOperand(const Value *Operand, bool PrintType) {
  if (Operand == 0) {
    Out << "<null operand!>";
    return;
  }
  if (PrintType) {
    TypePrinter.print(Operand->getType(), Out);
    Out << ' ';
  }
  WriteAsOperandInternal(Out, Operand, &TypePrinter, &Machine, TheModule);
}

void AssemblyWriter::writeParamOperand(const Value *Operand,
                                       Attributes Attrs) {
  if (Operand == 0) {
    Out << "<null operand!>";
    return;
  }

  // Print the type
  TypePrinter.print(Operand->getType(), Out);
  // Print parameter attributes list
  if (Attrs != Attribute::None)
    Out << ' ' << Attribute::getAsString(Attrs);
  Out << ' ';
  // Print the operand
  WriteAsOperandInternal(Out, Operand, &TypePrinter, &Machine, TheModule);
}

void AssemblyWriter::printModule(const Module *M) {
  if (!M->getModuleIdentifier().empty() &&
      // Don't print the ID if it will start a new line (which would
      // require a comment char before it).
      M->getModuleIdentifier().find('\n') == std::string::npos)
    Out << "; ModuleID = '" << M->getModuleIdentifier() << "'\n";

  if (!M->getDataLayout().empty())
    Out << "target datalayout = \"" << M->getDataLayout() << "\"\n";
  if (!M->getTargetTriple().empty())
    Out << "target triple = \"" << M->getTargetTriple() << "\"\n";

  if (!M->getModuleInlineAsm().empty()) {
    // Split the string into lines, to make it easier to read the .ll file.
    std::string Asm = M->getModuleInlineAsm();
    size_t CurPos = 0;
    size_t NewLine = Asm.find_first_of('\n', CurPos);
    Out << '\n';
    while (NewLine != std::string::npos) {
      // We found a newline, print the portion of the asm string from the
      // last newline up to this newline.
      Out << "module asm \"";
      PrintEscapedString(std::string(Asm.begin()+CurPos, Asm.begin()+NewLine),
                         Out);
      Out << "\"\n";
      CurPos = NewLine+1;
      NewLine = Asm.find_first_of('\n', CurPos);
    }
    std::string rest(Asm.begin()+CurPos, Asm.end());
    if (!rest.empty()) {
      Out << "module asm \"";
      PrintEscapedString(rest, Out);
      Out << "\"\n";
    }
  }

  // Loop over the dependent libraries and emit them.
  Module::lib_iterator LI = M->lib_begin();
  Module::lib_iterator LE = M->lib_end();
  if (LI != LE) {
    Out << '\n';
    Out << "deplibs = [ ";
    while (LI != LE) {
      Out << '"' << *LI << '"';
      ++LI;
      if (LI != LE)
        Out << ", ";
    }
    Out << " ]";
  }

  // Loop over the symbol table, emitting all id'd types.
  if (!M->getTypeSymbolTable().empty() || !NumberedTypes.empty()) Out << '\n';
  printTypeSymbolTable(M->getTypeSymbolTable());

  // Output all globals.
  if (!M->global_empty()) Out << '\n';
  for (Module::const_global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I)
    printGlobal(I);

  // Output all aliases.
  if (!M->alias_empty()) Out << "\n";
  for (Module::const_alias_iterator I = M->alias_begin(), E = M->alias_end();
       I != E; ++I)
    printAlias(I);

  // Output all of the functions.
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
    printFunction(I);

  // Output named metadata.
  if (!M->named_metadata_empty()) Out << '\n';
  
  for (Module::const_named_metadata_iterator I = M->named_metadata_begin(),
       E = M->named_metadata_end(); I != E; ++I)
    printNamedMDNode(I);

  // Output metadata.
  if (!Machine.mdn_empty()) {
    Out << '\n';
    writeAllMDNodes();
  }
}

void AssemblyWriter::printNamedMDNode(const NamedMDNode *NMD) {
  Out << "!" << NMD->getName() << " = !{";
  for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i) {
    if (i) Out << ", ";
    int Slot = Machine.getMetadataSlot(NMD->getOperand(i));
    if (Slot == -1)
      Out << "<badref>";
    else
      Out << '!' << Slot;
  }
  Out << "}\n";
}


static void PrintLinkage(GlobalValue::LinkageTypes LT,
                         formatted_raw_ostream &Out) {
  switch (LT) {
  case GlobalValue::ExternalLinkage: break;
  case GlobalValue::PrivateLinkage:       Out << "private ";        break;
  case GlobalValue::LinkerPrivateLinkage: Out << "linker_private "; break;
  case GlobalValue::LinkerPrivateWeakLinkage:
    Out << "linker_private_weak ";
    break;
  case GlobalValue::LinkerPrivateWeakDefAutoLinkage:
    Out << "linker_private_weak_def_auto ";
    break;
  case GlobalValue::InternalLinkage:      Out << "internal ";       break;
  case GlobalValue::LinkOnceAnyLinkage:   Out << "linkonce ";       break;
  case GlobalValue::LinkOnceODRLinkage:   Out << "linkonce_odr ";   break;
  case GlobalValue::WeakAnyLinkage:       Out << "weak ";           break;
  case GlobalValue::WeakODRLinkage:       Out << "weak_odr ";       break;
  case GlobalValue::CommonLinkage:        Out << "common ";         break;
  case GlobalValue::AppendingLinkage:     Out << "appending ";      break;
  case GlobalValue::DLLImportLinkage:     Out << "dllimport ";      break;
  case GlobalValue::DLLExportLinkage:     Out << "dllexport ";      break;
  case GlobalValue::ExternalWeakLinkage:  Out << "extern_weak ";    break;
  case GlobalValue::AvailableExternallyLinkage:
    Out << "available_externally ";
    break;
  }
}


static void PrintVisibility(GlobalValue::VisibilityTypes Vis,
                            formatted_raw_ostream &Out) {
  switch (Vis) {
  case GlobalValue::DefaultVisibility: break;
  case GlobalValue::HiddenVisibility:    Out << "hidden "; break;
  case GlobalValue::ProtectedVisibility: Out << "protected "; break;
  }
}

void AssemblyWriter::printGlobal(const GlobalVariable *GV) {
  if (GV->isMaterializable())
    Out << "; Materializable\n";

  WriteAsOperandInternal(Out, GV, &TypePrinter, &Machine, GV->getParent());
  Out << " = ";

  if (!GV->hasInitializer() && GV->hasExternalLinkage())
    Out << "external ";

  PrintLinkage(GV->getLinkage(), Out);
  PrintVisibility(GV->getVisibility(), Out);

  if (GV->isThreadLocal()) Out << "thread_local ";
  if (unsigned AddressSpace = GV->getType()->getAddressSpace())
    Out << "addrspace(" << AddressSpace << ") ";
  if (GV->hasUnnamedAddr()) Out << "unnamed_addr ";
  Out << (GV->isConstant() ? "constant " : "global ");
  TypePrinter.print(GV->getType()->getElementType(), Out);

  if (GV->hasInitializer()) {
    Out << ' ';
    writeOperand(GV->getInitializer(), false);
  }

  if (GV->hasSection()) {
    Out << ", section \"";
    PrintEscapedString(GV->getSection(), Out);
    Out << '"';
  }
  if (GV->getAlignment())
    Out << ", align " << GV->getAlignment();

  printInfoComment(*GV);
  Out << '\n';
}

void AssemblyWriter::printAlias(const GlobalAlias *GA) {
  if (GA->isMaterializable())
    Out << "; Materializable\n";

  // Don't crash when dumping partially built GA
  if (!GA->hasName())
    Out << "<<nameless>> = ";
  else {
    PrintLLVMName(Out, GA);
    Out << " = ";
  }
  PrintVisibility(GA->getVisibility(), Out);

  Out << "alias ";

  PrintLinkage(GA->getLinkage(), Out);

  const Constant *Aliasee = GA->getAliasee();

  if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(Aliasee)) {
    TypePrinter.print(GV->getType(), Out);
    Out << ' ';
    PrintLLVMName(Out, GV);
  } else if (const Function *F = dyn_cast<Function>(Aliasee)) {
    TypePrinter.print(F->getFunctionType(), Out);
    Out << "* ";

    WriteAsOperandInternal(Out, F, &TypePrinter, &Machine, F->getParent());
  } else if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(Aliasee)) {
    TypePrinter.print(GA->getType(), Out);
    Out << ' ';
    PrintLLVMName(Out, GA);
  } else {
    const ConstantExpr *CE = cast<ConstantExpr>(Aliasee);
    // The only valid GEP is an all zero GEP.
    assert((CE->getOpcode() == Instruction::BitCast ||
            CE->getOpcode() == Instruction::GetElementPtr) &&
           "Unsupported aliasee");
    writeOperand(CE, false);
  }

  printInfoComment(*GA);
  Out << '\n';
}

void AssemblyWriter::printTypeSymbolTable(const TypeSymbolTable &ST) {
  // Emit all numbered types.
  for (unsigned i = 0, e = NumberedTypes.size(); i != e; ++i) {
    Out << '%' << i << " = type ";

    // Make sure we print out at least one level of the type structure, so
    // that we do not get %2 = type %2
    TypePrinter.printAtLeastOneLevel(NumberedTypes[i], Out);
    Out << '\n';
  }

  // Print the named types.
  for (TypeSymbolTable::const_iterator TI = ST.begin(), TE = ST.end();
       TI != TE; ++TI) {
    PrintLLVMName(Out, TI->first, LocalPrefix);
    Out << " = type ";

    // Make sure we print out at least one level of the type structure, so
    // that we do not get %FILE = type %FILE
    TypePrinter.printAtLeastOneLevel(TI->second, Out);
    Out << '\n';
  }
}

/// printFunction - Print all aspects of a function.
///
void AssemblyWriter::printFunction(const Function *F) {
  // Print out the return type and name.
  Out << '\n';

  if (AnnotationWriter) AnnotationWriter->emitFunctionAnnot(F, Out);

  if (F->isMaterializable())
    Out << "; Materializable\n";

  if (F->isDeclaration())
    Out << "declare ";
  else
    Out << "define ";

  PrintLinkage(F->getLinkage(), Out);
  PrintVisibility(F->getVisibility(), Out);

  // Print the calling convention.
  switch (F->getCallingConv()) {
  case CallingConv::C: break;   // default
  case CallingConv::Fast:         Out << "fastcc "; break;
  case CallingConv::Cold:         Out << "coldcc "; break;
  case CallingConv::X86_StdCall:  Out << "x86_stdcallcc "; break;
  case CallingConv::X86_FastCall: Out << "x86_fastcallcc "; break;
  case CallingConv::X86_ThisCall: Out << "x86_thiscallcc "; break;
  case CallingConv::ARM_APCS:     Out << "arm_apcscc "; break;
  case CallingConv::ARM_AAPCS:    Out << "arm_aapcscc "; break;
  case CallingConv::ARM_AAPCS_VFP:Out << "arm_aapcs_vfpcc "; break;
  case CallingConv::MSP430_INTR:  Out << "msp430_intrcc "; break;
  case CallingConv::PTX_Kernel:   Out << "ptx_kernel "; break;
  case CallingConv::PTX_Device:   Out << "ptx_device "; break;
  default: Out << "cc" << F->getCallingConv() << " "; break;
  }

  const FunctionType *FT = F->getFunctionType();
  const AttrListPtr &Attrs = F->getAttributes();
  Attributes RetAttrs = Attrs.getRetAttributes();
  if (RetAttrs != Attribute::None)
    Out <<  Attribute::getAsString(Attrs.getRetAttributes()) << ' ';
  TypePrinter.print(F->getReturnType(), Out);
  Out << ' ';
  WriteAsOperandInternal(Out, F, &TypePrinter, &Machine, F->getParent());
  Out << '(';
  Machine.incorporateFunction(F);

  // Loop over the arguments, printing them...

  unsigned Idx = 1;
  if (!F->isDeclaration()) {
    // If this isn't a declaration, print the argument names as well.
    for (Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
         I != E; ++I) {
      // Insert commas as we go... the first arg doesn't get a comma
      if (I != F->arg_begin()) Out << ", ";
      printArgument(I, Attrs.getParamAttributes(Idx));
      Idx++;
    }
  } else {
    // Otherwise, print the types from the function type.
    for (unsigned i = 0, e = FT->getNumParams(); i != e; ++i) {
      // Insert commas as we go... the first arg doesn't get a comma
      if (i) Out << ", ";

      // Output type...
      TypePrinter.print(FT->getParamType(i), Out);

      Attributes ArgAttrs = Attrs.getParamAttributes(i+1);
      if (ArgAttrs != Attribute::None)
        Out << ' ' << Attribute::getAsString(ArgAttrs);
    }
  }

  // Finish printing arguments...
  if (FT->isVarArg()) {
    if (FT->getNumParams()) Out << ", ";
    Out << "...";  // Output varargs portion of signature!
  }
  Out << ')';
  if (F->hasUnnamedAddr())
    Out << " unnamed_addr";
  Attributes FnAttrs = Attrs.getFnAttributes();
  if (FnAttrs != Attribute::None)
    Out << ' ' << Attribute::getAsString(Attrs.getFnAttributes());
  if (F->hasSection()) {
    Out << " section \"";
    PrintEscapedString(F->getSection(), Out);
    Out << '"';
  }
  if (F->getAlignment())
    Out << " align " << F->getAlignment();
  if (F->hasGC())
    Out << " gc \"" << F->getGC() << '"';
  if (F->isDeclaration()) {
    Out << '\n';
  } else {
    Out << " {";
    // Output all of the function's basic blocks.
    for (Function::const_iterator I = F->begin(), E = F->end(); I != E; ++I)
      printBasicBlock(I);

    Out << "}\n";
  }

  Machine.purgeFunction();
}

/// printArgument - This member is called for every argument that is passed into
/// the function.  Simply print it out
///
void AssemblyWriter::printArgument(const Argument *Arg,
                                   Attributes Attrs) {
  // Output type...
  TypePrinter.print(Arg->getType(), Out);

  // Output parameter attributes list
  if (Attrs != Attribute::None)
    Out << ' ' << Attribute::getAsString(Attrs);

  // Output name, if available...
  if (Arg->hasName()) {
    Out << ' ';
    PrintLLVMName(Out, Arg);
  }
}

/// printBasicBlock - This member is called for each basic block in a method.
///
void AssemblyWriter::printBasicBlock(const BasicBlock *BB) {
  if (BB->hasName()) {              // Print out the label if it exists...
    Out << "\n";
    PrintLLVMName(Out, BB->getName(), LabelPrefix);
    Out << ':';
  } else if (!BB->use_empty()) {      // Don't print block # of no uses...
    Out << "\n; <label>:";
    int Slot = Machine.getLocalSlot(BB);
    if (Slot != -1)
      Out << Slot;
    else
      Out << "<badref>";
  }

  if (BB->getParent() == 0) {
    Out.PadToColumn(50);
    Out << "; Error: Block without parent!";
  } else if (BB != &BB->getParent()->getEntryBlock()) {  // Not the entry block?
    // Output predecessors for the block.
    Out.PadToColumn(50);
    Out << ";";
    const_pred_iterator PI = pred_begin(BB), PE = pred_end(BB);

    if (PI == PE) {
      Out << " No predecessors!";
    } else {
      Out << " preds = ";
      writeOperand(*PI, false);
      for (++PI; PI != PE; ++PI) {
        Out << ", ";
        writeOperand(*PI, false);
      }
    }
  }

  Out << "\n";

  if (AnnotationWriter) AnnotationWriter->emitBasicBlockStartAnnot(BB, Out);

  // Output all of the instructions in the basic block...
  for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
    printInstruction(*I);
    Out << '\n';
  }

  if (AnnotationWriter) AnnotationWriter->emitBasicBlockEndAnnot(BB, Out);
}

/// printInfoComment - Print a little comment after the instruction indicating
/// which slot it occupies.
///
void AssemblyWriter::printInfoComment(const Value &V) {
  if (AnnotationWriter) {
    AnnotationWriter->printInfoComment(V, Out);
    return;
  }
}

// This member is called for each Instruction in a function..
void AssemblyWriter::printInstruction(const Instruction &I) {
  if (AnnotationWriter) AnnotationWriter->emitInstructionAnnot(&I, Out);

  // Print out indentation for an instruction.
  Out << "  ";

  // Print out name if it exists...
  if (I.hasName()) {
    PrintLLVMName(Out, &I);
    Out << " = ";
  } else if (!I.getType()->isVoidTy()) {
    // Print out the def slot taken.
    int SlotNum = Machine.getLocalSlot(&I);
    if (SlotNum == -1)
      Out << "<badref> = ";
    else
      Out << '%' << SlotNum << " = ";
  }

  // If this is a volatile load or store, print out the volatile marker.
  if ((isa<LoadInst>(I)  && cast<LoadInst>(I).isVolatile()) ||
      (isa<StoreInst>(I) && cast<StoreInst>(I).isVolatile())) {
      Out << "volatile ";
  } else if (isa<CallInst>(I) && cast<CallInst>(I).isTailCall()) {
    // If this is a call, check if it's a tail call.
    Out << "tail ";
  }

  // Print out the opcode...
  Out << I.getOpcodeName();

  // Print out optimization information.
  WriteOptimizationInfo(Out, &I);

  // Print out the compare instruction predicates
  if (const CmpInst *CI = dyn_cast<CmpInst>(&I))
    Out << ' ' << getPredicateText(CI->getPredicate());

  // Print out the type of the operands...
  const Value *Operand = I.getNumOperands() ? I.getOperand(0) : 0;

  // Special case conditional branches to swizzle the condition out to the front
  if (isa<BranchInst>(I) && cast<BranchInst>(I).isConditional()) {
    BranchInst &BI(cast<BranchInst>(I));
    Out << ' ';
    writeOperand(BI.getCondition(), true);
    Out << ", ";
    writeOperand(BI.getSuccessor(0), true);
    Out << ", ";
    writeOperand(BI.getSuccessor(1), true);

  } else if (isa<SwitchInst>(I)) {
    // Special case switch instruction to get formatting nice and correct.
    Out << ' ';
    writeOperand(Operand        , true);
    Out << ", ";
    writeOperand(I.getOperand(1), true);
    Out << " [";

    for (unsigned op = 2, Eop = I.getNumOperands(); op < Eop; op += 2) {
      Out << "\n    ";
      writeOperand(I.getOperand(op  ), true);
      Out << ", ";
      writeOperand(I.getOperand(op+1), true);
    }
    Out << "\n  ]";
  } else if (isa<IndirectBrInst>(I)) {
    // Special case indirectbr instruction to get formatting nice and correct.
    Out << ' ';
    writeOperand(Operand, true);
    Out << ", [";
    
    for (unsigned i = 1, e = I.getNumOperands(); i != e; ++i) {
      if (i != 1)
        Out << ", ";
      writeOperand(I.getOperand(i), true);
    }
    Out << ']';
  } else if (isa<PHINode>(I)) {
    Out << ' ';
    TypePrinter.print(I.getType(), Out);
    Out << ' ';

    for (unsigned op = 0, Eop = I.getNumOperands(); op < Eop; op += 2) {
      if (op) Out << ", ";
      Out << "[ ";
      writeOperand(I.getOperand(op  ), false); Out << ", ";
      writeOperand(I.getOperand(op+1), false); Out << " ]";
    }
  } else if (const ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(&I)) {
    Out << ' ';
    writeOperand(I.getOperand(0), true);
    for (const unsigned *i = EVI->idx_begin(), *e = EVI->idx_end(); i != e; ++i)
      Out << ", " << *i;
  } else if (const InsertValueInst *IVI = dyn_cast<InsertValueInst>(&I)) {
    Out << ' ';
    writeOperand(I.getOperand(0), true); Out << ", ";
    writeOperand(I.getOperand(1), true);
    for (const unsigned *i = IVI->idx_begin(), *e = IVI->idx_end(); i != e; ++i)
      Out << ", " << *i;
  } else if (isa<ReturnInst>(I) && !Operand) {
    Out << " void";
  } else if (const CallInst *CI = dyn_cast<CallInst>(&I)) {
    // Print the calling convention being used.
    switch (CI->getCallingConv()) {
    case CallingConv::C: break;   // default
    case CallingConv::Fast:  Out << " fastcc"; break;
    case CallingConv::Cold:  Out << " coldcc"; break;
    case CallingConv::X86_StdCall:  Out << " x86_stdcallcc"; break;
    case CallingConv::X86_FastCall: Out << " x86_fastcallcc"; break;
    case CallingConv::X86_ThisCall: Out << " x86_thiscallcc"; break;
    case CallingConv::ARM_APCS:     Out << " arm_apcscc "; break;
    case CallingConv::ARM_AAPCS:    Out << " arm_aapcscc "; break;
    case CallingConv::ARM_AAPCS_VFP:Out << " arm_aapcs_vfpcc "; break;
    case CallingConv::MSP430_INTR:  Out << " msp430_intrcc "; break;
    case CallingConv::PTX_Kernel:   Out << " ptx_kernel"; break;
    case CallingConv::PTX_Device:   Out << " ptx_device"; break;
    default: Out << " cc" << CI->getCallingConv(); break;
    }

    Operand = CI->getCalledValue();
    const PointerType    *PTy = cast<PointerType>(Operand->getType());
    const FunctionType   *FTy = cast<FunctionType>(PTy->getElementType());
    const Type         *RetTy = FTy->getReturnType();
    const AttrListPtr &PAL = CI->getAttributes();

    if (PAL.getRetAttributes() != Attribute::None)
      Out << ' ' << Attribute::getAsString(PAL.getRetAttributes());

    // If possible, print out the short form of the call instruction.  We can
    // only do this if the first argument is a pointer to a nonvararg function,
    // and if the return type is not a pointer to a function.
    //
    Out << ' ';
    if (!FTy->isVarArg() &&
        (!RetTy->isPointerTy() ||
         !cast<PointerType>(RetTy)->getElementType()->isFunctionTy())) {
      TypePrinter.print(RetTy, Out);
      Out << ' ';
      writeOperand(Operand, false);
    } else {
      writeOperand(Operand, true);
    }
    Out << '(';
    for (unsigned op = 0, Eop = CI->getNumArgOperands(); op < Eop; ++op) {
      if (op > 0)
        Out << ", ";
      writeParamOperand(CI->getArgOperand(op), PAL.getParamAttributes(op + 1));
    }
    Out << ')';
    if (PAL.getFnAttributes() != Attribute::None)
      Out << ' ' << Attribute::getAsString(PAL.getFnAttributes());
  } else if (const InvokeInst *II = dyn_cast<InvokeInst>(&I)) {
    Operand = II->getCalledValue();
    const PointerType    *PTy = cast<PointerType>(Operand->getType());
    const FunctionType   *FTy = cast<FunctionType>(PTy->getElementType());
    const Type         *RetTy = FTy->getReturnType();
    const AttrListPtr &PAL = II->getAttributes();

    // Print the calling convention being used.
    switch (II->getCallingConv()) {
    case CallingConv::C: break;   // default
    case CallingConv::Fast:  Out << " fastcc"; break;
    case CallingConv::Cold:  Out << " coldcc"; break;
    case CallingConv::X86_StdCall:  Out << " x86_stdcallcc"; break;
    case CallingConv::X86_FastCall: Out << " x86_fastcallcc"; break;
    case CallingConv::X86_ThisCall: Out << " x86_thiscallcc"; break;
    case CallingConv::ARM_APCS:     Out << " arm_apcscc "; break;
    case CallingConv::ARM_AAPCS:    Out << " arm_aapcscc "; break;
    case CallingConv::ARM_AAPCS_VFP:Out << " arm_aapcs_vfpcc "; break;
    case CallingConv::MSP430_INTR:  Out << " msp430_intrcc "; break;
    case CallingConv::PTX_Kernel:   Out << " ptx_kernel"; break;
    case CallingConv::PTX_Device:   Out << " ptx_device"; break;
    default: Out << " cc" << II->getCallingConv(); break;
    }

    if (PAL.getRetAttributes() != Attribute::None)
      Out << ' ' << Attribute::getAsString(PAL.getRetAttributes());

    // If possible, print out the short form of the invoke instruction. We can
    // only do this if the first argument is a pointer to a nonvararg function,
    // and if the return type is not a pointer to a function.
    //
    Out << ' ';
    if (!FTy->isVarArg() &&
        (!RetTy->isPointerTy() ||
         !cast<PointerType>(RetTy)->getElementType()->isFunctionTy())) {
      TypePrinter.print(RetTy, Out);
      Out << ' ';
      writeOperand(Operand, false);
    } else {
      writeOperand(Operand, true);
    }
    Out << '(';
    for (unsigned op = 0, Eop = II->getNumArgOperands(); op < Eop; ++op) {
      if (op)
        Out << ", ";
      writeParamOperand(II->getArgOperand(op), PAL.getParamAttributes(op + 1));
    }

    Out << ')';
    if (PAL.getFnAttributes() != Attribute::None)
      Out << ' ' << Attribute::getAsString(PAL.getFnAttributes());

    Out << "\n          to ";
    writeOperand(II->getNormalDest(), true);
    Out << " unwind ";
    writeOperand(II->getUnwindDest(), true);

  } else if (const AllocaInst *AI = dyn_cast<AllocaInst>(&I)) {
    Out << ' ';
    TypePrinter.print(AI->getType()->getElementType(), Out);
    if (!AI->getArraySize() || AI->isArrayAllocation()) {
      Out << ", ";
      writeOperand(AI->getArraySize(), true);
    }
    if (AI->getAlignment()) {
      Out << ", align " << AI->getAlignment();
    }
  } else if (isa<CastInst>(I)) {
    if (Operand) {
      Out << ' ';
      writeOperand(Operand, true);   // Work with broken code
    }
    Out << " to ";
    TypePrinter.print(I.getType(), Out);
  } else if (isa<VAArgInst>(I)) {
    if (Operand) {
      Out << ' ';
      writeOperand(Operand, true);   // Work with broken code
    }
    Out << ", ";
    TypePrinter.print(I.getType(), Out);
  } else if (Operand) {   // Print the normal way.

    // PrintAllTypes - Instructions who have operands of all the same type
    // omit the type from all but the first operand.  If the instruction has
    // different type operands (for example br), then they are all printed.
    bool PrintAllTypes = false;
    const Type *TheType = Operand->getType();

    // Select, Store and ShuffleVector always print all types.
    if (isa<SelectInst>(I) || isa<StoreInst>(I) || isa<ShuffleVectorInst>(I)
        || isa<ReturnInst>(I)) {
      PrintAllTypes = true;
    } else {
      for (unsigned i = 1, E = I.getNumOperands(); i != E; ++i) {
        Operand = I.getOperand(i);
        // note that Operand shouldn't be null, but the test helps make dump()
        // more tolerant of malformed IR
        if (Operand && Operand->getType() != TheType) {
          PrintAllTypes = true;    // We have differing types!  Print them all!
          break;
        }
      }
    }

    if (!PrintAllTypes) {
      Out << ' ';
      TypePrinter.print(TheType, Out);
    }

    Out << ' ';
    for (unsigned i = 0, E = I.getNumOperands(); i != E; ++i) {
      if (i) Out << ", ";
      writeOperand(I.getOperand(i), PrintAllTypes);
    }
  }

  // Print post operand alignment for load/store.
  if (isa<LoadInst>(I) && cast<LoadInst>(I).getAlignment()) {
    Out << ", align " << cast<LoadInst>(I).getAlignment();
  } else if (isa<StoreInst>(I) && cast<StoreInst>(I).getAlignment()) {
    Out << ", align " << cast<StoreInst>(I).getAlignment();
  }

  // Print Metadata info.
  SmallVector<std::pair<unsigned, MDNode*>, 4> InstMD;
  I.getAllMetadata(InstMD);
  if (!InstMD.empty()) {
    SmallVector<StringRef, 8> MDNames;
    I.getType()->getContext().getMDKindNames(MDNames);
    for (unsigned i = 0, e = InstMD.size(); i != e; ++i) {
      unsigned Kind = InstMD[i].first;
       if (Kind < MDNames.size()) {
         Out << ", !" << MDNames[Kind];
      } else {
        Out << ", !<unknown kind #" << Kind << ">";
      }
      Out << ' ';
      WriteAsOperandInternal(Out, InstMD[i].second, &TypePrinter, &Machine,
                             TheModule);
    }
  }
  printInfoComment(I);
}

static void WriteMDNodeComment(const MDNode *Node,
                               formatted_raw_ostream &Out) {
  if (Node->getNumOperands() < 1)
    return;
  ConstantInt *CI = dyn_cast_or_null<ConstantInt>(Node->getOperand(0));
  if (!CI) return;
  APInt Val = CI->getValue();
  APInt Tag = Val & ~APInt(Val.getBitWidth(), LLVMDebugVersionMask);
  if (Val.ult(LLVMDebugVersion))
    return;
  
  Out.PadToColumn(50);
  if (Tag == dwarf::DW_TAG_user_base)
    Out << "; [ DW_TAG_user_base ]";
  else if (Tag.isIntN(32)) {
    if (const char *TagName = dwarf::TagString(Tag.getZExtValue()))
      Out << "; [ " << TagName << " ]";
  }
}

void AssemblyWriter::writeAllMDNodes() {
  SmallVector<const MDNode *, 16> Nodes;
  Nodes.resize(Machine.mdn_size());
  for (SlotTracker::mdn_iterator I = Machine.mdn_begin(), E = Machine.mdn_end();
       I != E; ++I)
    Nodes[I->second] = cast<MDNode>(I->first);
  
  for (unsigned i = 0, e = Nodes.size(); i != e; ++i) {
    Out << '!' << i << " = metadata ";
    printMDNodeBody(Nodes[i]);
  }
}

void AssemblyWriter::printMDNodeBody(const MDNode *Node) {
  WriteMDNodeBodyInternal(Out, Node, &TypePrinter, &Machine, TheModule);
  WriteMDNodeComment(Node, Out);
  Out << "\n";
}

//===----------------------------------------------------------------------===//
//                       External Interface declarations
//===----------------------------------------------------------------------===//

void Module::print(raw_ostream &ROS, AssemblyAnnotationWriter *AAW) const {
  SlotTracker SlotTable(this);
  formatted_raw_ostream OS(ROS);
  AssemblyWriter W(OS, SlotTable, this, AAW);
  W.printModule(this);
}

void NamedMDNode::print(raw_ostream &ROS, AssemblyAnnotationWriter *AAW) const {
  SlotTracker SlotTable(getParent());
  formatted_raw_ostream OS(ROS);
  AssemblyWriter W(OS, SlotTable, getParent(), AAW);
  W.printNamedMDNode(this);
}

void Type::print(raw_ostream &OS) const {
  if (this == 0) {
    OS << "<null Type>";
    return;
  }
  TypePrinting().print(this, OS);
}

void Value::print(raw_ostream &ROS, AssemblyAnnotationWriter *AAW) const {
  if (this == 0) {
    ROS << "printing a <null> value\n";
    return;
  }
  formatted_raw_ostream OS(ROS);
  if (const Instruction *I = dyn_cast<Instruction>(this)) {
    const Function *F = I->getParent() ? I->getParent()->getParent() : 0;
    SlotTracker SlotTable(F);
    AssemblyWriter W(OS, SlotTable, getModuleFromVal(I), AAW);
    W.printInstruction(*I);
  } else if (const BasicBlock *BB = dyn_cast<BasicBlock>(this)) {
    SlotTracker SlotTable(BB->getParent());
    AssemblyWriter W(OS, SlotTable, getModuleFromVal(BB), AAW);
    W.printBasicBlock(BB);
  } else if (const GlobalValue *GV = dyn_cast<GlobalValue>(this)) {
    SlotTracker SlotTable(GV->getParent());
    AssemblyWriter W(OS, SlotTable, GV->getParent(), AAW);
    if (const GlobalVariable *V = dyn_cast<GlobalVariable>(GV))
      W.printGlobal(V);
    else if (const Function *F = dyn_cast<Function>(GV))
      W.printFunction(F);
    else
      W.printAlias(cast<GlobalAlias>(GV));
  } else if (const MDNode *N = dyn_cast<MDNode>(this)) {
    const Function *F = N->getFunction();
    SlotTracker SlotTable(F);
    AssemblyWriter W(OS, SlotTable, F ? F->getParent() : 0, AAW);
    W.printMDNodeBody(N);
  } else if (const Constant *C = dyn_cast<Constant>(this)) {
    TypePrinting TypePrinter;
    TypePrinter.print(C->getType(), OS);
    OS << ' ';
    WriteConstantInternal(OS, C, TypePrinter, 0, 0);
  } else if (isa<InlineAsm>(this) || isa<MDString>(this) ||
             isa<Argument>(this)) {
    WriteAsOperand(OS, this, true, 0);
  } else {
    // Otherwise we don't know what it is. Call the virtual function to
    // allow a subclass to print itself.
    printCustom(OS);
  }
}

// Value::printCustom - subclasses should override this to implement printing.
void Value::printCustom(raw_ostream &OS) const {
  llvm_unreachable("Unknown value to print out!");
}

// Value::dump - allow easy printing of Values from the debugger.
void Value::dump() const { print(dbgs()); dbgs() << '\n'; }

// Type::dump - allow easy printing of Types from the debugger.
// This one uses type names from the given context module
void Type::dump(const Module *Context) const {
  WriteTypeSymbolic(dbgs(), this, Context);
  dbgs() << '\n';
}

// Type::dump - allow easy printing of Types from the debugger.
void Type::dump() const { dump(0); }

// Module::dump() - Allow printing of Modules from the debugger.
void Module::dump() const { print(dbgs(), 0); }
