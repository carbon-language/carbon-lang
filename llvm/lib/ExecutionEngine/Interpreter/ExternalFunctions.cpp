//===-- ExternalMethods.cpp - Implement External Methods ------------------===//
// 
//  This file contains both code to deal with invoking "external" methods, but
//  also contains code that implements "exported" external methods. 
//
//  External methods in LLI are implemented by dlopen'ing the lli executable and
//  using dlsym to look op the methods that we want to invoke.  If a method is
//  found, then the arguments are mangled and passed in to the function call.
//
//===----------------------------------------------------------------------===//

#include "Interpreter.h"
#include "llvm/DerivedTypes.h"
#include <map>
#include <dlfcn.h>
#include <link.h>
#include <math.h>

typedef GenericValue (*ExFunc)(MethodType *, const vector<GenericValue> &);
static map<const Method *, ExFunc> Functions;
static map<string, ExFunc> FuncNames;

static Interpreter *TheInterpreter;

// getCurrentExecutablePath() - Return the directory that the lli executable
// lives in.
//
string Interpreter::getCurrentExecutablePath() const {
  Dl_info Info;
  if (dladdr(&TheInterpreter, &Info) == 0) return "";
  
  string LinkAddr(Info.dli_fname);
  unsigned SlashPos = LinkAddr.rfind('/');
  if (SlashPos != string::npos)
    LinkAddr.resize(SlashPos);    // Trim the executable name off...

  return LinkAddr;
}


static char getTypeID(const Type *Ty) {
  switch (Ty->getPrimitiveID()) {
  case Type::VoidTyID:    return 'V';
  case Type::BoolTyID:    return 'o';
  case Type::UByteTyID:   return 'B';
  case Type::SByteTyID:   return 'b';
  case Type::UShortTyID:  return 'S';
  case Type::ShortTyID:   return 's';
  case Type::UIntTyID:    return 'I';
  case Type::IntTyID:     return 'i';
  case Type::ULongTyID:   return 'L';
  case Type::LongTyID:    return 'l';
  case Type::FloatTyID:   return 'F';
  case Type::DoubleTyID:  return 'D';
  case Type::PointerTyID: return 'P';
  case Type::MethodTyID:  return 'M';
  case Type::StructTyID:  return 'T';
  case Type::ArrayTyID:   return 'A';
  case Type::OpaqueTyID:  return 'O';
  default: return 'U';
  }
}

static ExFunc lookupMethod(const Method *M) {
  // Function not found, look it up... start by figuring out what the
  // composite function name should be.
  string ExtName = "lle_";
  const MethodType *MT = M->getMethodType();
  for (unsigned i = 0; const Type *Ty = MT->getContainedType(i); ++i)
    ExtName += getTypeID(Ty);
  ExtName += "_" + M->getName();

  //cout << "Tried: '" << ExtName << "'\n";
  ExFunc FnPtr = FuncNames[ExtName];
  if (FnPtr == 0)
    FnPtr = (ExFunc)dlsym(RTLD_DEFAULT, ExtName.c_str());
  if (FnPtr == 0)
    FnPtr = FuncNames["lle_X_"+M->getName()];
  if (FnPtr == 0)  // Try calling a generic function... if it exists...
    FnPtr = (ExFunc)dlsym(RTLD_DEFAULT, ("lle_X_"+M->getName()).c_str());
  if (FnPtr != 0)
    Functions.insert(make_pair(M, FnPtr));  // Cache for later
  return FnPtr;
}

GenericValue Interpreter::callExternalMethod(Method *M,
                                         const vector<GenericValue> &ArgVals) {
  TheInterpreter = this;

  // Do a lookup to see if the method is in our cache... this should just be a
  // defered annotation!
  map<const Method *, ExFunc>::iterator FI = Functions.find(M);
  ExFunc Fn = (FI == Functions.end()) ? lookupMethod(M) : FI->second;
  if (Fn == 0) {
    cout << "Tried to execute an unknown external method: "
	 << M->getType()->getDescription() << " " << M->getName() << endl;
    return GenericValue();
  }

  // TODO: FIXME when types are not const!
  GenericValue Result = Fn(const_cast<MethodType*>(M->getMethodType()),ArgVals);
  return Result;
}


//===----------------------------------------------------------------------===//
//  Methods "exported" to the running application...
//
extern "C" {  // Don't add C++ manglings to llvm mangling :)

// Implement void printstr([ubyte {x N}] *)
GenericValue lle_VP_printstr(MethodType *M, const vector<GenericValue> &ArgVal){
  assert(ArgVal.size() == 1 && "printstr only takes one argument!");
  cout << (char*)ArgVal[0].PointerVal;
  return GenericValue();
}

// Implement 'void print(X)' for every type...
GenericValue lle_X_print(MethodType *M, const vector<GenericValue> &ArgVals) {
  assert(ArgVals.size() == 1 && "generic print only takes one argument!");

  Interpreter::print(M->getParamTypes()[0], ArgVals[0]);
  return GenericValue();
}

// Implement 'void printVal(X)' for every type...
GenericValue lle_X_printVal(MethodType *M, const vector<GenericValue> &ArgVal) {
  assert(ArgVal.size() == 1 && "generic print only takes one argument!");

  // Specialize print([ubyte {x N} ] *) and print(sbyte *)
  if (PointerType *PTy = dyn_cast<PointerType>(M->getParamTypes()[0].get()))
    if (PTy->getValueType() == Type::SByteTy ||
        isa<ArrayType>(PTy->getValueType())) {
      return lle_VP_printstr(M, ArgVal);
    }

  Interpreter::printValue(M->getParamTypes()[0], ArgVal[0]);
  return GenericValue();
}

// Implement 'void printString(X)'
// Argument must be [ubyte {x N} ] * or sbyte *
GenericValue lle_X_printString(MethodType *M, const vector<GenericValue> &ArgVal) {
  assert(ArgVal.size() == 1 && "generic print only takes one argument!");
  return lle_VP_printstr(M, ArgVal);
}

// Implement 'void print<TYPE>(X)' for each primitive type or pointer type
#define PRINT_TYPE_FUNC(TYPENAME,TYPEID) \
  GenericValue lle_X_print##TYPENAME(MethodType *M,\
                                     const vector<GenericValue> &ArgVal) {\
    assert(ArgVal.size() == 1 && "generic print only takes one argument!");\
    assert(M->getParamTypes()[0].get()->getPrimitiveID() == Type::##TYPEID);\
    Interpreter::printValue(M->getParamTypes()[0], ArgVal[0]);\
    return GenericValue();\
  }

PRINT_TYPE_FUNC(SByte,   SByteTyID)
PRINT_TYPE_FUNC(UByte,   UByteTyID)
PRINT_TYPE_FUNC(Short,   ShortTyID)
PRINT_TYPE_FUNC(UShort,  UShortTyID)
PRINT_TYPE_FUNC(Int,     IntTyID)
PRINT_TYPE_FUNC(UInt,    UIntTyID)
PRINT_TYPE_FUNC(Long,    LongTyID)
PRINT_TYPE_FUNC(ULong,   ULongTyID)
PRINT_TYPE_FUNC(Float,   FloatTyID)
PRINT_TYPE_FUNC(Double,  DoubleTyID)
PRINT_TYPE_FUNC(Pointer, PointerTyID)


// void "putchar"(sbyte)
GenericValue lle_Vb_putchar(MethodType *M, const vector<GenericValue> &Args) {
  cout << Args[0].SByteVal;
  return GenericValue();
}

// int "putchar"(int)
GenericValue lle_ii_putchar(MethodType *M, const vector<GenericValue> &Args) {
  cout << ((char)Args[0].IntVal) << flush;
  return Args[0];
}

// void "putchar"(ubyte)
GenericValue lle_VB_putchar(MethodType *M, const vector<GenericValue> &Args) {
  cout << Args[0].SByteVal << flush;
  return Args[0];
}

// void "__main"()
GenericValue lle_V___main(MethodType *M, const vector<GenericValue> &Args) {
  return GenericValue();
}

// void "exit"(int)
GenericValue lle_Vi_exit(MethodType *M, const vector<GenericValue> &Args) {
  TheInterpreter->exitCalled(Args[0]);
  return GenericValue();
}

// void *malloc(uint)
GenericValue lle_PI_malloc(MethodType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 1 && "Malloc expects one argument!");
  GenericValue GV;
  GV.PointerVal = (uint64_t)malloc(Args[0].UIntVal);
  return GV;
}

// void free(void *)
GenericValue lle_VP_free(MethodType *M, const vector<GenericValue> &Args) {
  free((void*)Args[0].PointerVal);
  return GenericValue();
}

// double pow(double, double)
GenericValue lle_DDD_pow(MethodType *M, const vector<GenericValue> &Args) {
  GenericValue GV;
  GV.DoubleVal = pow(Args[0].DoubleVal, Args[1].DoubleVal);
  return GV;
}


// int printf(sbyte *, ...) - a very rough implementation to make output useful.
GenericValue lle_iP_printf(MethodType *M, const vector<GenericValue> &Args) {
  const char *FmtStr = (const char *)Args[0].PointerVal;
  unsigned ArgNo = 1;

  // printf should return # chars printed.  This is completely incorrect, but
  // close enough for now.
  GenericValue GV; GV.IntVal = strlen(FmtStr);
  while (1) {
    switch (*FmtStr) {
    case 0: return GV;             // Null terminator...
    default:                       // Normal nonspecial character
      cout << *FmtStr++;
      break;
    case '\\': {                   // Handle escape codes
      char Buffer[3];
      Buffer[0] = *FmtStr++;
      Buffer[1] = *FmtStr++;
      Buffer[2] = 0;
      cout << Buffer;
      break;
    }
    case '%': {                    // Handle format specifiers
      bool isLong = false;
      ++FmtStr;
      if (*FmtStr == 'l') {
        isLong = true;
        FmtStr++;
      }

      switch (*FmtStr) {
      case '%': cout << *FmtStr; break;                             // %%
      case 'd':                                                     // %d %i
      case 'i': cout << Args[ArgNo++].IntVal; break;
      case 'u': cout << Args[ArgNo++].UIntVal; break;               // %u
      case 'o': cout << oct << Args[ArgNo++].UIntVal << dec; break; // %o
      case 'x':
      case 'X': cout << hex << Args[ArgNo++].UIntVal << dec; break; // %x %X
      case 'e': case 'E': case 'g': case 'G':                       // %[eEgG]
        cout /*<< std::scientific*/ << Args[ArgNo++].DoubleVal
             /*<< std::fixed*/; break;
      case 'f': cout << Args[ArgNo++].DoubleVal; break;             // %f
      case 'c': cout << Args[ArgNo++].UByteVal; break;              // %c
      case 's': cout << (char*)Args[ArgNo++].PointerVal; break;     // %s
      default:  cout << "<unknown printf code '" << *FmtStr << "'!>";
                ArgNo++; break;
      }
      ++FmtStr;
      break;
    }
    }
  }
}

} // End extern "C"


void Interpreter::initializeExternalMethods() {
  FuncNames["lle_VP_printstr"] = lle_VP_printstr;
  FuncNames["lle_X_print"] = lle_X_print;
  FuncNames["lle_X_printVal"] = lle_X_printVal;
  FuncNames["lle_X_printString"] = lle_X_printString;
  FuncNames["lle_X_printUByte"] = lle_X_printUByte;
  FuncNames["lle_X_printSByte"] = lle_X_printSByte;
  FuncNames["lle_X_printUShort"] = lle_X_printUShort;
  FuncNames["lle_X_printShort"] = lle_X_printShort;
  FuncNames["lle_X_printInt"] = lle_X_printInt;
  FuncNames["lle_X_printUInt"] = lle_X_printUInt;
  FuncNames["lle_X_printLong"] = lle_X_printLong;
  FuncNames["lle_X_printULong"] = lle_X_printULong;
  FuncNames["lle_X_printFloat"] = lle_X_printFloat;
  FuncNames["lle_X_printDouble"] = lle_X_printDouble;
  FuncNames["lle_X_printPointer"] = lle_X_printPointer;
  FuncNames["lle_Vb_putchar"] = lle_Vb_putchar;
  FuncNames["lle_ii_putchar"] = lle_ii_putchar;
  FuncNames["lle_VB_putchar"] = lle_VB_putchar;
  FuncNames["lle_V___main"] = lle_V___main;
  FuncNames["lle_Vi_exit"] = lle_Vi_exit;
  FuncNames["lle_PI_malloc"] = lle_PI_malloc;
  FuncNames["lle_VP_free"] = lle_VP_free;
  FuncNames["lle_DDD_pow"] = lle_DDD_pow;
  FuncNames["lle_iP_printf"] = lle_iP_printf;
}
