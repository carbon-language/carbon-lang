//===-- ExternalFunctions.cpp - Implement External Functions --------------===//
// 
//  This file contains both code to deal with invoking "external" functions, but
//  also contains code that implements "exported" external functions.
//
//  External functions in LLI are implemented by dlopen'ing the lli executable
//  and using dlsym to look op the functions that we want to invoke.  If a
//  function is found, then the arguments are mangled and passed in to the
//  function call.
//
//===----------------------------------------------------------------------===//

#include "Interpreter.h"
#include "ExecutionAnnotations.h"
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/SymbolTable.h"
#include "llvm/Target/TargetData.h"
#include <map>
#include <dlfcn.h>
#include <link.h>
#include <math.h>
#include <stdio.h>
using std::vector;
using std::cout;

typedef GenericValue (*ExFunc)(FunctionType *, const vector<GenericValue> &);
static std::map<const Function *, ExFunc> Functions;
static std::map<std::string, ExFunc> FuncNames;

static Interpreter *TheInterpreter;

// getCurrentExecutablePath() - Return the directory that the lli executable
// lives in.
//
std::string Interpreter::getCurrentExecutablePath() const {
  Dl_info Info;
  if (dladdr(&TheInterpreter, &Info) == 0) return "";
  
  std::string LinkAddr(Info.dli_fname);
  unsigned SlashPos = LinkAddr.rfind('/');
  if (SlashPos != std::string::npos)
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
  case Type::FunctionTyID:  return 'M';
  case Type::StructTyID:  return 'T';
  case Type::ArrayTyID:   return 'A';
  case Type::OpaqueTyID:  return 'O';
  default: return 'U';
  }
}

static ExFunc lookupFunction(const Function *M) {
  // Function not found, look it up... start by figuring out what the
  // composite function name should be.
  std::string ExtName = "lle_";
  const FunctionType *MT = M->getFunctionType();
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
    Functions.insert(std::make_pair(M, FnPtr));  // Cache for later
  return FnPtr;
}

GenericValue Interpreter::callExternalMethod(Function *M,
                                         const vector<GenericValue> &ArgVals) {
  TheInterpreter = this;

  // Do a lookup to see if the function is in our cache... this should just be a
  // defered annotation!
  std::map<const Function *, ExFunc>::iterator FI = Functions.find(M);
  ExFunc Fn = (FI == Functions.end()) ? lookupFunction(M) : FI->second;
  if (Fn == 0) {
    cout << "Tried to execute an unknown external function: "
	 << M->getType()->getDescription() << " " << M->getName() << "\n";
    return GenericValue();
  }

  // TODO: FIXME when types are not const!
  GenericValue Result = Fn(const_cast<FunctionType*>(M->getFunctionType()),
                           ArgVals);
  return Result;
}


//===----------------------------------------------------------------------===//
//  Functions "exported" to the running application...
//
extern "C" {  // Don't add C++ manglings to llvm mangling :)

// Implement void printstr([ubyte {x N}] *)
GenericValue lle_VP_printstr(FunctionType *M,
			     const vector<GenericValue> &ArgVal){
  assert(ArgVal.size() == 1 && "printstr only takes one argument!");
  cout << (char*)GVTOP(ArgVal[0]);
  return GenericValue();
}

// Implement 'void print(X)' for every type...
GenericValue lle_X_print(FunctionType *M, const vector<GenericValue> &ArgVals) {
  assert(ArgVals.size() == 1 && "generic print only takes one argument!");

  Interpreter::print(M->getParamTypes()[0], ArgVals[0]);
  return GenericValue();
}

// Implement 'void printVal(X)' for every type...
GenericValue lle_X_printVal(FunctionType *M,
			    const vector<GenericValue> &ArgVal) {
  assert(ArgVal.size() == 1 && "generic print only takes one argument!");

  // Specialize print([ubyte {x N} ] *) and print(sbyte *)
  if (const PointerType *PTy = 
      dyn_cast<PointerType>(M->getParamTypes()[0].get()))
    if (PTy->getElementType() == Type::SByteTy ||
        isa<ArrayType>(PTy->getElementType())) {
      return lle_VP_printstr(M, ArgVal);
    }

  Interpreter::printValue(M->getParamTypes()[0], ArgVal[0]);
  return GenericValue();
}

// Implement 'void printString(X)'
// Argument must be [ubyte {x N} ] * or sbyte *
GenericValue lle_X_printString(FunctionType *M,
			       const vector<GenericValue> &ArgVal) {
  assert(ArgVal.size() == 1 && "generic print only takes one argument!");
  return lle_VP_printstr(M, ArgVal);
}

// Implement 'void print<TYPE>(X)' for each primitive type or pointer type
#define PRINT_TYPE_FUNC(TYPENAME,TYPEID) \
  GenericValue lle_X_print##TYPENAME(FunctionType *M,\
                                     const vector<GenericValue> &ArgVal) {\
    assert(ArgVal.size() == 1 && "generic print only takes one argument!");\
    assert(M->getParamTypes()[0].get()->getPrimitiveID() == Type::TYPEID);\
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


// void putchar(sbyte)
GenericValue lle_Vb_putchar(FunctionType *M, const vector<GenericValue> &Args) {
  cout << Args[0].SByteVal;
  return GenericValue();
}

// int putchar(int)
GenericValue lle_ii_putchar(FunctionType *M, const vector<GenericValue> &Args) {
  cout << ((char)Args[0].IntVal) << std::flush;
  return Args[0];
}

// void putchar(ubyte)
GenericValue lle_VB_putchar(FunctionType *M, const vector<GenericValue> &Args) {
  cout << Args[0].SByteVal << std::flush;
  return Args[0];
}

// void __main()
GenericValue lle_V___main(FunctionType *M, const vector<GenericValue> &Args) {
  return GenericValue();
}

// void exit(int)
GenericValue lle_X_exit(FunctionType *M, const vector<GenericValue> &Args) {
  TheInterpreter->exitCalled(Args[0]);
  return GenericValue();
}

// void abort(void)
GenericValue lle_X_abort(FunctionType *M, const vector<GenericValue> &Args) {
  std::cerr << "***PROGRAM ABORTED***!\n";
  GenericValue GV;
  GV.IntVal = 1;
  TheInterpreter->exitCalled(GV);
  return GenericValue();
}

// void *malloc(uint)
GenericValue lle_X_malloc(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 1 && "Malloc expects one argument!");
  return PTOGV(malloc(Args[0].UIntVal));
}

// void *calloc(uint, uint)
GenericValue lle_X_calloc(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 2 && "calloc expects two arguments!");
  return PTOGV(calloc(Args[0].UIntVal, Args[1].UIntVal));
}

// void free(void *)
GenericValue lle_X_free(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  free(GVTOP(Args[0]));
  return GenericValue();
}

// int atoi(char *)
GenericValue lle_X_atoi(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;
  GV.IntVal = atoi((char*)GVTOP(Args[0]));
  return GV;
}

// double pow(double, double)
GenericValue lle_X_pow(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 2);
  GenericValue GV;
  GV.DoubleVal = pow(Args[0].DoubleVal, Args[1].DoubleVal);
  return GV;
}

// double exp(double)
GenericValue lle_X_exp(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;
  GV.DoubleVal = exp(Args[0].DoubleVal);
  return GV;
}

// double sqrt(double)
GenericValue lle_X_sqrt(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;
  GV.DoubleVal = sqrt(Args[0].DoubleVal);
  return GV;
}

// double log(double)
GenericValue lle_X_log(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;
  GV.DoubleVal = log(Args[0].DoubleVal);
  return GV;
}

// int isnan(double value);
GenericValue lle_X_isnan(FunctionType *F, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;
  GV.IntVal = isnan(Args[0].DoubleVal);
  return GV;
}

// double floor(double)
GenericValue lle_X_floor(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;
  GV.DoubleVal = floor(Args[0].DoubleVal);
  return GV;
}

// double drand48()
GenericValue lle_X_drand48(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 0);
  GenericValue GV;
  GV.DoubleVal = drand48();
  return GV;
}

// long lrand48()
GenericValue lle_X_lrand48(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 0);
  GenericValue GV;
  GV.IntVal = lrand48();
  return GV;
}

// void srand48(long)
GenericValue lle_X_srand48(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  srand48(Args[0].IntVal);
  return GenericValue();
}

// void srand(uint)
GenericValue lle_X_srand(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  srand(Args[0].UIntVal);
  return GenericValue();
}

// int puts(const char*)
GenericValue lle_X_puts(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;
  GV.IntVal = puts((char*)GVTOP(Args[0]));
  return GV;
}

// int sprintf(sbyte *, sbyte *, ...) - a very rough implementation to make
// output useful.
GenericValue lle_X_sprintf(FunctionType *M, const vector<GenericValue> &Args) {
  char *OutputBuffer = (char *)GVTOP(Args[0]);
  const char *FmtStr = (const char *)GVTOP(Args[1]);
  unsigned ArgNo = 2;

  // printf should return # chars printed.  This is completely incorrect, but
  // close enough for now.
  GenericValue GV; GV.IntVal = strlen(FmtStr);
  while (1) {
    switch (*FmtStr) {
    case 0: return GV;             // Null terminator...
    default:                       // Normal nonspecial character
      sprintf(OutputBuffer++, "%c", *FmtStr++);
      break;
    case '\\': {                   // Handle escape codes
      sprintf(OutputBuffer, "%c%c", *FmtStr, *(FmtStr+1));
      FmtStr += 2; OutputBuffer += 2;
      break;
    }
    case '%': {                    // Handle format specifiers
      char FmtBuf[100] = "", Buffer[1000] = "";
      char *FB = FmtBuf;
      *FB++ = *FmtStr++;
      char Last = *FB++ = *FmtStr++;
      unsigned HowLong = 0;
      while (Last != 'c' && Last != 'd' && Last != 'i' && Last != 'u' &&
             Last != 'o' && Last != 'x' && Last != 'X' && Last != 'e' &&
             Last != 'E' && Last != 'g' && Last != 'G' && Last != 'f' &&
             Last != 'p' && Last != 's' && Last != '%') {
        if (Last == 'l' || Last == 'L') HowLong++;  // Keep track of l's
        Last = *FB++ = *FmtStr++;
      }
      *FB = 0;
      
      switch (Last) {
      case '%':
        sprintf(Buffer, FmtBuf); break;
      case 'c':
        sprintf(Buffer, FmtBuf, Args[ArgNo++].IntVal); break;
      case 'd': case 'i':
      case 'u': case 'o':
      case 'x': case 'X':
        if (HowLong >= 1) {
          if (HowLong == 1) {
            // Make sure we use %lld with a 64 bit argument because we might be
            // compiling LLI on a 32 bit compiler.
            unsigned Size = strlen(FmtBuf);
            FmtBuf[Size] = FmtBuf[Size-1];
            FmtBuf[Size+1] = 0;
            FmtBuf[Size-1] = 'l';
          }
          sprintf(Buffer, FmtBuf, Args[ArgNo++].ULongVal);
        } else
          sprintf(Buffer, FmtBuf, Args[ArgNo++].IntVal); break;
      case 'e': case 'E': case 'g': case 'G': case 'f':
        sprintf(Buffer, FmtBuf, Args[ArgNo++].DoubleVal); break;
      case 'p':
        sprintf(Buffer, FmtBuf, (void*)GVTOP(Args[ArgNo++])); break;
      case 's': 
        sprintf(Buffer, FmtBuf, (char*)GVTOP(Args[ArgNo++])); break;
      default:  cout << "<unknown printf code '" << *FmtStr << "'!>";
        ArgNo++; break;
      }
      strcpy(OutputBuffer, Buffer);
      OutputBuffer += strlen(Buffer);
      }
      break;
    }
  }
}

// int printf(sbyte *, ...) - a very rough implementation to make output useful.
GenericValue lle_X_printf(FunctionType *M, const vector<GenericValue> &Args) {
  char Buffer[10000];
  vector<GenericValue> NewArgs;
  NewArgs.push_back(PTOGV(Buffer));
  NewArgs.insert(NewArgs.end(), Args.begin(), Args.end());
  GenericValue GV = lle_X_sprintf(M, NewArgs);
  cout << Buffer;
  return GV;
}

static void ByteswapSCANFResults(const char *Fmt, void *Arg0, void *Arg1,
                                 void *Arg2, void *Arg3, void *Arg4, void *Arg5,
                                 void *Arg6, void *Arg7, void *Arg8) {
  void *Args[] = { Arg0, Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7, Arg8, 0 };

  // Loop over the format string, munging read values as appropriate (performs
  // byteswaps as neccesary).
  unsigned ArgNo = 0;
  while (*Fmt) {
    if (*Fmt++ == '%') {
      // Read any flag characters that may be present...
      bool Suppress = false;
      bool Half = false;
      bool Long = false;
      bool LongLong = false;  // long long or long double

      while (1) {
        switch (*Fmt++) {
        case '*': Suppress = true; break;
        case 'a': /*Allocate = true;*/ break;  // We don't need to track this
        case 'h': Half = true; break;
        case 'l': Long = true; break;
        case 'q':
        case 'L': LongLong = true; break;
        default:
          if (Fmt[-1] > '9' || Fmt[-1] < '0')   // Ignore field width specs
            goto Out;
        }
      }
    Out:

      // Read the conversion character
      if (!Suppress && Fmt[-1] != '%') { // Nothing to do?
        unsigned Size = 0;
        const Type *Ty = 0;

        switch (Fmt[-1]) {
        case 'i': case 'o': case 'u': case 'x': case 'X': case 'n': case 'p':
        case 'd':
          if (Long || LongLong) {
            Size = 8; Ty = Type::ULongTy;
          } else if (Half) {
            Size = 4; Ty = Type::UShortTy;
          } else {
            Size = 4; Ty = Type::UIntTy;
          }
          break;

        case 'e': case 'g': case 'E':
        case 'f':
          if (Long || LongLong) {
            Size = 8; Ty = Type::DoubleTy;
          } else {
            Size = 4; Ty = Type::FloatTy;
          }
          break;

        case 's': case 'c': case '[':  // No byteswap needed
          Size = 1;
          Ty = Type::SByteTy;
          break;

        default: break;
        }

        if (Size) {
          GenericValue GV;
          void *Arg = Args[ArgNo++];
          memcpy(&GV, Arg, Size);
          TheInterpreter->StoreValueToMemory(GV, (GenericValue*)Arg, Ty);
        }
      }
    }
  }
}

// int sscanf(const char *format, ...);
GenericValue lle_X_sscanf(FunctionType *M, const vector<GenericValue> &args) {
  assert(args.size() < 10 && "Only handle up to 10 args to sscanf right now!");

  char *Args[10];
  for (unsigned i = 0; i < args.size(); ++i)
    Args[i] = (char*)GVTOP(args[i]);

  GenericValue GV;
  GV.IntVal = sscanf(Args[0], Args[1], Args[2], Args[3], Args[4],
                     Args[5], Args[6], Args[7], Args[8], Args[9]);
  ByteswapSCANFResults(Args[1], Args[2], Args[3], Args[4],
                       Args[5], Args[6], Args[7], Args[8], Args[9], 0);
  return GV;
}

// int scanf(const char *format, ...);
GenericValue lle_X_scanf(FunctionType *M, const vector<GenericValue> &args) {
  assert(args.size() < 10 && "Only handle up to 10 args to scanf right now!");

  char *Args[10];
  for (unsigned i = 0; i < args.size(); ++i)
    Args[i] = (char*)GVTOP(args[i]);

  GenericValue GV;
  GV.IntVal = scanf(Args[0], Args[1], Args[2], Args[3], Args[4],
                    Args[5], Args[6], Args[7], Args[8], Args[9]);
  ByteswapSCANFResults(Args[0], Args[1], Args[2], Args[3], Args[4],
                       Args[5], Args[6], Args[7], Args[8], Args[9]);
  return GV;
}


// int clock(void) - Profiling implementation
GenericValue lle_i_clock(FunctionType *M, const vector<GenericValue> &Args) {
  extern int clock(void);
  GenericValue GV; GV.IntVal = clock();
  return GV;
}


//===----------------------------------------------------------------------===//
// String Functions...
//===----------------------------------------------------------------------===//

// int strcmp(const char *S1, const char *S2);
GenericValue lle_X_strcmp(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 2);
  GenericValue Ret;
  Ret.IntVal = strcmp((char*)GVTOP(Args[0]), (char*)GVTOP(Args[1]));
  return Ret;
}

// char *strcat(char *Dest, const char *src);
GenericValue lle_X_strcat(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 2);
  return PTOGV(strcat((char*)GVTOP(Args[0]), (char*)GVTOP(Args[1])));
}

// char *strcpy(char *Dest, const char *src);
GenericValue lle_X_strcpy(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 2);
  return PTOGV(strcpy((char*)GVTOP(Args[0]), (char*)GVTOP(Args[1])));
}

// long strlen(const char *src);
GenericValue lle_X_strlen(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue Ret;
  Ret.LongVal = strlen((char*)GVTOP(Args[0]));
  return Ret;
}

// void *memset(void *S, int C, size_t N)
GenericValue lle_X_memset(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 3);
  return PTOGV(memset(GVTOP(Args[0]), Args[1].IntVal, Args[2].UIntVal));
}

// void *memcpy(void *Dest, void *src, size_t Size);
GenericValue lle_X_memcpy(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 3);
  return PTOGV(memcpy((char*)GVTOP(Args[0]), (char*)GVTOP(Args[1]),
                      Args[2].UIntVal));
}

//===----------------------------------------------------------------------===//
// IO Functions...
//===----------------------------------------------------------------------===//

// getFILE - Turn a pointer in the host address space into a legit pointer in
// the interpreter address space.  For the most part, this is an identity
// transformation, but if the program refers to stdio, stderr, stdin then they
// have pointers that are relative to the __iob array.  If this is the case,
// change the FILE into the REAL stdio stream.
// 
static FILE *getFILE(void *Ptr) {
  static Module *LastMod = 0;
  static PointerTy IOBBase = 0;
  static unsigned FILESize;

  if (LastMod != &TheInterpreter->getModule()) { // Module change or initialize?
    Module *M = LastMod = &TheInterpreter->getModule();

    // Check to see if the currently loaded module contains an __iob symbol...
    GlobalVariable *IOB = 0;
    SymbolTable &ST = M->getSymbolTable();
    for (SymbolTable::iterator I = ST.begin(), E = ST.end(); I != E; ++I) {
      SymbolTable::VarMap &M = I->second;
      for (SymbolTable::VarMap::iterator J = M.begin(), E = M.end();
           J != E; ++J)
        if (J->first == "__iob")
          if ((IOB = dyn_cast<GlobalVariable>(J->second)))
            break;
      if (IOB) break;
    }

#if 0   /// FIXME!  __iob support for LLI
    // If we found an __iob symbol now, find out what the actual address it's
    // held in is...
    if (IOB) {
      // Get the address the array lives in...
      GlobalAddress *Address = 
        (GlobalAddress*)IOB->getOrCreateAnnotation(GlobalAddressAID);
      IOBBase = (PointerTy)(GenericValue*)Address->Ptr;

      // Figure out how big each element of the array is...
      const ArrayType *AT =
        dyn_cast<ArrayType>(IOB->getType()->getElementType());
      if (AT)
        FILESize = TD.getTypeSize(AT->getElementType());
      else
        FILESize = 16*8;  // Default size
    }
#endif
  }

  // Check to see if this is a reference to __iob...
  if (IOBBase) {
    unsigned FDNum = ((unsigned long)Ptr-IOBBase)/FILESize;
    if (FDNum == 0)
      return stdin;
    else if (FDNum == 1)
      return stdout;
    else if (FDNum == 2)
      return stderr;
  }

  return (FILE*)Ptr;
}


// FILE *fopen(const char *filename, const char *mode);
GenericValue lle_X_fopen(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 2);
  return PTOGV(fopen((const char *)GVTOP(Args[0]),
		     (const char *)GVTOP(Args[1])));
}

// int fclose(FILE *F);
GenericValue lle_X_fclose(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;
  GV.IntVal = fclose(getFILE(GVTOP(Args[0])));
  return GV;
}

// int feof(FILE *stream);
GenericValue lle_X_feof(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;

  GV.IntVal = feof(getFILE(GVTOP(Args[0])));
  return GV;
}

// size_t fread(void *ptr, size_t size, size_t nitems, FILE *stream);
GenericValue lle_X_fread(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 4);
  GenericValue GV;

  GV.UIntVal = fread((void*)GVTOP(Args[0]), Args[1].UIntVal,
                     Args[2].UIntVal, getFILE(GVTOP(Args[3])));
  return GV;
}

// size_t fwrite(const void *ptr, size_t size, size_t nitems, FILE *stream);
GenericValue lle_X_fwrite(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 4);
  GenericValue GV;

  GV.UIntVal = fwrite((void*)GVTOP(Args[0]), Args[1].UIntVal,
                      Args[2].UIntVal, getFILE(GVTOP(Args[3])));
  return GV;
}

// char *fgets(char *s, int n, FILE *stream);
GenericValue lle_X_fgets(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 3);
  return GVTOP(fgets((char*)GVTOP(Args[0]), Args[1].IntVal,
		     getFILE(GVTOP(Args[2]))));
}

// FILE *freopen(const char *path, const char *mode, FILE *stream);
GenericValue lle_X_freopen(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 3);
  return PTOGV(freopen((char*)GVTOP(Args[0]), (char*)GVTOP(Args[1]),
		       getFILE(GVTOP(Args[2]))));
}

// int fflush(FILE *stream);
GenericValue lle_X_fflush(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;
  GV.IntVal = fflush(getFILE(GVTOP(Args[0])));
  return GV;
}

// int getc(FILE *stream);
GenericValue lle_X_getc(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;
  GV.IntVal = getc(getFILE(GVTOP(Args[0])));
  return GV;
}

// int _IO_getc(FILE *stream);
GenericValue lle_X__IO_getc(FunctionType *F, const vector<GenericValue> &Args) {
  return lle_X_getc(F, Args);
}

// int fputc(int C, FILE *stream);
GenericValue lle_X_fputc(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 2);
  GenericValue GV;
  GV.IntVal = fputc(Args[0].IntVal, getFILE(GVTOP(Args[1])));
  return GV;
}

// int ungetc(int C, FILE *stream);
GenericValue lle_X_ungetc(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 2);
  GenericValue GV;
  GV.IntVal = ungetc(Args[0].IntVal, getFILE(GVTOP(Args[1])));
  return GV;
}

// int fprintf(FILE *,sbyte *, ...) - a very rough implementation to make output
// useful.
GenericValue lle_X_fprintf(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() >= 2);
  char Buffer[10000];
  vector<GenericValue> NewArgs;
  NewArgs.push_back(PTOGV(Buffer));
  NewArgs.insert(NewArgs.end(), Args.begin()+1, Args.end());
  GenericValue GV = lle_X_sprintf(M, NewArgs);

  fputs(Buffer, getFILE(GVTOP(Args[0])));
  return GV;
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
  FuncNames["lle_Vb_putchar"]     = lle_Vb_putchar;
  FuncNames["lle_ii_putchar"]     = lle_ii_putchar;
  FuncNames["lle_VB_putchar"]     = lle_VB_putchar;
  FuncNames["lle_V___main"]       = lle_V___main;
  FuncNames["lle_X_exit"]         = lle_X_exit;
  FuncNames["lle_X_abort"]        = lle_X_abort;
  FuncNames["lle_X_malloc"]       = lle_X_malloc;
  FuncNames["lle_X_calloc"]       = lle_X_calloc;
  FuncNames["lle_X_free"]         = lle_X_free;
  FuncNames["lle_X_atoi"]         = lle_X_atoi;
  FuncNames["lle_X_pow"]          = lle_X_pow;
  FuncNames["lle_X_exp"]          = lle_X_exp;
  FuncNames["lle_X_log"]          = lle_X_log;
  FuncNames["lle_X_isnan"]        = lle_X_isnan;
  FuncNames["lle_X_floor"]        = lle_X_floor;
  FuncNames["lle_X_srand"]        = lle_X_srand;
  FuncNames["lle_X_drand48"]      = lle_X_drand48;
  FuncNames["lle_X_srand48"]      = lle_X_srand48;
  FuncNames["lle_X_lrand48"]      = lle_X_lrand48;
  FuncNames["lle_X_sqrt"]         = lle_X_sqrt;
  FuncNames["lle_X_puts"]         = lle_X_puts;
  FuncNames["lle_X_printf"]       = lle_X_printf;
  FuncNames["lle_X_sprintf"]      = lle_X_sprintf;
  FuncNames["lle_X_sscanf"]       = lle_X_sscanf;
  FuncNames["lle_X_scanf"]        = lle_X_scanf;
  FuncNames["lle_i_clock"]        = lle_i_clock;

  FuncNames["lle_X_strcmp"]       = lle_X_strcmp;
  FuncNames["lle_X_strcat"]       = lle_X_strcat;
  FuncNames["lle_X_strcpy"]       = lle_X_strcpy;
  FuncNames["lle_X_strlen"]       = lle_X_strlen;
  FuncNames["lle_X_memset"]       = lle_X_memset;
  FuncNames["lle_X_memcpy"]       = lle_X_memcpy;

  FuncNames["lle_X_fopen"]        = lle_X_fopen;
  FuncNames["lle_X_fclose"]       = lle_X_fclose;
  FuncNames["lle_X_feof"]         = lle_X_feof;
  FuncNames["lle_X_fread"]        = lle_X_fread;
  FuncNames["lle_X_fwrite"]       = lle_X_fwrite;
  FuncNames["lle_X_fgets"]        = lle_X_fgets;
  FuncNames["lle_X_fflush"]       = lle_X_fflush;
  FuncNames["lle_X_fgetc"]        = lle_X_getc;
  FuncNames["lle_X_getc"]         = lle_X_getc;
  FuncNames["lle_X__IO_getc"]     = lle_X__IO_getc;
  FuncNames["lle_X_fputc"]        = lle_X_fputc;
  FuncNames["lle_X_ungetc"]       = lle_X_ungetc;
  FuncNames["lle_X_fprintf"]      = lle_X_fprintf;
  FuncNames["lle_X_freopen"]      = lle_X_freopen;
}
