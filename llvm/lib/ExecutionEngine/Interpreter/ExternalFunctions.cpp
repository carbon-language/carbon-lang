//===-- ExternalFunctions.cpp - Implement External Functions --------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
//  This file contains both code to deal with invoking "external" functions, but
//  also contains code that implements "exported" external functions.
//
//  External functions in the interpreter are implemented by 
//  using the system's dynamic loader to look up the address of the function
//  we want to invoke.  If a function is found, then one of the
//  many lle_* wrapper functions in this file will translate its arguments from
//  GenericValues to the types the function is actually expecting, before the
//  function is called.
//
//===----------------------------------------------------------------------===//

#include "Interpreter.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/SymbolTable.h"
#include "llvm/Target/TargetData.h"
#include "Support/DynamicLinker.h"
#include "Config/dlfcn.h"
#include "Config/link.h"
#include <cmath>
#include <csignal>
#include <map>
using std::vector;

namespace llvm {

typedef GenericValue (*ExFunc)(FunctionType *, const vector<GenericValue> &);
static std::map<const Function *, ExFunc> Functions;
static std::map<std::string, ExFunc> FuncNames;

static Interpreter *TheInterpreter;

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

static ExFunc lookupFunction(const Function *F) {
  // Function not found, look it up... start by figuring out what the
  // composite function name should be.
  std::string ExtName = "lle_";
  const FunctionType *FT = F->getFunctionType();
  for (unsigned i = 0, e = FT->getNumContainedTypes(); i != e; ++i)
    ExtName += getTypeID(FT->getContainedType(i));
  ExtName += "_" + F->getName();

  ExFunc FnPtr = FuncNames[ExtName];
  if (FnPtr == 0)
    FnPtr = (ExFunc)GetAddressOfSymbol(ExtName);
  if (FnPtr == 0)
    FnPtr = FuncNames["lle_X_"+F->getName()];
  if (FnPtr == 0)  // Try calling a generic function... if it exists...
    FnPtr = (ExFunc)GetAddressOfSymbol(("lle_X_"+F->getName()).c_str());
  if (FnPtr != 0)
    Functions.insert(std::make_pair(F, FnPtr));  // Cache for later
  return FnPtr;
}

GenericValue Interpreter::callExternalFunction(Function *M,
                                     const std::vector<GenericValue> &ArgVals) {
  TheInterpreter = this;

  // Do a lookup to see if the function is in our cache... this should just be a
  // deferred annotation!
  std::map<const Function *, ExFunc>::iterator FI = Functions.find(M);
  ExFunc Fn = (FI == Functions.end()) ? lookupFunction(M) : FI->second;
  if (Fn == 0) {
    std::cout << "Tried to execute an unknown external function: "
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

// void putchar(sbyte)
GenericValue lle_Vb_putchar(FunctionType *M, const vector<GenericValue> &Args) {
  std::cout << Args[0].SByteVal;
  return GenericValue();
}

// int putchar(int)
GenericValue lle_ii_putchar(FunctionType *M, const vector<GenericValue> &Args) {
  std::cout << ((char)Args[0].IntVal) << std::flush;
  return Args[0];
}

// void putchar(ubyte)
GenericValue lle_VB_putchar(FunctionType *M, const vector<GenericValue> &Args) {
  std::cout << Args[0].SByteVal << std::flush;
  return Args[0];
}

// void atexit(Function*)
GenericValue lle_X_atexit(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  TheInterpreter->addAtExitHandler((Function*)GVTOP(Args[0]));
  GenericValue GV;
  GV.IntVal = 0;
  return GV;
}

// void exit(int)
GenericValue lle_X_exit(FunctionType *M, const vector<GenericValue> &Args) {
  TheInterpreter->exitCalled(Args[0]);
  return GenericValue();
}

// void abort(void)
GenericValue lle_X_abort(FunctionType *M, const vector<GenericValue> &Args) {
  raise (SIGABRT);
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
          if (HowLong == 1 &&
              TheInterpreter->getModule().getPointerSize()==Module::Pointer64 &&
              sizeof(long) < sizeof(long long)) {
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
      default:  std::cout << "<unknown printf code '" << *FmtStr << "'!>";
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
  std::cout << Buffer;
  return GV;
}

static void ByteswapSCANFResults(const char *Fmt, void *Arg0, void *Arg1,
                                 void *Arg2, void *Arg3, void *Arg4, void *Arg5,
                                 void *Arg6, void *Arg7, void *Arg8) {
  void *Args[] = { Arg0, Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7, Arg8, 0 };

  // Loop over the format string, munging read values as appropriate (performs
  // byteswaps as necessary).
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

// char *strdup(const char *src);
GenericValue lle_X_strdup(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  return PTOGV(strdup((char*)GVTOP(Args[0])));
}

// char *__strdup(const char *src);
GenericValue lle_X___strdup(FunctionType *M, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  return PTOGV(strdup((char*)GVTOP(Args[0])));
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

//===----------------------------------------------------------------------===//
// LLVM Intrinsic Functions...
//===----------------------------------------------------------------------===//

// <va_list> llvm.va_start() - Implement the va_start operation...
GenericValue llvm_va_start(FunctionType *F, const vector<GenericValue> &Args) {
  assert(Args.size() == 0);
  return TheInterpreter->getFirstVarArg();
}

// void llvm.va_end(<va_list> *) - Implement the va_end operation...
GenericValue llvm_va_end(FunctionType *F, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  return GenericValue();    // Noop!
}

// <va_list> llvm.va_copy(<va_list>) - Implement the va_copy operation...
GenericValue llvm_va_copy(FunctionType *F, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  return Args[0];
}

} // End extern "C"


void Interpreter::initializeExternalFunctions() {
  FuncNames["lle_Vb_putchar"]     = lle_Vb_putchar;
  FuncNames["lle_ii_putchar"]     = lle_ii_putchar;
  FuncNames["lle_VB_putchar"]     = lle_VB_putchar;
  FuncNames["lle_X_exit"]         = lle_X_exit;
  FuncNames["lle_X_abort"]        = lle_X_abort;
  FuncNames["lle_X_malloc"]       = lle_X_malloc;
  FuncNames["lle_X_calloc"]       = lle_X_calloc;
  FuncNames["lle_X_free"]         = lle_X_free;
  FuncNames["lle_X_atoi"]         = lle_X_atoi;
  FuncNames["lle_X_pow"]          = lle_X_pow;
  FuncNames["lle_X_exp"]          = lle_X_exp;
  FuncNames["lle_X_log"]          = lle_X_log;
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
  FuncNames["lle_X___strdup"]     = lle_X___strdup;
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

  FuncNames["lle_X_llvm.va_start"]= llvm_va_start;
  FuncNames["lle_X_llvm.va_end"]  = llvm_va_end;
  FuncNames["lle_X_llvm.va_copy"] = llvm_va_copy;
}

} // End llvm namespace
