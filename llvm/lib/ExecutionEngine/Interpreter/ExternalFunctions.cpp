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

typedef GenericValue (*ExFunc)(MethodType *, const vector<GenericValue> &);
static map<const Method *, ExFunc> Functions;

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
  const MethodType *MT = M->getType();
  for (unsigned i = 0; const Type *Ty = MT->getContainedType(i); ++i)
    ExtName += getTypeID(Ty);
  ExtName += "_" + M->getName();

  //cout << "Tried: '" << ExtName << "'\n";
  ExFunc FnPtr = (ExFunc)dlsym(RTLD_DEFAULT, ExtName.c_str());
  if (FnPtr == 0)  // Try calling a generic function... if it exists...
    FnPtr = (ExFunc)dlsym(RTLD_DEFAULT, ("lle_X_"+M->getName()).c_str());
  if (FnPtr != 0)
    Functions.insert(make_pair(M, FnPtr));  // Cache for later
  return FnPtr;
}

void Interpreter::callExternalMethod(Method *M,
				     const vector<GenericValue> &ArgVals) {
  // Do a lookup to see if the method is in our cache... this should just be a
  // defered annotation!
  map<const Method *, ExFunc>::iterator FI = Functions.find(M);
  ExFunc Fn = (FI == Functions.end()) ? lookupMethod(M) : FI->second;
  if (Fn == 0) {
    cout << "Tried to execute an unknown external method: "
	 << M->getType()->getDescription() << " " << M->getName() << endl;
    return;
  }

  // TODO: FIXME when types are not const!
  GenericValue Result = Fn(const_cast<MethodType*>(M->getType()), ArgVals);
  
  // Copy the result back into the result variable if we are not returning void.
  if (M->getReturnType() != Type::VoidTy) {
    CallInst *Caller = ECStack.back().Caller;
    if (Caller) {

    } else {
      // print it.
    }
  }
}


//===----------------------------------------------------------------------===//
//  Methods "exported" to the running application...
//
extern "C" {  // Don't add C++ manglings to llvm mangling :)

// Implement 'void print(X)' for every type...
GenericValue lle_X_print(MethodType *M, const vector<GenericValue> &ArgVals) {
  assert(ArgVals.size() == 1 && "generic print only takes one argument!");
  Interpreter::printValue(M->getParamTypes()[0], ArgVals[0]);
  return GenericValue();
}

// void "putchar"(sbyte)
GenericValue lle_Vb_putchar(MethodType *M, const vector<GenericValue> &Args) {
  cout << Args[0].SByteVal;
  return GenericValue();
}

} // End extern "C"
