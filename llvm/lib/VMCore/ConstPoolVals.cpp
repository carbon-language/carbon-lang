//===-- llvm/ConstPoolVals.cpp - Implement Constant Value nodes -*- C++ -*--=//
//
// This file implements the ConstPoolVal class and associated functions.
//
//===---------------------------------------------------------------------===//

#include "llvm/Type.h"
#include "llvm/Value.h"
#include "llvm/ConstPoolVals.h"


//===--------------------------------------------------------------------------
// External functions
//

// Convenience functions to get the value of an integer constant, for an
// appropriate integer or non-integer type that can be held in an integer.
// The type of the argument must be the following:
//   GetSignedIntConstantValue:   signed integer or bool
//   GetUnsignedIntConstantValue: unsigned integer, bool, or pointer
//   GetConstantValueAsSignedInt: any of the above, but the value
//				  must fit into a int64_t.
// 
// isValidConstant is set to true if a valid constant was found.
// 

int64_t
GetSignedIntConstantValue(const Value* val, bool& isValidConstant)
{
  int64_t intValue = 0;
  isValidConstant = false;
  
  if (val->getValueType() == Value::ConstantVal)
    {
      switch(val->getType()->getPrimitiveID())
	{
	case Type::BoolTyID:
	  intValue = ((ConstPoolBool*) val)->getValue()? 1 : 0;
	  isValidConstant = true;
	  break;
	case Type::SByteTyID:
	case Type::ShortTyID:
	case Type::IntTyID:
	case Type::LongTyID:
	  intValue = ((ConstPoolSInt*) val)->getValue();
	  isValidConstant = true;
	  break;
	default:
	  break;
	}
    }
  
  return intValue;
}

uint64_t
GetUnsignedIntConstantValue(const Value* val, bool& isValidConstant)
{
  uint64_t intValue = 0;
  isValidConstant = false;
  
  if (val->getValueType() == Value::ConstantVal)
    {
      switch(val->getType()->getPrimitiveID())
	{
	case Type::BoolTyID:
	  intValue = ((ConstPoolBool*) val)->getValue()? 1 : 0;
	  isValidConstant = true;
	  break;
	case Type::UByteTyID:
	case Type::UShortTyID:
	case Type::UIntTyID:
	case Type::ULongTyID:
	case Type::PointerTyID:
	  intValue = ((ConstPoolUInt*) val)->getValue();
	  isValidConstant = true;
	  break;
	default:
	  break;
	}
    }
  
  return intValue;
}


int64_t
GetConstantValueAsSignedInt(const Value* val, bool& isValidConstant)
{
  int64_t intValue = 0;
  
  if (val->getType()->isSigned())
    {
      intValue = GetSignedIntConstantValue(val, isValidConstant);
    }
  else				// non-numeric types will fall here
    {
      uint64_t uintValue = GetUnsignedIntConstantValue(val, isValidConstant);
      if (isValidConstant && uintValue < INT64_MAX)	// then safe to cast to signed
	intValue = (int64_t) uintValue;
      else 
	isValidConstant = false;
    }
  
  return intValue;
}

//===--------------------------------------------------------------------------
