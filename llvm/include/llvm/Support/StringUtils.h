// $Id$ -*-c++-*-
//***************************************************************************
//
// File:
//	ProgramOptions.h
//
// Purpose:
//	A representation of options for any program.
//
// History:
//	08/08/95 - adve  - Created in the dHPF compiler
//      10/10/96 - mpal, dbaker - converted to const member functions.
//      10/19/96 - meven - slightly changed interface to accomodate 
//                         arguments other than -X type options
//	07/15/01 - vadve - Copied to LLVM system and modified
//
//**************************************************************************/

#ifndef LLVM_SUPPORT_PROGRAMOPTIONS_h
#define LLVM_SUPPORT_PROGRAMOPTIONS_h

//************************** System Include Files **************************/

#include <string>
#include <hash_map>

//*************************** User Include Files ***************************/

#include "llvm/Support/Unique.h"
class ProgramOption;

//************************ Forward Declarations ****************************/

//***************************** String Functions ****************************/

struct eqstr
{
  bool operator()(const char* s1, const char* s2) const
  {
    return strcmp(s1, s2) == 0;
  }
};

//***************************** String Classes *****************************/

template <class DataType>
class StringMap:
  public hash_map<const char*, DataType, hash<const char*>, eqstr>
{
public:
  typedef hash_map<const char*, DataType, hash<const char*>, eqstr>::iterator
	iterator;
  typedef hash_map<const char*, DataType, hash<const char*>, eqstr>::const_iterator
	const_iterator;
  
public:
  DataType*		query(const char* _key)
  {
    hash_map<const char*, DataType, hash<const char*>, eqstr>::iterator
      hashPair = this->find(_key);
    return (hashPair == this->end())? NULL : & (*hashPair).second;
  }
  
  const DataType*	query(const char* _key) const
  {
    hash_map<const char*, DataType, hash<const char*>, eqstr>::const_iterator
      hashPair = this->find(_key);
    return (hashPair == this->end())? NULL : & (*hashPair).second;
  }
};

//**************************************************************************/

#endif

