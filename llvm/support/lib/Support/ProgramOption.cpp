// $Id$
//***************************************************************************
//
// File:
//    ProgramOption.C
//
// Purpose:
//    General representations for a program option.
//
// History:
//    08/08/95 - adve  - created in the dHPF compiler
//    11/26/96 - adve  - EvalOpt now returns #args consumed, or -1 for error
//    07/15/01 - vadve - Copied to LLVM system and modified
//
//**************************************************************************/

#include "llvm/Support/ProgramOption.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

//**************************************************************************/

StringOption::StringOption(const string &_argString,
			   const string &_helpMesg,
			   const string &_initValue, 
			   bool _append)
  : ProgramOption(_argString, _helpMesg),
    value(_initValue),
    append(_append) 
{}

int StringOption::EvalOpt(const string &optarg) {
  if (optarg == (char*) NULL)
    return -1;			// flag the error
  
  if (this->append)
    value += optarg;
  else
    value = optarg;
  
  optionSpecified = true;
  return 1;				// one additional argument consumed
}


//**************************************************************************/

FlagOption::FlagOption(const string &_argString,
		       const string &_helpMesg,
		       bool _initValue)
  : ProgramOption(_argString, _helpMesg, 0),
    value(_initValue)
{}

int FlagOption::EvalOpt(const string &optarg) {
  if (optarg == "0") {
    value = false;
    return 1;				// one additional argument consumed
  }
  else {
    value = true; 
    return 0;				// zero ... consumed
  }
}

//**************************************************************************/

RealValuedOption::RealValuedOption(const string &_argString,
				   const string &_helpMesg,
				   double _initValue)
  : ProgramOption(_argString, _helpMesg),
    value(_initValue)
{}

int RealValuedOption::EvalOpt(const string &optarg) {
  if (optarg == "") return -1;
    
  char* lastCharScanned = NULL;
  value = strtod(optarg.c_str(), &lastCharScanned);
  if (! (*lastCharScanned == '\0'))	// look for incorrect or partially
    return -1;				// correct numerical argument
  
  optionSpecified = true;
  return 1;
}

string RealValuedOption::GetTextValue() const {
  char buffer[40];
  sprintf(buffer, "%f", value);
  return buffer;
}

//**************************************************************************/

IntegerValuedOption::IntegerValuedOption(const string &_argString,
					 const string &_helpMesg,
					 int _initValue)
  : RealValuedOption(_argString, _helpMesg, (double) _initValue)
{}

int IntegerValuedOption::Value() const {
  double realValue = RealValuedOption::Value();
  assert(realValue == (int) realValue);
  return (int) realValue;
}

string IntegerValuedOption::GetTextValue() const {
  char buffer[40];
  sprintf(buffer, "%d", Value());
  return buffer;
}


