// $Id$ -*-c++-*-
//***************************************************************************
//
// File:
//    ProgramOption.h
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

#ifndef LLVM_SUPPORT_PROGRAMOPTION_H
#define LLVM_SUPPORT_PROGRAMOPTION_H

//************************** System Include Files **************************/

#include <string>

//*************************** User Include Files ***************************/

#include "llvm/Support/Unique.h"

//********************** Local Variable Definitions ************************/

class ProgramOption: public Unique {
public:
  /*ctor*/	ProgramOption	(const char* _argString,
				 const char* _helpMesg,
				 int _minExpectedArgs = 1)
			: optionSpecified(false),
			  argString(_argString),
			  helpMesg(_helpMesg),
			  minExpectedArgs(_minExpectedArgs) {}
    
  /*dtor*/ virtual ~ProgramOption() {}
  
  // Pure virtual function for an option with 0 or more arguments.
  // `optarg' points to the start of the next word in argv[].
  // It will be NULL if there are no more words.
  // The return value indicates the number of words of argv[] that
  // were consumed by EvalOpt and should be discarded.
  // A return value of -1 indicates an error.
  // 
  virtual int		EvalOpt		(const char* optarg) = 0;

  // Returns the value associated with the option as a human-readable
  // string.  The memory returned is allocated via `malloc'.
  virtual char*		GetTextValue	() const = 0;
  
  // Inline accessor functions for common option information
  // 
  bool			OptionSpecified	() const { return optionSpecified; }
  const char*		ArgString	() const { return argString.c_str(); }
  const char*		HelpMesg	() const { return helpMesg.c_str(); }
  int			MinExpectedArgs	() const { return minExpectedArgs; }
  
protected:
  bool	 optionSpecified;
  string argString;
  string helpMesg;
  int	 minExpectedArgs;
};

//**************************************************************************/

class StringOption : public ProgramOption {
public:
  /*ctor*/		StringOption	(const char* argString,
					 const char* helpMesg,
					 const char* initValue = "", 
					 bool append = false);
	// append = false:  EvalOpt will overwrite preexisting value 
	// append = true :  EvalOpt will append <optArg> to value 
  
  /*dtor*/ virtual	~StringOption	() {}
  
  virtual int		EvalOpt		(const char* optarg);
  
  const char*		Value		() const { return value.c_str(); }
  virtual char*		GetTextValue	() const { return strdup(Value()); }
  
protected:
  string value;
  bool	append; 
};

//**************************************************************************/

// -<flag_opt>	 sets the flag to TRUE
// -<flag_opt> 0 sets the flag to FALSE
// 
// To provide an actual argument (not option) of "0", mark the
// end of the options with "--" (see getopt(1)).

class FlagOption : public ProgramOption {
public:
  /*ctor*/		FlagOption	(const char* argString,
					 const char* helpMesg,
					 bool initValue = false);
    
  /*dtor*/ virtual	~FlagOption	() {}
    
  virtual int		EvalOpt		(const char* optarg);
    
  bool			Value		() const { return value; }
  virtual char*		GetTextValue	() const { return strdup(
						    value ? "true" : "false");}
private:
  bool	value;
};

//**************************************************************************/

class RealValuedOption : public ProgramOption {
public:
  /*ctor*/		RealValuedOption(const char* argString,
					 const char* helpMesg,
					 double initValue = 0.0);
  /*dtor*/ virtual	~RealValuedOption() {}
    
  virtual int		EvalOpt		(const char* optarg);
  
  double		Value		() const { return value; }
  virtual char*		GetTextValue	() const;
    
private:
  double	value;
};

//**************************************************************************/

class IntegerValuedOption : public RealValuedOption {
public:
  /*ctor*/		IntegerValuedOption(const char* argString,
					    const char* helpMesg,
					    int initValue = 0);
  /*ctor*/ virtual	~IntegerValuedOption() {}
  
  int			Value		() const;
  virtual char*		GetTextValue	() const;
};

//**************************************************************************/

#endif
