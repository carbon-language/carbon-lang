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

#include "llvm/Support/Unique.h"
#include <string>


class ProgramOption: public Unique {
public:
  /*ctor*/	ProgramOption	(const string &_argString,
				 const string &_helpMesg,
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
  virtual int		EvalOpt		(const string &) = 0;

  // Returns the value associated with the option as a human-readable
  // string.
  virtual string GetTextValue	() const = 0;
  
  // Inline accessor functions for common option information
  // 
  bool			OptionSpecified	() const { return optionSpecified; }
  const string		ArgString	() const { return argString; }
  const string 		HelpMesg	() const { return helpMesg; }
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
  /*ctor*/		StringOption	(const string &argString,
					 const string &helpMesg,
					 const string &initValue = "", 
					 bool append = false);
	// append = false:  EvalOpt will overwrite preexisting value 
	// append = true :  EvalOpt will append <optArg> to value 
  
  /*dtor*/ virtual	~StringOption	() {}
  
  virtual int		EvalOpt		(const string &optarg);
  
  const string &Value() const { return value; }
  virtual string GetTextValue() const { return value; }
  
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
  FlagOption(const string &argString, const string &helpMesg,
	     bool initValue = false);
    
  virtual ~FlagOption() {}
    
  virtual int		EvalOpt		(const string &optarg);
    
  bool			Value		() const { return value; }
  virtual string GetTextValue() const { return value ? "true" : "false";}
private:
  bool value;
};

//**************************************************************************/

class RealValuedOption : public ProgramOption {
public:
  /*ctor*/		RealValuedOption(const string &argString,
					 const string &helpMesg,
					 double initValue = 0.0);
  /*dtor*/ virtual	~RealValuedOption() {}
    
  virtual int		EvalOpt		(const string &optarg);
  
  double		Value		() const { return value; }
  virtual string GetTextValue() const;
    
private:
  double value;
};

//**************************************************************************/

class IntegerValuedOption : public RealValuedOption {
public:
  /*ctor*/		IntegerValuedOption(const string &argString,
					    const string &helpMesg,
					    int initValue = 0);
  /*ctor*/ virtual	~IntegerValuedOption() {}
  
  int Value() const;
  virtual string GetTextValue() const;
};

//**************************************************************************/

#endif
