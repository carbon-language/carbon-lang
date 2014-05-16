//===-- MIUtilDateTimeStd.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MIUtilDateTimeStd.h
//
// Overview:	CMIUtilDateTimeStd interface.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

#pragma once

// Include compiler configuration
#include "MICmnConfig.h"

// In-house headers:
#include "MIUtilString.h"  

//++ ============================================================================
// Details:	MI common code utility class. Used to retrieve system local date
//			time.
// Gotchas:	None.
// Authors:	Aidan Dodds 10/03/2014.
// Changes:	None.
//--
class CMIUtilDateTimeStd
{
// Methods:
public:
	/* ctor */	 CMIUtilDateTimeStd( void );
	
	CMIUtilString 	GetDate( void ) const;
	CMIUtilString 	GetTime( void ) const;

// Overrideable:
public:
	// From CMICmnBase
	/* dtor */ virtual ~CMIUtilDateTimeStd( void );

// Methods:
private:
	bool GetDateTimeShort( CMIUtilString & vrwLocalDate, CMIUtilString & vrwLocalTime ) const;
};
