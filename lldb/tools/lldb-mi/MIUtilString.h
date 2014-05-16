//===-- MIUtilString.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MIUtilString.h
//
// Overview:	CMIUtilString interface.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

#pragma once

// Third party headers:
#include <string>
#include <vector>

// In-house headers:
#include "MIDataTypes.h"  

//++ ============================================================================
// Details:	MI common code utility class. Used to help handle text. 
//			Derived from std::string
// Gotchas:	None.
// Authors:	Illya Rudkin 02/02/2014.
// Changes:	None.
//--
class CMIUtilString : public std::string
{
// Typdefs:
public:
	typedef std::vector< CMIUtilString >	VecString_t;

// Static method:
public:
	static CMIUtilString	Format( const CMIUtilString & vrFormating, ... );
	static CMIUtilString	FormatBinary( const MIuint64 vnDecimal );
	static CMIUtilString	FormatValist( const CMIUtilString & vrFormating, va_list vArgs );
	static bool				IsAllValidAlphaAndNumeric( const MIchar & vrText );
	static bool             Compare( const CMIUtilString & vrLhs, const CMIUtilString & vrRhs );

// Methods:
public:
	/* ctor */	CMIUtilString( void );
	/* ctor */	CMIUtilString( const MIchar * vpData );
	/* ctor */	CMIUtilString( const MIchar * const * vpData );
	//
	MIuint			Split( const CMIUtilString & vDelimiter, VecString_t & vwVecSplits ) const;
	CMIUtilString	Trim( void ) const;
	CMIUtilString	Trim( const MIchar vChar ) const;
	CMIUtilString	StripCREndOfLine( void ) const;
	CMIUtilString	StripCRAll( void ) const;
	CMIUtilString	FindAndReplace( const CMIUtilString & vFind, const CMIUtilString & vReplaceWith ) const;
	bool			IsNumber( void ) const;
	bool			ExtractNumber( MIint64 & vwrNumber ) const;

// Overrideable:
public:
	/* dtor */ virtual ~CMIUtilString( void );
	
// Static method:
private:
	static CMIUtilString	FormatPriv( const CMIUtilString & vrFormat, va_list vArgs );

// Methods:
private:
	bool	ExtractNumberFromHexadecimal( MIint64 & vwrNumber ) const;
};
