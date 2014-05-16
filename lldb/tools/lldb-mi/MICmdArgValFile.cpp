//===-- MICmdArgValFile.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MICmdArgValFile.cpp
//
// Overview:	CMICmdArgValFile implementation.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

// In-house headers:
#include "MICmdArgValFile.h"
#include "MICmdArgContext.h"

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdArgValFile constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdArgValFile::CMICmdArgValFile( void )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdArgValFile constructor.
// Type:	Method.
// Args:	vrArgName		- (R) Argument's name to search by.
//			vbMandatory		- (R) True = Yes must be present, false = optional argument.
//			vbHandleByCmd	- (R) True = Command processes *this option, false = not handled.
// Return:	None.
// Throws:	None.
//--
CMICmdArgValFile::CMICmdArgValFile( const CMIUtilString & vrArgName, const bool vbMandatory, const bool vbHandleByCmd )
:	CMICmdArgValBaseTemplate( vrArgName, vbMandatory, vbHandleByCmd )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdArgValFile destructor.
// Type:	Overidden.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdArgValFile::~CMICmdArgValFile( void )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	Parse the command's argument options string and try to extract the value *this
//			argument is looking for.
// Type:	Overridden.
// Args:	vwArgContext	- (R) The command's argument options string.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdArgValFile::Validate( CMICmdArgContext & vwArgContext )
{
	if( vwArgContext.IsEmpty() )
		return MIstatus::success;

	// The GDB/MI spec suggests there is only parameter
	if( vwArgContext.GetNumberArgsPresent() == 1 )
	{
		const CMIUtilString & rFile( vwArgContext.GetArgsLeftToParse() ); 
		if( IsFilePath( rFile ) )
		{
			m_bFound = true;
			m_bValid = true;
			m_argValue = rFile;
			vwArgContext.RemoveArg( rFile );
			return MIstatus::success;
		}
		else
			return MIstatus::failure;
	}
	
	// In reality there are more than one option,  if so the file option 
	// is the last one (don't handle that here - find the best looking one)
	const CMIUtilString::VecString_t vecOptions( vwArgContext.GetArgs() );
	CMIUtilString::VecString_t::const_iterator it = vecOptions.begin();
	while( it != vecOptions.end() )
	{
		const CMIUtilString & rTxt( *it ); 
		if( IsFilePath( rTxt ) )
		{
			m_bFound = true;
				
			if( vwArgContext.RemoveArg( rTxt ) )
			{
				m_bValid = true;
				m_argValue = rTxt;
				return MIstatus::success;
			}
			else
				return MIstatus::success;
		}
		
		// Next
		++it;
	}

	return MIstatus::failure;
}

//++ ------------------------------------------------------------------------------------
// Details:	Given some text extract the file name path from it.
// Type:	Method.
// Args:	vrTxt	- (R) The text to extract the file name path from.
// Return:	CMIUtilString -	File name and or path.
// Throws:	None.
//--
CMIUtilString CMICmdArgValFile::GetFileNamePath( const CMIUtilString & vrTxt ) const
{
	return vrTxt;
}

//++ ------------------------------------------------------------------------------------
// Details:	Examine the string and determine if it is a valid file name path.
// Type:	Method.
// Args:	vrFileNamePath	- (R) File's name and directory path.
// Return:	bool -	True = yes valid file path, false = no.
// Throws:	None.
//--
bool CMICmdArgValFile::IsFilePath( const CMIUtilString & vrFileNamePath ) const
{
	const bool bHavePosSlash = (vrFileNamePath.find_first_of( "/" ) != std::string::npos);
	const bool bHaveBckSlash = (vrFileNamePath.find_first_of( "\\" ) != std::string::npos);
	
	// Look for --someLongOption
	MIint nPos = vrFileNamePath.find_first_of( "--" );
	const bool bLong = (nPos == 0);
	if( bLong )
		return false;
	
	// Look for -f type short parameters
	nPos = vrFileNamePath.find_first_of( "-" );
	const bool bShort = (nPos == 0);
	if( bShort )
		return false;
	
	// Look for i1 i2 i3....
	nPos = vrFileNamePath.find_first_of( "i" );
	const bool bFoundI1 = ((nPos == 0) && (::isdigit( vrFileNamePath[ 1 ] )) );
	if( bFoundI1 )
		return false;
	
	const bool bValidChars = CMIUtilString::IsAllValidAlphaAndNumeric( *vrFileNamePath.c_str() );
	if( bValidChars || bHavePosSlash || bHaveBckSlash )
		return true;

	return false;
}
