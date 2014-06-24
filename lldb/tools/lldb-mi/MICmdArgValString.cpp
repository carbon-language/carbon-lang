//===-- MICmdArgValString.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MICmdArgValString.cpp
//
// Overview:	CMICmdArgValString implementation.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

// In-house headers:
#include "MICmdArgValString.h"
#include "MICmdArgContext.h"

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdArgValString constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdArgValString::CMICmdArgValString( void )
:	m_bHandleQuotedString( false )
,	m_bAcceptNumbers( false )
,	m_bHandleDirPaths( false )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdArgValString constructor.
// Type:	Method.
// Args:	vbHandleQuotes		- (R) True = Parse a string surrounded by quotes spaces are not delimitors, false = only text up to next delimiting space character.
//			vbAcceptNumbers		- (R) True = Parse a string and accept as a number if number, false = numbers not recognised as string types. 
//			vbHandleDirPaths	- (R) True = Parse a string and accept as a file path if a path, false = file paths are not recognised as string types. 
// Return:	None.
// Throws:	None.
//--
CMICmdArgValString::CMICmdArgValString( const bool vbHandleQuotes, const bool vbAcceptNumbers, const bool vbHandleDirPaths )
:	m_bHandleQuotedString( vbHandleQuotes )
,	m_bAcceptNumbers( vbAcceptNumbers )
,	m_bHandleDirPaths( vbHandleDirPaths )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdArgValString constructor.
// Type:	Method.
// Args:	vrArgName		- (R) Argument's name to search by.
//			vbMandatory		- (R) True = Yes must be present, false = optional argument.
//			vbHandleByCmd	- (R) True = Command processes *this option, false = not handled.
//			vbHandleQuotes	- (R) True = Parse a string surrounded by quotes spaces are not delimitors, false = only text up to next delimiting space character. (Dflt = false)
//			vbAcceptNumbers	- (R) True = Parse a string and accept as a number if number, false = numbers not recognised as string types. (Dflt = false)
// Return:	None.
// Throws:	None.
//--
CMICmdArgValString::CMICmdArgValString( const CMIUtilString & vrArgName, const bool vbMandatory, const bool vbHandleByCmd, const bool vbHandleQuotes /* = false */, const bool vbAcceptNumbers  /* = false */ )
:	CMICmdArgValBaseTemplate( vrArgName, vbMandatory, vbHandleByCmd )
,	m_bHandleQuotedString( vbHandleQuotes )
,	m_bAcceptNumbers( vbAcceptNumbers )
,	m_bHandleDirPaths( false )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdArgValString destructor.
// Type:	Overidden.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdArgValString::~CMICmdArgValString( void )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	Parse the command's argument options string and try to extract the value *this
//			argument is looking for.
// Type:	Overridden.
// Args:	vrwArgContext	- (RW) The command's argument options string.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdArgValString::Validate( CMICmdArgContext & vrwArgContext )
{
	if( vrwArgContext.IsEmpty() )
		return MIstatus::success;

	if( m_bHandleQuotedString )
		return ValidateQuotedText( vrwArgContext );

	return ValidateSingleText( vrwArgContext );
}

//++ ------------------------------------------------------------------------------------
// Details:	Parse the command's argument options string and try to extract only the next
//			word delimited by the next space.
// Type:	Method.
// Args:	vrwArgContext	- (RW) The command's argument options string.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdArgValString::ValidateSingleText( CMICmdArgContext & vrwArgContext )
{
	if( vrwArgContext.GetNumberArgsPresent() == 1 )
	{
		const CMIUtilString & rArg( vrwArgContext.GetArgsLeftToParse() ); 
		if( IsStringArg( rArg ) )
		{
			m_bFound = true;
			m_bValid = true;
			m_argValue = rArg;
			vrwArgContext.RemoveArg( rArg );
			return MIstatus::success;
		}
		else
			return MIstatus::failure;
	}
	
	// More than one option...
	const CMIUtilString::VecString_t vecOptions( vrwArgContext.GetArgs() );
	CMIUtilString::VecString_t::const_iterator it = vecOptions.begin();
	while( it != vecOptions.end() )
	{
		const CMIUtilString & rArg( *it ); 
		if( IsStringArg( rArg )  )
		{
			m_bFound = true;
				
			if( vrwArgContext.RemoveArg( rArg ) )
			{
				m_bValid = true;
				m_argValue = rArg;
				return MIstatus::success;
			}
			else
				return MIstatus::failure;
		}
		
		// Next
		++it;
	}

	return MIstatus::failure;
}

//++ ------------------------------------------------------------------------------------
// Details:	Parse the command's argument options string and try to extract all the words
//			between quotes then delimited by the next space.
// Type:	Method.
// Args:	vrwArgContext	- (RW) The command's argument options string.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdArgValString::ValidateQuotedText( CMICmdArgContext & vrwArgContext )
{
	// CODETAG_QUOTEDTEXT_SIMILAR_CODE
	const CMIUtilString strOptions = vrwArgContext.GetArgsLeftToParse();
	const MIchar cQuote = '"';
	const MIint nPos = strOptions.find( cQuote );
	if( nPos == (MIint) std::string::npos )
		return ValidateSingleText( vrwArgContext );

	// Is one and only quote at end of the string
	if( nPos == (MIint)(strOptions.length() - 1) )
		return MIstatus::failure;

	// Quote must be the first character in the string or be preceeded by a space
	if( (nPos > 0) && (strOptions[ nPos - 1 ] != ' ' ) )
		return MIstatus::failure;
	
	// Need to find the other quote
	const MIint nPos2 = strOptions.find( cQuote, nPos + 1 );
	if( nPos2 == (MIint) std::string::npos )
		return MIstatus::failure;

	// Extract quoted text
	const CMIUtilString strQuotedTxt = strOptions.substr( nPos, nPos2 - nPos + 1 ).c_str();
	if( vrwArgContext.RemoveArg( strQuotedTxt ) )
	{
		m_bFound = true;
		m_bValid = true;
		m_argValue = strOptions.substr( nPos + 1, nPos2 - nPos - 1 ).c_str();;	
		return MIstatus::success;
	}

	return MIstatus::failure;
}

//++ ------------------------------------------------------------------------------------
// Details:	Examine the string and determine if it is a valid string type argument.
// Type:	Method.
// Args:	vrTxt	- (R) Some text.
// Return:	bool -	True = yes valid arg, false = no.
// Throws:	None.
//--
bool CMICmdArgValString::IsStringArg( const CMIUtilString & vrTxt ) const
{
	if( m_bHandleQuotedString )
		return IsStringArgQuotedText( vrTxt );
	
	return IsStringArgSingleText( vrTxt );
}

//++ ------------------------------------------------------------------------------------
// Details:	Examine the string and determine if it is a valid string type argument or 
//			option value. If the string looks like a long option, short option, a thread
//			group ID or just a number it is rejected as a string type value. There is an
//			option to allow the string to accept a number as a string type.
// Type:	Method.
// Args:	vrTxt	- (R) Some text.
// Return:	bool -	True = yes valid argument value, false = something else.
// Throws:	None.
//--
bool CMICmdArgValString::IsStringArgSingleText( const CMIUtilString & vrTxt ) const
{
	if( !m_bHandleDirPaths )
	{
		// Look for directory file paths, if found reject
		const bool bHavePosSlash = (vrTxt.find_first_of( "/" ) != std::string::npos);
		const bool bHaveBckSlash = (vrTxt.find_first_of( "\\" ) != std::string::npos);
		if( bHavePosSlash || bHaveBckSlash )
			return false;
	}

	// Look for --someLongOption, if found reject
	if( 0 == vrTxt.find( "--" ) )
		return false;
	
	// Look for -f type short options, if found reject
	if( (0 == vrTxt.find( "-" )) && (vrTxt.length() == 2 ) )
		return false;
		
	// Look for thread group i1 i2 i3...., if found reject
	if( (vrTxt.find( "i" ) == 0) && (::isdigit( vrTxt[ 1 ] )) )
		return false;
	
	// Look for numbers, if found reject
	if( !m_bAcceptNumbers && vrTxt.IsNumber() )
		return false;

	return true;
}

//++ ------------------------------------------------------------------------------------
// Details:	Examine the string and determine if it is a valid string type argument.
// Type:	Method.
// Args:	vrTxt	- (R) Some text.
// Return:	bool -	True = yes valid arg, false = no.
// Throws:	None.
//--
bool CMICmdArgValString::IsStringArgQuotedText( const CMIUtilString & vrTxt ) const
{
	// CODETAG_QUOTEDTEXT_SIMILAR_CODE
	const MIchar cQuote = '"';
	const MIint nPos = vrTxt.find( cQuote );
	if( nPos == (MIint) std::string::npos )
		return IsStringArgSingleText( vrTxt );

	// Is one and only quote at end of the string
	if( nPos == (MIint)(vrTxt.length() - 1) )
		return false;

	// Quote must be the first character in the string or be preceeded by a space
	if( (nPos > 0) && (vrTxt[ nPos - 1 ] != ' ' ) )
		return false;
	
	// Need to find the other quote
	const MIint nPos2 = vrTxt.find( cQuote, nPos + 1 );
	if( nPos2 == (MIint) std::string::npos )
		return false;

	return true;
}
