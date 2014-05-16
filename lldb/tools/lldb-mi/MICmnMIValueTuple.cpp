//===-- Platform.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MICmnMIValueTuple.h
//
// Overview:	CMICmnMIValueTuple implementation.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

// In-house headers:
#include "MICmnMIValueTuple.h"

//++ ------------------------------------------------------------------------------------
// Details:	CMICmnMIValueTuple constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmnMIValueTuple::CMICmnMIValueTuple( void )
:	m_bSpaceAfterComma( false )
{
	m_strValue = "{}";
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmnMIValueTuple constructor.
// Type:	Method.
// Args:	vResult	- (R) MI result object.
// Return:	None.
// Throws:	None.
//--
CMICmnMIValueTuple::CMICmnMIValueTuple( const CMICmnMIValueResult & vResult )
:	m_bSpaceAfterComma( false )
{
	m_strValue = vResult.GetString();
	BuildTuple();
	m_bJustConstructed = false;
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmnMIValueTuple constructor.
// Type:	Method.
// Args:	vResult			- (R) MI result object.
//			vbUseSpacing	- (R) True = put space seperators into the string, false = no spaces used.
// Return:	None.
// Throws:	None.
//--
CMICmnMIValueTuple::CMICmnMIValueTuple( const CMICmnMIValueResult & vResult, const bool vbUseSpacing )
:	m_bSpaceAfterComma( vbUseSpacing )
{
	m_strValue = vResult.GetString();
	BuildTuple();
	m_bJustConstructed = false;
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmnMIValueTuple destructor.
// Type:	Overrideable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmnMIValueTuple::~CMICmnMIValueTuple( void )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	Build the result value's mandatory data part, one tuple
// Type:	Method.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnMIValueTuple::BuildTuple( void )
{
	const char * pFormat = "{%s}";
	m_strValue = CMIUtilString::Format( pFormat, m_strValue.c_str() );
		
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Add another MI result object to  the value's list of tuples. 
// Type:	Method.
// Args:	vResult	- (R) The MI result object.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnMIValueTuple::BuildTuple( const CMICmnMIValueResult & vResult )
{
	// Clear out the default "<Invalid>" text
	if( m_bJustConstructed )
	{
		m_bJustConstructed = false;
		m_strValue = vResult.GetString();
		return BuildTuple();
	}

	// ToDo: Fix this fudge with std::string cause CMIUtilString does not have a copy constructor
	std::string str = m_strValue;
	if( str[ 0 ] == '{' )
	{
		str = str.substr( 1, m_strValue.size() - 1 );
	}
	if( str[ str.size() - 1 ] == '}' )
	{
		str = str.substr( 0, m_strValue.size() - 2 );
	}
	m_strValue = str.c_str();

	const char * pFormat = m_bSpaceAfterComma ? "{%s, %s}" : "{%s,%s}";
	m_strValue = CMIUtilString::Format( pFormat, m_strValue.c_str(), vResult.GetString().c_str() );

	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Add another MI value object to  the value list's of list is values.
//			Only values objects can be added to a list of values otherwise this function 
//			will return MIstatus::failure.
// Type:	Method.
// Args:	vValue	- (R) The MI value object.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnMIValueTuple::Add( const CMICmnMIValueResult & vResult )
{
	return BuildTuple( vResult );
}

//++ ------------------------------------------------------------------------------------
// Details:	Add another MI value object to  the value list's of list is values.
//			Only values objects can be added to a list of values otherwise this function 
//			will return MIstatus::failure.
// Type:	Method.
// Args:	vValue			- (R) The MI value object.
//			vbUseSpacing	- (R) True = put space seperators into the string, false = no spaces used.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnMIValueTuple::Add( const CMICmnMIValueResult & vResult, const bool vbUseSpacing )
{
	m_bSpaceAfterComma = vbUseSpacing;
	return BuildTuple( vResult );
}

