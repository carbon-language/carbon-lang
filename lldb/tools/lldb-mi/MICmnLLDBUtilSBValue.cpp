//===-- MICmnLLDBUtilSBValue.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MICmnLLDBUtilSBValue.cpp
//
// Overview:	CMICmnLLDBUtilSBValue implementation.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

// In-house headers:
#include "MICmnLLDBUtilSBValue.h"
#include "MIUtilString.h"
#include "MICmnLLDBDebugSessionInfo.h"

//++ ------------------------------------------------------------------------------------
// Details:	CMICmnLLDBUtilSBValue constructor.
// Type:	Method.
// Args:	vrValue				- (R) The LLDb value object.
//			vbHandleCharType	- (R) True = Yes return text molding to char type, 
//									  False = just return data.
// Return:	None.
// Throws:	None.
//--
CMICmnLLDBUtilSBValue::CMICmnLLDBUtilSBValue( const lldb::SBValue & vrValue, const bool vbHandleCharType /* = false */ )
:	m_rValue( const_cast< lldb::SBValue & >( vrValue ) )
,	m_pUnkwn( "??" )
,	m_bHandleCharType( vbHandleCharType )
{
	m_bValidSBValue = m_rValue.IsValid();
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmnLLDBUtilSBValue destructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmnLLDBUtilSBValue::~CMICmnLLDBUtilSBValue( void )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve from the LLDB SB Value object the name of the variable. If the name
//			is invalid (or the SBValue object invalid) then "??" is returned.
// Type:	Method.
// Args:	None.
// Return:	CMIUtilString	- Name of the variable or "??" for unknown.
// Throws:	None.
//--
CMIUtilString CMICmnLLDBUtilSBValue::GetName( void ) const
{
	const MIchar * pName = m_bValidSBValue ? m_rValue.GetName() : nullptr;
	const CMIUtilString text( (pName != nullptr) ? pName : m_pUnkwn );
				
	return text;
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve from the LLDB SB Value object the value of the variable described in
//			text. If the value is invalid (or the SBValue object invalid) then "??" is 
//			returned.
// Type:	Method.
// Args:	None.
// Return:	CMIUtilString	- Text description of the variable's value or "??".
// Throws:	None.
//--
CMIUtilString CMICmnLLDBUtilSBValue::GetValue( void ) const
{
	CMIUtilString text; 

	if( m_bHandleCharType && IsCharType() )
	{
		const lldb::addr_t addr = m_rValue.GetLoadAddress();
		text = CMIUtilString::Format( "0x%08x", addr );
		const CMIUtilString cString( GetValueCString() );
		if( !cString.empty() )
			text += CMIUtilString::Format( " %s", cString.c_str() );
	}
	else
	{
		const MIchar * pValue = m_bValidSBValue ? m_rValue.GetValue() : nullptr;
		text = (pValue != nullptr) ? pValue : m_pUnkwn;
	}
	
	return text;
}

//++ ------------------------------------------------------------------------------------
// Details:	If the LLDB SB Value object is a char type then form the text data string
//			otherwise return nothing. m_bHandleCharType must be true to return text data
//			if any.
// Type:	Method.
// Args:	None.
// Return:	CMIUtilString	- Text description of the variable's value.
// Throws:	None.
//--
CMIUtilString CMICmnLLDBUtilSBValue::GetValueCString( void ) const
{
	CMIUtilString text; 

	if( m_bHandleCharType && IsCharType() )
	{
		text = ReadCStringFromHostMemory( m_rValue );
	}
	
	return text;
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve the flag stating whether this value object is a char type or some
//			other type. Char type can be signed or unsigned.
// Type:	Method.
// Args:	None.
// Return:	bool	- True = Yes is a char type, false = some other type.
// Throws:	None.
//--
bool CMICmnLLDBUtilSBValue::IsCharType( void ) const
{
	const MIchar * pName = m_rValue.GetName(); MIunused( pName );
	const lldb::BasicType eType = m_rValue.GetType().GetBasicType();
	return ((eType == lldb::eBasicTypeChar) || 
		    (eType == lldb::eBasicTypeSignedChar) || 
			(eType == lldb::eBasicTypeUnsignedChar) );
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve the flag stating whether any child value object of *this object is a 
//			char type or some other type. Returns false if there are not children. Char 
//			type can be signed or unsigned.
// Type:	Method.
// Args:	None.
// Return:	bool	- True = Yes is a char type, false = some other type.
// Throws:	None.
//--
bool CMICmnLLDBUtilSBValue::IsChildCharType( void ) const
{
	const MIuint nChildren = m_rValue.GetNumChildren();
	
	// Is it a basic type
	if( nChildren == 0 )
		return false;

	// Is it a composite type
	if( nChildren > 1 )
		return false;

	lldb::SBValue member = m_rValue.GetChildAtIndex( 0 );
	const CMICmnLLDBUtilSBValue utilValue( member );
	return utilValue.IsCharType();			
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve the C string data for a child of char type (one and only child) for
//			the parent value object. If the child is not a char type or the parent has
//			more than one child then an empty string is returned. Char type can be 
//			signed or unsigned.
// Type:	Method.
// Args:	None.
// Return:	CMIUtilString	- Text description of the variable's value.
// Throws:	None.
//--
CMIUtilString CMICmnLLDBUtilSBValue::GetChildValueCString( void ) const
{
	CMIUtilString text;
	const MIuint nChildren = m_rValue.GetNumChildren();
	
	// Is it a basic type
	if( nChildren == 0 )
		return text;

	// Is it a composite type
	if( nChildren > 1 )
		return text;

	lldb::SBValue member = m_rValue.GetChildAtIndex( 0 );
	const CMICmnLLDBUtilSBValue utilValue( member );
	if( m_bHandleCharType && utilValue.IsCharType() )
	{
		text = ReadCStringFromHostMemory( member );
	}

	return text;
}
	
//++ ------------------------------------------------------------------------------------
// Details:	Retrieve the C string data of value object by read the memory where the 
//			variable is held. 
// Type:	Method.
// Args:	vrValueObj	- (R) LLDB SBValue variable object.
// Return:	CMIUtilString	- Text description of the variable's value.
// Throws:	None.
//--
CMIUtilString CMICmnLLDBUtilSBValue::ReadCStringFromHostMemory( const lldb::SBValue & vrValueObj ) const
{
	CMIUtilString text;

	lldb::SBValue & rValue = const_cast< lldb::SBValue & >( vrValueObj );
	const lldb::addr_t addr = rValue.GetLoadAddress();
	CMICmnLLDBDebugSessionInfo & rSessionInfo( CMICmnLLDBDebugSessionInfo::Instance() ); 
	const MIuint nBytes( 128 );
	const MIchar * pBufferMemory = new MIchar[ nBytes ];
	lldb::SBError error;
	const MIuint64 nReadBytes = rSessionInfo.m_lldbProcess.ReadMemory( addr, (void *) pBufferMemory, nBytes, error ); MIunused( nReadBytes );
	text = CMIUtilString::Format( "\\\"%s\\\"", pBufferMemory );
	delete [] pBufferMemory;
	
	return text;
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve the state of the value object's name. 
// Type:	Method.
// Args:	None.
// Return:	bool	- True = yes name is indeterminate, false = name is valid.
// Throws:	None.
//--
bool CMICmnLLDBUtilSBValue::IsNameUnknown( void ) const
{
	const CMIUtilString name( GetName() );
	return (name == m_pUnkwn);
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve the state of the value object's value data. 
// Type:	Method.
// Args:	None.
// Return:	bool	- True = yes value is indeterminate, false = value valid.
// Throws:	None.
//--
bool CMICmnLLDBUtilSBValue::IsValueUnknown( void ) const
{
	const CMIUtilString value( GetValue() );
	return (value == m_pUnkwn);
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve the value object's type name if valid.
// Type:	Method.
// Args:	None.
// Return:	CMIUtilString	- The type name or "??".
// Throws:	None.
//--
CMIUtilString CMICmnLLDBUtilSBValue::GetTypeName( void ) const
{
	const MIchar * pName = m_bValidSBValue ? m_rValue.GetTypeName() : nullptr;
	const CMIUtilString text( (pName != nullptr) ? pName : m_pUnkwn );
				
	return text;
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve the value object's display type name if valid. 
// Type:	Method.
// Args:	None.
// Return:	CMIUtilString	- The type name or "??".
// Throws:	None.
//--
CMIUtilString CMICmnLLDBUtilSBValue::GetTypeNameDisplay( void ) const
{
	const MIchar * pName = m_bValidSBValue ? m_rValue.GetDisplayTypeName() : nullptr;
	const CMIUtilString text( (pName != nullptr) ? pName : m_pUnkwn );
				
	return text;
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve whether the value object's is valid or not. 
// Type:	Method.
// Args:	None.
// Return:	bool	- True = valid, false = not valid.
// Throws:	None.
//--
bool CMICmnLLDBUtilSBValue::IsValid( void ) const
{
	return m_bValidSBValue;
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve the value object' has a name. A value object can be valid but still
//			have no name which suggest it is not a variable. 
// Type:	Method.
// Args:	None.
// Return:	bool	- True = valid, false = not valid.
// Throws:	None.
//--
bool CMICmnLLDBUtilSBValue::HasName( void ) const
{
	bool bHasAName = false;

	const MIchar * pName = m_bValidSBValue ? m_rValue.GetDisplayTypeName() : nullptr;
	if( pName != nullptr )
	{
		bHasAName = (CMIUtilString( pName ).length() > 0);
	}

	return bHasAName;
}

//++ ------------------------------------------------------------------------------------
// Details:	Determine if the value object' respresents a LLDB variable i.e. "$0".
// Type:	Method.
// Args:	None.
// Return:	bool	- True = Yes LLDB variable, false = no.
// Throws:	None.
//--
bool CMICmnLLDBUtilSBValue::IsLLDBVariable( void ) const
{
	return (GetName().at( 0 ) == '$' );
}
	

